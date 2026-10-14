"""Independent one-shot federated OPCM with common-origin local experts.

Collect exactly one multi-epoch upload from EVERY client before merging. Reuse
the project's unchanged OPCMLoRA algebra in an offline deterministic order.
Only one public server mutation exists; no merged model returns to clients.
LoRA rank truncation/null-space treatment remain disclosed project adaptations.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import (
    ClientContext,
    MethodCapabilities,
    ScheduledEvent,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.methods.base import Method
from afl_vlm.methods.continual_merge.method import OPCMLoRA
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, clone_state, state_to_device, subtract


@register_method("opcm_oneshot")
class OPCMOneShot(Method):
    name = "opcm_oneshot"
    capabilities = MethodCapabilities(
        mode="synchronous",
        requires_one_shot=True,
        requires_buffer=True,
        requires_task_id=True,
        requires_local_state=True,
    )
    allowed_params = {"merge_weight", "retain", "eps", "merge_order", "merge_seed"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.merge_order = str(self.params.get("merge_order", "seeded_random"))
        if self.merge_order not in {"seeded_random", "client_id", "arrival"}:
            raise ValueError(
                "OPCM one-shot merge_order must be seeded_random, client_id or arrival"
            )
        self.merge_seed = self.params.get("merge_seed", 42)
        if isinstance(self.merge_seed, bool) or not isinstance(self.merge_seed, int):
            raise ValueError("OPCM one-shot merge_seed must be an integer")
        self.merger = OPCMLoRA(
            {
                "merge_weight": self.params.get("merge_weight", 1.0),
                "retain": self.params.get("retain", 0.5),
                "eps": self.params.get("eps", 1e-8),
                "staleness": {"type": "constant"},
            }
        )
        self.origin: LoRAState | None = None
        self.clients: dict[str, tuple[str, str, int]] = {}
        self.experts: dict[str, LoRAState] = {}
        self.upload_records: dict[str, dict[str, Any]] = {}
        self.finished = False
        self.merge_trace: list[dict[str, Any]] = []
        self.buffer_aggregations = 0
        self.occupancy_sum = 0
        self.arrivals = 0

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        # Same LoRA-only checks as OPCM; does not change model/trainable parameters.
        self.merger.configure_model(model, clients)

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        if self.origin is not None:
            raise RuntimeError("OPCM one-shot server is already configured")
        if not clients or len({item.id for item in clients}) != len(clients):
            raise ValueError("OPCM one-shot requires a nonempty unique client population")
        self.merger.configure_server(initial_state, clients)
        self.origin = clone_state(initial_state)
        self.clients = {item.id: (item.task, item.dataset, item.num_samples) for item in clients}

    def configure_schedule(self, events: list[ScheduledEvent]) -> None:
        if not self.clients and not events:
            return
        if len(events) != len(self.clients) or {e.client_id for e in events} != set(self.clients):
            raise ValueError(
                "OPCM one-shot Plan must contain exactly one job per registered client"
            )
        if any(e.local_round != 0 for e in events):
            raise ValueError("OPCM one-shot Plan must have only local_round=0")
        for event in events:
            if (event.task, event.dataset) != self.clients[event.client_id][:2]:
                raise ValueError("OPCM one-shot Plan task/dataset differs from client metadata")

    def prepare_download(self, global_state: LoRAState, context: ClientContext) -> LoRAState:
        if self.origin is None:
            raise RuntimeError("Configure OPCM one-shot server before downloading")
        if self.finished or context.local_round != 0 or context.base_version != 0:
            raise ValueError("OPCM one-shot only permits common-origin version0/round0 downloads")
        if context.client_id not in self.clients or context.client_id in self.experts:
            raise ValueError("OPCM one-shot client is unknown or already uploaded")
        if (context.task, context.dataset, context.num_samples) != self.clients[context.client_id]:
            raise ValueError("OPCM one-shot download metadata differs from registered client")
        return clone_state(self.origin)

    def on_arrival(self, update: Update, context: ServerContext) -> list[ServerMutation]:
        import torch

        if self.origin is None:
            raise RuntimeError("Configure OPCM one-shot server before receiving updates")
        if self.finished:
            raise ValueError("OPCM one-shot merge is complete; additional uploads are forbidden")
        if context.version != 0 or update.base_version != 0 or update.local_round != 0:
            raise ValueError("OPCM one-shot uploads must all start from server version0/round0")
        if context.expected_clients != len(self.clients):
            raise ValueError("OPCM one-shot expected client population changed")
        if update.client_id not in self.clients or update.client_id in self.experts:
            raise ValueError("OPCM one-shot rejects unknown or duplicate client uploads")
        if any(r["update_id"] == update.update_id for r in self.upload_records.values()):
            raise ValueError("OPCM one-shot duplicate update_id")
        if (update.task, update.dataset, update.num_samples) != self.clients[update.client_id]:
            raise ValueError("OPCM one-shot upload task/dataset/sample count differs from metadata")
        if update.optimizer_steps <= 0 or not math.isfinite(update.arrival_time):
            raise ValueError(
                "OPCM one-shot upload must contain trained steps and finite arrival time"
            )
        if any(
            set(state) != set(self.origin)
            for state in (
                update.base_state,
                update.local_state,
                context.global_state,
            )
        ):
            raise ValueError("OPCM one-shot upload does not match common-origin LoRA keys")
        checks = []
        device = next(iter(self.origin.values())).device
        for name, initial in self.origin.items():
            local, base, live = (
                update.local_state[name],
                update.base_state[name],
                context.global_state[name],
            )
            if any(not isinstance(value, torch.Tensor) for value in (local, base, live)):
                raise ValueError("OPCM one-shot requires tensor LoRA factors")
            if any(value.shape != initial.shape for value in (local, base, live)):
                raise ValueError("OPCM one-shot upload LoRA shapes differ from common origin")
            checks.extend(
                (
                    (base.to(device) == initial).all(),
                    torch.isfinite(local).all().to(device),
                    (live.to(device) == initial).all(),
                )
            )
        if not bool(torch.stack(checks).all()):
            raise ValueError("OPCM one-shot upload is non-finite or trained from another origin")
        # Store only local factors, not duplicate download anchors and deltas.
        self.experts[update.client_id] = clone_state(state_to_device(update.local_state, device))
        self.upload_records[update.client_id] = {
            "update_id": update.update_id,
            "client_id": update.client_id,
            "task": update.task,
            "dataset": update.dataset,
            "num_samples": update.num_samples,
            "arrival_time": update.arrival_time,
            "seed": update.seed,
            "optimizer_steps": update.optimizer_steps,
            "sample_ids_hash": update.sample_ids_hash,
        }
        self.arrivals += 1
        self.occupancy_sum += len(self.experts)
        if len(self.experts) < len(self.clients):
            return []
        return [self._merge(context)]

    def _ordered_clients(self) -> list[str]:
        clients = sorted(self.experts)
        if self.merge_order == "seeded_random":
            random.Random(self.merge_seed).shuffle(clients)
        elif self.merge_order == "arrival":
            clients.sort(key=lambda cid: (self.upload_records[cid]["arrival_time"], cid))
        return clients

    def _merge(self, context: ServerContext) -> ServerMutation:
        if self.origin is None or len(self.experts) != len(self.clients):
            raise RuntimeError("OPCM one-shot cannot merge an incomplete expert collection")
        order = self._ordered_clients()
        current = clone_state(self.origin)
        # Work on an independent merger; offline steps are NOT client uploads or
        # public server-version increments. Keep original opcm_lora untouched.
        merger = OPCMLoRA(self.merger.params)
        trace = []
        for index, client_id in enumerate(order):
            row = self.upload_records[client_id]
            local = self.experts[client_id]
            expert = Update(
                update_id=row["update_id"],
                client_id=client_id,
                task=row["task"],
                dataset=row["dataset"],
                num_samples=row["num_samples"],
                local_round=0,
                base_version=0,
                arrival_time=row["arrival_time"],
                base_state=self.origin,
                local_state=local,
                delta=subtract(local, self.origin),
                seed=row["seed"],
                sample_ids_hash=row["sample_ids_hash"],
                losses=[],
                optimizer_steps=row["optimizer_steps"],
            )
            # All independent experts share W0; the OPCM functional increment is
            # therefore BA_local - BA_origin, not an arrival-time rebased expert.
            mutation = merger.on_arrival(expert, ServerContext(current, 0, len(order)))[0]
            current = mutation.new_state
            trace.append(
                {
                    "merge_index": index + 1,
                    "client_id": client_id,
                    "update_id": row["update_id"],
                    **mutation.metadata,
                }
            )
        self.merger = merger
        self.merge_trace = trace
        self.finished = True
        self.buffer_aggregations = 1
        self.experts.clear()
        last_arrival = max(row["arrival_time"] for row in self.upload_records.values())
        return ServerMutation(
            current,
            1.0,
            [self.upload_records[cid]["update_id"] for cid in order],
            metadata={
                "one_shot": True,
                "offline_opcm": True,
                "adaptation": "independent_common_origin_experts_fixed_rank_lora",
                "uploads_per_client": 1,
                "download_version": 0,
                "merge_order_policy": self.merge_order,
                "merge_seed": self.merge_seed,
                "merge_client_order": order,
                "merge_steps": trace,
                "expert_count": len(order),
                "buffer_occupancy": len(order),
                "buffer_aggregation": 1,
                "public_server_aggregations": 1,
                "merge_weight": self.merger.params["merge_weight"],
                "mean_buffer_waiting_time": sum(
                    last_arrival - row["arrival_time"] for row in self.upload_records.values()
                )
                / len(order),
                "expert_training": copy.deepcopy(self.upload_records),
            },
        )

    def on_finish(self, context: ServerContext) -> list[ServerMutation]:
        if self.finished:
            return []
        missing = sorted(set(self.clients) - set(self.experts))
        raise RuntimeError(f"OPCM one-shot requires ALL client uploads; missing: {missing}")

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **super().state_dict(),
                "origin": self.origin,
                "clients": self.clients,
                "experts": self.experts,
                "upload_records": self.upload_records,
                "finished": self.finished,
                "merge_trace": self.merge_trace,
                "merger": self.merger.state_dict(),
                "buffer_aggregations": self.buffer_aggregations,
                "occupancy_sum": self.occupancy_sum,
                "arrivals": self.arrivals,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["params"])
        for key in (
            "origin",
            "clients",
            "experts",
            "upload_records",
            "finished",
            "merge_trace",
            "buffer_aggregations",
            "occupancy_sum",
            "arrivals",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
        self.merger.load_state_dict(state["merger"])
