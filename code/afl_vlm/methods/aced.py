"""ACED-LoRA: Appendix D Algorithm a.1 of arXiv:2511.19066v1.

Mitigating Participation Imbalance Bias in Asynchronous Federated Learning.
Preserve initial all-client gradients and first server descent, immediate uniform
active-cache aggregation, post-update activity timestamp reset and rejoining.
The project retains multi-step local AdamW and caches actual trajectory-mean
gradients (not AdamW delta/lr); this is a disclosed extension of the paper's K=1.
Activity timestamps are distinct from true cached-gradient origin versions.
See docs/aced.md for this important source limitation and accounting semantics.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from dataclasses import replace
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
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState


@register_method("aced")
class ACED(Method):
    name = "aced"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        supports_client_ddp=False,
        requires_staleness=True,
        requires_mean_gradient=True,
        requires_initial_gradient=True,
    )
    allowed_params = {"server_lr", "lr_scale", "delay_threshold", "cache_bits", "cache_seed"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_lr = self.params.get("server_lr")
        if self.server_lr is not None:
            self.server_lr = float(self.server_lr)
            if not math.isfinite(self.server_lr) or self.server_lr <= 0:
                raise ValueError("ACED server_lr must be null or finite and positive")
        self.lr_scale = float(self.params.get("lr_scale", 0.2))
        if not math.isfinite(self.lr_scale) or self.lr_scale <= 0:
            raise ValueError("ACED lr_scale must be finite and positive")
        self.delay_threshold = self.params.get("delay_threshold", 10)
        if type(self.delay_threshold) is not int or self.delay_threshold < 1:
            raise ValueError("ACED delay_threshold must be a positive integer")
        self.cache_bits = self.params.get("cache_bits", 32)
        if type(self.cache_bits) is not int or self.cache_bits not in {32, 8}:
            raise ValueError("ACED cache_bits must be 32 or 8")
        self.cache_seed = self.params.get("cache_seed", 42)
        if type(self.cache_seed) is not int or self.cache_seed < 0:
            raise ValueError("ACED cache_seed must be a non-negative integer")
        self.layout: tuple[tuple[str, tuple[int, ...]], ...] = ()
        self.client_ids: tuple[str, ...] = ()
        self.initial_anchor: Any = None
        self.cache: dict[str, Any] = {}
        self.activity_versions: dict[str, int] = {}
        self.origin_versions: dict[str, int] = {}
        self.pending_ids: dict[str, str] = {}
        self.client_rounds: dict[str, int] = {}
        self.seen_update_ids: set[str] = set()
        self.initial_gradient_only = False
        self.initialized = False
        self.bootstrap_applied = False
        self.clock = 0
        self.planned_updates = 0
        self.arrivals = 0
        self.skipped_steps = 0

    def _vector(self, state: Mapping[str, Any], device: Any = None) -> Any:
        import torch

        if not state:
            raise ValueError("ACED requires non-empty federated trainable state")
        if device is None:
            device = next(
                (value.device for value in state.values() if isinstance(value, torch.Tensor)),
                torch.device("cpu"),
            )
        values = [
            torch.as_tensor(state[name], device=device).detach().float() for name in sorted(state)
        ]
        layout = tuple(
            (name, tuple(value.shape)) for name, value in zip(sorted(state), values, strict=True)
        )
        if self.layout and self.layout != layout:
            raise ValueError("ACED federated parameter names/shapes changed")
        result = torch.cat([value.reshape(-1) for value in values])
        if result.numel() == 0 or not bool(torch.isfinite(result).all()):
            raise ValueError("ACED requires finite, non-empty federated values")
        if not self.layout:
            self.layout = layout
        return result

    def _state(self, vector: Any, reference: LoRAState) -> LoRAState:
        import torch

        result, offset = {}, 0
        for name, shape in self.layout:
            size = math.prod(shape)
            value = vector[offset : offset + size].reshape(shape).clone()
            ref = reference[name]
            result[name] = value.to(ref) if isinstance(ref, torch.Tensor) else value
            offset += size
        return result

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        initial = self._vector(initial_state)
        ids = tuple(sorted(str(client.id) for client in clients))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("ACED requires distinct configured clients")
        if self.client_ids and ids != self.client_ids:
            raise ValueError("ACED configured population changed")
        self.client_ids = ids
        if self.initial_anchor is None and not self.bootstrap_applied:
            self.initial_anchor = initial.clone()

    def configure_schedule(self, events: list[ScheduledEvent]) -> None:
        if not events or any(row.client_id not in self.client_ids for row in events):
            raise ValueError("ACED Plan contains no jobs or unknown clients")
        if {row.client_id for row in events} != set(self.client_ids):
            raise ValueError("ACED Plan must contain every configured client")
        if self.planned_updates and self.planned_updates != len(events):
            raise ValueError("ACED Plan update budget changed")
        self.planned_updates = len(events)
        if self.params.get("server_lr") is None:
            # Algorithm a.1 counts the bootstrap as the first of T server steps.
            self.server_lr = self.lr_scale * math.sqrt(len(self.client_ids) / (len(events) + 1))

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        # Raw-gradient server movements can be below FP16 adapter resolution.
        # Method-owned models only: frozen backbone/other baseline runs unchanged.
        for _, parameter in model.named_federated_parameters():
            parameter.data = parameter.data.float()

    def configure_local_training(self, train_config: Any) -> Any:
        if self.initial_gradient_only:
            # One real reference-effective batch at the fixed common origin;
            # gradient-only mode guarantees NO optimizer.step or weight decay.
            return replace(
                train_config,
                max_local_steps=1,
                planned_optimizer_steps=1,
                collect_mean_gradient=True,
                gradient_estimation_only=True,
            )
        return train_config

    def load_client_runtime_state(self, state: Mapping[str, Any], context: ClientContext) -> None:
        if set(state) - {"initial_gradient_only"}:
            raise ValueError("ACED has no client-private cache/control state")
        self.initial_gradient_only = bool(state.get("initial_gradient_only", False))

    def _encode(self, vector: Any, key: str) -> Any:
        if self.cache_bits == 32:
            return vector.clone()
        import torch

        lower, upper = vector.min(), vector.max()
        step = (upper - lower) / 255
        if not bool(torch.isfinite(step)):
            raise ValueError("ACED quantization range overflowed")
        scaled = ((vector - lower) / step.clamp_min(torch.finfo(vector.dtype).tiny)).clamp(0, 255)
        floor = scaled.floor()
        seed = int.from_bytes(
            hashlib.sha256(f"{self.cache_seed}:{key}".encode()).digest()[:8], "big"
        ) % (2**63 - 1)
        generator = torch.Generator(device=vector.device).manual_seed(seed)
        rounded = floor + (
            torch.rand(vector.shape, device=vector.device, generator=generator) < scaled - floor
        )
        return {"codes": rounded.to(torch.uint8), "lower": lower.clone(), "step": step.clone()}

    def _decode(self, cached: Any, reference: Any) -> Any:
        if self.cache_bits == 32:
            return cached.to(reference)
        return cached["lower"].to(reference) + cached["step"].to(reference) * cached["codes"].to(
            reference
        )

    def _mean(self, ids: list[str], reference: Any) -> Any:
        result = reference.new_zeros(reference.shape)
        for client in ids:
            result.add_(self._decode(self.cache[client], reference), alpha=1 / len(ids))
        return result

    @staticmethod
    def _storage_bytes(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(ACED._storage_bytes(item) for item in value.values())
        return value.numel() * value.element_size() if hasattr(value, "numel") else 0

    def initialize_client_gradient(self, update: Update) -> None:
        import torch

        if self.initialized or update.client_id not in self.client_ids:
            raise ValueError("ACED initializer is closed or client is unknown")
        if update.client_id in self.cache:
            raise ValueError("Duplicate ACED initial client gradient")
        if (
            update.mean_gradient is None
            or update.optimizer_steps != 1
            or update.base_version != 0
            or update.local_round != -1
            or not update.metadata.get("gradient_estimation_only", False)
        ):
            raise ValueError(
                "ACED initialization requires one real fixed-origin gradient observation"
            )
        reference = self.initial_anchor
        base = self._vector(update.base_state, reference.device)
        local = self._vector(update.local_state, reference.device)
        if not torch.equal(base, reference) or not torch.equal(local, reference):
            raise ValueError("ACED initialization must not change or mix model origins")
        gradient = self._vector(update.mean_gradient, reference.device)
        self.cache[update.client_id] = self._encode(gradient, f"initial:{update.client_id}")
        self.origin_versions[update.client_id] = 0

    def finalize_initial_gradients(self) -> None:
        if self.initialized or set(self.cache) != set(self.client_ids):
            raise ValueError("ACED needs exactly one initial gradient from every client")
        self.initialized = True

    def initial_server_update(self, context: ServerContext) -> ServerMutation:
        import torch

        if not self.initialized or self.bootstrap_applied or context.version != 0:
            raise RuntimeError("ACED bootstrap requires completed initialization at version zero")
        if self.server_lr is None or context.expected_clients != len(self.client_ids):
            raise RuntimeError("ACED bootstrap requires configured Plan and client population")
        current = self._vector(context.global_state)
        if not torch.equal(current, self.initial_anchor.to(current)):
            raise ValueError("ACED server changed before fixed-origin initialization completed")
        direction = self._mean(list(self.client_ids), current)
        new = current - self.server_lr * direction
        if not bool(torch.isfinite(new).all()):
            raise ValueError("ACED bootstrap produced non-finite parameters")
        self.bootstrap_applied = True
        self.activity_versions = dict.fromkeys(self.client_ids, 1)
        self.clock = 1
        self.initial_anchor = None
        return ServerMutation(
            self._state(new, context.global_state),
            self.server_lr,
            [],
            metadata={
                "aced_initialization": True,
                "gradient_observations": len(self.client_ids),
                "extra_client_gradient_messages": len(self.client_ids),
                "server_lr": self.server_lr,
                "algorithm_clock_after": self.clock,
                "planned_arrivals": self.planned_updates,
                "scheduled_server_iterations": self.planned_updates + 1,
                "initial_gradient_norm": float(direction.norm().item()),
                "cache_bits": self.cache_bits,
                "counted_as_upload": False,
            },
        )

    def on_arrival(self, update: Update, context: ServerContext) -> list[ServerMutation]:
        import torch

        if not self.bootstrap_applied or self.server_lr is None:
            raise RuntimeError("ACED all-client initialization/bootstrap has not completed")
        if (
            context.expected_clients != len(self.client_ids)
            or update.client_id not in self.client_ids
        ):
            raise ValueError("ACED client population mismatch")
        if update.update_id in self.seen_update_ids:
            raise ValueError("Duplicate ACED update")
        if update.local_round < 0 or update.local_round <= self.client_rounds.get(
            update.client_id, -1
        ):
            raise ValueError("ACED client local_round must strictly increase")
        if not 0 <= update.base_version <= context.version:
            raise ValueError("ACED update has invalid model provenance")
        if update.mean_gradient is None or update.optimizer_steps <= 0:
            raise ValueError("ACED requires actual trajectory-mean gradients, not delta/lr")
        current = self._vector(context.global_state)
        gradient = self._vector(update.mean_gradient, current.device)
        encoded = self._encode(gradient, update.update_id)
        overwritten = self.pending_ids.get(update.client_id)
        self.cache[update.client_id] = encoded
        self.origin_versions[update.client_id] = update.base_version
        self.pending_ids[update.client_id] = update.update_id
        ages = {client: self.clock - self.activity_versions[client] for client in self.client_ids}
        active = [client for client in self.client_ids if ages[client] <= self.delay_threshold]
        contributors = [self.pending_ids[client] for client in active if client in self.pending_ids]
        direction = self._mean(active, current)
        new = current - self.server_lr * direction if active else current.clone()
        if not bool(torch.isfinite(new).all()):
            raise ValueError("ACED active-cache descent produced non-finite parameters")
        for client in active:
            self.pending_ids.pop(client, None)
        clock_before = self.clock
        # Appendix D lines 8--10: update/skip FIRST, reset arrival activity AFTER.
        # Do not overwrite origin_versions: no stale gradient becomes truly fresh.
        self.activity_versions[update.client_id] = self.clock + 1
        self.clock += 1
        self.arrivals += 1
        self.skipped_steps += not bool(active)
        self.client_rounds[update.client_id] = update.local_round
        self.seen_update_ids.add(update.update_id)
        return [
            ServerMutation(
                self._state(new, context.global_state),
                self.server_lr if active else 0.0,
                contributors,
                increment_version=bool(active),
                metadata={
                    "aggregation_rule": "aced_uniform_active_cached_mean_gradient",
                    "cache_payload": "actual_local_trajectory_mean_gradient",
                    "server_lr": self.server_lr,
                    "delay_threshold": self.delay_threshold,
                    "algorithm_clock_before": clock_before,
                    "algorithm_clock_after": self.clock,
                    "active_clients": active,
                    "active_count": len(active),
                    "excluded_clients": [c for c in self.client_ids if c not in active],
                    "activity_ages_before": ages,
                    "cached_gradient_base_versions": dict(self.origin_versions),
                    "cached_gradient_true_staleness": {
                        client: context.version - self.origin_versions[client]
                        for client in self.client_ids
                    },
                    "arriving_client_included": update.client_id in active,
                    "rejoin_after_this_iteration": update.client_id not in active,
                    "empty_active_set": not active,
                    "skipped_steps": self.skipped_steps,
                    "cache_bits": self.cache_bits,
                    "cache_storage_bytes": sum(self._storage_bytes(x) for x in self.cache.values()),
                    "received_gradient_norm": float(gradient.norm().item()),
                    "active_gradient_norm": float(direction.norm().item()),
                    "quantization_error_norm": float(
                        (self._decode(encoded, current) - gradient).norm().item()
                    ),
                    "newly_incorporated_update_ids": contributors,
                    "pending_cached_update_ids": dict(self.pending_ids),
                    "overwritten_unincorporated_update_id": overwritten,
                    "arrival_base_version": update.base_version,
                    "arrival_staleness": context.version - update.base_version,
                    "gradient_observations": update.optimizer_steps,
                    "logical_arrival": update.arrival_time,
                    "arrival_update_id": update.update_id,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"schema_version": 1, **self.__dict__})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported ACED state schema")
        restored = type(self)(state["params"])
        if set(state) - {"schema_version"} != set(restored.__dict__):
            raise ValueError("ACED method-state fields do not match schema")
        payload = copy.deepcopy({key: state[key] for key in restored.__dict__})
        for key in ("lr_scale", "delay_threshold", "cache_bits", "cache_seed"):
            if payload[key] != getattr(restored, key):
                raise ValueError(f"ACED state {key} differs from parameters")
        expected_lr = restored.server_lr
        if expected_lr is None and payload["planned_updates"]:
            expected_lr = restored.lr_scale * math.sqrt(
                len(payload["client_ids"]) / (payload["planned_updates"] + 1)
            )
        if payload["server_lr"] != expected_lr:
            raise ValueError("ACED state server_lr differs from configured Plan/parameters")
        if not set(payload["cache"]) <= set(payload["client_ids"]):
            raise ValueError("ACED cache contains unknown clients")
        if payload["bootstrap_applied"] and (
            not payload["initialized"]
            or set(payload["cache"]) != set(payload["client_ids"])
            or payload["clock"] < 1
        ):
            raise ValueError("ACED warm-start state is inconsistent")
        self.__dict__.update(payload)
