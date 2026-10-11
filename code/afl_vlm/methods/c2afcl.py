"""C2-AFCL, ICML 2026: fixed-task federated-trainable-state specialization.

Cross-task Calibration for Asynchronous Federated Continual Learning,
https://proceedings.mlr.press/v306/li26gx.html, Sections 4.1/4.2, Eqs. 4--6.
Retain cached RAW update differences and joint-coordinate SVD projection.
Replace sequential previous tasks by other fixed tasks' incorporated history.
Default same-task buffer size one gives per-arrival Eq. 6 (K=1), not Eq. 2
raw-model interpolation. No changes to local loss, optimizer, Plan or model.
See docs/c2afcl.md for protocol ambiguity and deliberately disclosed adaptations.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState


@register_method("c2afcl")
class C2AFCL(Method):
    name = "c2afcl"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_local_state=True,
        requires_staleness=True,
        requires_buffer=True,
    )
    allowed_params = {"server_lr", "subspace_rank", "buffer_size", "svd_rtol", "eps"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.5))
        self.svd_rtol = float(self.params.get("svd_rtol", 1e-6))
        self.eps = float(self.params.get("eps", 1e-12))
        if (
            not all(math.isfinite(x) for x in (self.server_lr, self.svd_rtol, self.eps))
            or not 0 < self.server_lr <= 1
            or not 0 < self.svd_rtol < 1
            or self.eps <= 0
        ):
            raise ValueError("C2-AFCL requires 0 < server_lr <= 1, 0 < svd_rtol < 1, eps > 0")
        for key, default in (("subspace_rank", 5), ("buffer_size", 1)):
            value = self.params.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"C2-AFCL {key} must be a positive integer")
            setattr(self, key, value)
        self.layout: tuple[tuple[str, tuple[int, ...]], ...] = ()
        self.client_tasks: dict[str, str] = {}
        self.client_cache: dict[tuple[str, str], Any] = {}
        self.client_rounds: dict[tuple[str, str], int] = {}
        self.task_means: dict[str, Any] = {}
        self.task_counts: dict[str, int] = {}
        self.buffers: dict[str, list[dict[str, Any]]] = {}
        self.seen_update_ids: set[str] = set()
        self.buffer_aggregations = 0
        self.occupancy_sum = 0
        self.arrivals = 0

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        self._vector(initial_state)
        for client in clients:
            # ClientSpec supplies id; no index-derived task descriptors.
            client_id = str(client.id)
            task = str(client.task)
            if client_id in self.client_tasks and self.client_tasks[client_id] != task:
                raise ValueError(f"C2-AFCL client {client_id} changed fixed task")
            self.client_tasks[client_id] = task

    def _vector(self, state: LoRAState, device: Any | None = None) -> Any:
        """Detached FP32 joint coordinates, retaining the aggregation device."""
        import torch

        if not state:
            raise ValueError("C2-AFCL requires non-empty federated trainable state")
        if device is None:
            device = next(
                (value.device for value in state.values() if isinstance(value, torch.Tensor)),
                torch.device("cpu"),
            )
        tensors = [
            torch.as_tensor(state[name], device=device).detach().to(dtype=torch.float32)
            for name in sorted(state)
        ]
        layout = tuple(
            (name, tuple(t.shape)) for name, t in zip(sorted(state), tensors, strict=True)
        )
        if self.layout and layout != self.layout:
            raise ValueError("C2-AFCL federated parameter names/shapes changed")
        vector = torch.cat([t.reshape(-1) for t in tensors])
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("C2-AFCL received non-finite federated state")
        if not self.layout:
            self.layout = layout
        return vector

    def _state(self, vector: Any, reference: LoRAState) -> LoRAState:
        import torch

        result: LoRAState = {}
        finite_checks = []
        offset = 0
        for name, shape in self.layout:
            count = math.prod(shape)
            value = vector[offset : offset + count].reshape(shape)
            original = reference[name]
            if isinstance(original, torch.Tensor):
                result[name] = value.to(device=original.device, dtype=original.dtype).clone()
                finite_checks.append(torch.isfinite(result[name]).all().to(vector.device))
            else:
                result[name] = value.tolist() if isinstance(original, list) else float(value.item())
            offset += count
        if finite_checks and not bool(torch.stack(finite_checks).all()):
            raise ValueError("C2-AFCL aggregation overflowed destination parameter dtype")
        return result

    def _project(self, delta: Any, task: str) -> tuple[Any, dict[str, Any]]:
        """Eqs. 4--5: thin SVD, delta - U_d(U_d^T delta); never build D x D P."""
        import torch

        tasks = sorted(q for q in self.task_means if q != task)
        info: dict[str, Any] = {"protected_tasks": tasks, "subspace_rank": 0}
        if not tasks:
            return delta.clone(), info
        history = torch.stack([self.task_means[q].to(delta.device) for q in tasks], dim=1)
        basis, singular, _ = torch.linalg.svd(history, full_matrices=False)
        cutoff = torch.maximum(singular[0] * self.svd_rtol, singular.new_tensor(self.eps))
        rank = min(self.subspace_rank, int((singular > cutoff).sum().item()))
        basis = basis[:, :rank]
        safe = delta - basis @ (basis.T @ delta)
        info.update(
            subspace_rank=rank,
            singular_values=singular.detach().cpu().tolist(),
            projection_residual_norm=float(torch.linalg.vector_norm(basis.T @ safe).item()),
        )
        return safe, info

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        import torch

        if update.base_version < 0 or update.base_version > server_context.version:
            raise ValueError("C2-AFCL received an invalid/future base version")
        if update.num_samples <= 0 or not math.isfinite(update.arrival_time):
            raise ValueError("C2-AFCL requires samples and a finite arrival time")
        if not update.task or not update.client_id:
            raise ValueError("C2-AFCL requires client_id and task metadata")
        if update.update_id in self.seen_update_ids:
            raise ValueError(f"C2-AFCL duplicate update: {update.update_id}")
        if self.client_tasks.get(update.client_id, update.task) != update.task:
            raise ValueError(f"C2-AFCL client {update.client_id} changed fixed task")
        key = (update.client_id, update.task)
        if update.local_round <= self.client_rounds.get(key, -1):
            raise ValueError("C2-AFCL requires causal client local-round order")
        with torch.no_grad():
            current = self._vector(server_context.global_state)
            # Algorithm 1: u_k = theta_local - theta_download, NOT theta_local - theta_now.
            raw = self._vector(update.local_state, current.device) - self._vector(
                update.base_state, current.device
            )
            previous = self.client_cache.get(key)
            calibrated = raw if previous is None else raw - previous.to(raw.device)
            safe, projection = self._project(calibrated, update.task)
            if not bool(torch.isfinite(safe).all()):
                raise ValueError("C2-AFCL calibration produced non-finite values")
            diagnostic = {
                "client_id": update.client_id,
                "task": update.task,
                "cache_hit": previous is not None,
                "base_version": update.base_version,
                "receive_version": server_context.version,
                "staleness": server_context.version - update.base_version,
                "raw_delta_norm": float(torch.linalg.vector_norm(raw).item()),
                "calibrated_delta_norm": float(torch.linalg.vector_norm(calibrated).item()),
                "safe_delta_norm": float(torch.linalg.vector_norm(safe).item()),
                "removed_delta_norm": float(torch.linalg.vector_norm(calibrated - safe).item()),
                **projection,
            }
            pending = self.buffers.setdefault(update.task, [])
            pending.append(
                {
                    "update_id": update.update_id,
                    "raw": raw,
                    "safe": safe,
                    "arrival_time": update.arrival_time,
                    "diagnostic": diagnostic,
                }
            )
            # Section 4.1 caches the RAW update on reception, never the projected one.
            self.client_cache[key] = raw.clone()
            self.client_rounds[key] = update.local_round
            self.client_tasks[update.client_id] = update.task
            self.seen_update_ids.add(update.update_id)
            self.arrivals += 1
            self.occupancy_sum += len(pending)
            if len(pending) < self.buffer_size:
                return []
            return [self._flush(update.task, server_context, update.arrival_time)]

    def _flush(self, task: str, context: ServerContext, flush_time: float) -> ServerMutation:
        import torch

        with torch.no_grad():
            rows = self.buffers[task]
            current = self._vector(context.global_state)
            safe_mean = torch.stack([row["safe"].to(current.device) for row in rows]).mean(dim=0)
            # Eq. 6: equal update mean, added to CURRENT server (no stale-model pullback).
            new_state = self._state(current + self.server_lr * safe_mean, context.global_state)
            count = self.task_counts.get(task, 0)
            mean = self.task_means.get(task)
            if mean is not None:
                mean = mean.to(current.device)
            for row in rows:
                raw = row["raw"].to(current.device)
                mean = raw.clone() if mean is None else mean + (raw - mean) / (count + 1)
                count += 1
            self.task_means[task] = mean
            self.task_counts[task] = count
            self.buffers.pop(task)
            self.buffer_aggregations += 1
            return ServerMutation(
                new_state,
                self.server_lr,
                [row["update_id"] for row in rows],
                metadata={
                    "c2afcl": True,
                    "aggregation_rule": "equal_mean_calibrated_projected_delta",
                    "projection_space": "joint_federated_trainable_coordinates",
                    "history_scope": "incorporated_other_fixed_tasks",
                    "buffer_occupancy": len(rows),
                    "buffer_aggregation": self.buffer_aggregations,
                    "buffer_task": task,
                    "mean_buffer_waiting_time": sum(flush_time - r["arrival_time"] for r in rows)
                    / len(rows),
                    "task_history_counts": dict(self.task_counts),
                    "c2afcl_diagnostics": {r["update_id"]: r["diagnostic"] for r in rows},
                },
            )

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        """Flush every residual task buffer; server calls repeatedly with fresh state."""
        if not self.buffers:
            return []
        task = min(self.buffers, key=lambda q: (self.buffers[q][0]["arrival_time"], q))
        last_arrival = max(row["arrival_time"] for rows in self.buffers.values() for row in rows)
        return [self._flush(task, server_context, last_arrival)]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **super().state_dict(),
                "layout": self.layout,
                "client_tasks": self.client_tasks,
                "client_cache": self.client_cache,
                "client_rounds": self.client_rounds,
                "task_means": self.task_means,
                "task_counts": self.task_counts,
                "buffers": self.buffers,
                "seen_update_ids": self.seen_update_ids,
                "buffer_aggregations": self.buffer_aggregations,
                "occupancy_sum": self.occupancy_sum,
                "arrivals": self.arrivals,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Revalidate/restore parameter-derived values, not just self.params.
        restored = type(self)(state["params"])
        self.__dict__.update(restored.__dict__)
        for key in (
            "layout",
            "client_tasks",
            "client_cache",
            "client_rounds",
            "task_means",
            "task_counts",
            "buffers",
            "seen_update_ids",
            "buffer_aggregations",
            "occupancy_sum",
            "arrivals",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
