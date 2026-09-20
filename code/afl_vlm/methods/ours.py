"""Sensitivity-aware asynchronous LoRA consolidation.

This is the proposed AFVLM-CM method described in the author's design note.
Only federated LoRA parameters participate. A client estimates non-negative
LoRA-group sensitivity during its existing backward passes; the server then:

1. measures functional rather than version-only staleness;
2. maintains task-balanced historical sensitivity memory; and
3. performs precision-weighted, group-wise online consolidation.

The default estimator is the scale-invariant module-gate statistic
``(<grad_B, B> + <grad_A, A>) / 2`` squared and tracked by EMA. Rank-gate and
grouped Adam-second-moment estimators are selectable ablations. No frozen
backbone parameter is inspected, uploaded, or aggregated.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from afl_vlm.federation.types import (
    MethodCapabilities,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, clone_state

_LORA_PARAMETER = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")
_ESTIMATORS = {"module_gate", "rank_gate", "adam_v"}
_QUALITY_MODES = {"constant", "num_samples", "sqrt_num_samples", "optimizer_steps"}


@dataclass(frozen=True, slots=True)
class _GroupMember:
    parameter: str
    axis: int | None = None
    index: int | None = None


def _parameter_identity(name: str) -> tuple[str, str]:
    match = _LORA_PARAMETER.match(name)
    if match is None:
        raise ValueError(
            "The proposed method supports LoRA-only federated state; "
            f"unsupported trainable parameter: {name}"
        )
    return match.group("module"), match.group("side")


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(item) for item in value.shape)
    if isinstance(value, list):
        if not value:
            return (0,)
        child = _shape(value[0])
        if any(_shape(item) != child for item in value):
            raise ValueError("Ragged LoRA state is not supported")
        return (len(value), *child)
    return ()


def _sum_squares_and_count(value: Any) -> tuple[Any, int]:
    if isinstance(value, list):
        total: Any = 0.0
        count = 0
        for item in value:
            term, size = _sum_squares_and_count(item)
            total = total + term
            count += size
        return total, count
    if hasattr(value, "detach"):
        return value.detach().float().square().sum(), int(value.numel())
    scalar = float(value)
    return scalar * scalar, 1


def _difference(left: Any, right: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("State vectors have different lengths")
        return [_difference(a, b) for a, b in zip(left, right, strict=True)]
    return left - right


def _blend_value(current: Any, local: Any, alpha: float) -> Any:
    if isinstance(current, list) and isinstance(local, list):
        if len(current) != len(local):
            raise ValueError("State vectors have different lengths")
        return [_blend_value(a, b, alpha) for a, b in zip(current, local, strict=True)]
    return current + alpha * (local - current)


@register_method("ours")
class Ours(Method):
    """Sensitivity-aware asynchronous LoRA consolidation (proposed method)."""

    name = "ours"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_staleness=True,
        requires_local_state=True,
        requires_group_sensitivity=True,
    )
    allowed_params = {
        "sensitivity_estimator",
        "sensitivity_beta",
        "adam_beta2",
        "sensitivity_floor",
        "sensitivity_ceiling",
        "gamma",
        "delta_max",
        "lambda_history",
        "memory_beta",
        "omega_prior",
        "alpha_max",
        "epsilon_update",
        "epsilon_precision",
        "min_update_energy",
        "min_optimizer_steps",
        "quality_weighting",
        "task_weights",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.task_memory: dict[str, dict[str, float]] = {}
        self.task_weights: dict[str, float] = {}
        self._groups: dict[str, tuple[_GroupMember, ...]] = {}
        self._local_sensitivity: dict[str, Any] = {}
        self._local_sensitivity_steps = 0
        self._expected_local_groups: set[str] = set()
        self._local_pairs: dict[str, dict[str, tuple[str, Any]]] = {}
        self._validate_parameters()

    @property
    def estimator(self) -> str:
        return str(self.params.get("sensitivity_estimator", "module_gate"))

    def _validate_parameters(self) -> None:
        if self.estimator not in _ESTIMATORS:
            raise ValueError(
                f"sensitivity_estimator must be one of {sorted(_ESTIMATORS)}, "
                f"got {self.estimator!r}"
            )
        for key, default in (
            ("sensitivity_beta", 0.95),
            ("adam_beta2", 0.999),
            ("memory_beta", 0.9),
        ):
            value = float(self.params.get(key, default))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{key} must be in [0, 1)")
        for key, default in (
            ("sensitivity_floor", 1e-6),
            ("sensitivity_ceiling", 1e3),
            ("gamma", 1.0),
            ("delta_max", 10.0),
            ("lambda_history", 1.0),
            ("omega_prior", 1.0),
            ("epsilon_update", 1e-12),
            ("epsilon_precision", 1e-12),
            ("min_update_energy", 1e-12),
        ):
            value = float(self.params.get(key, default))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{key} must be non-negative")
        floor = float(self.params.get("sensitivity_floor", 1e-6))
        ceiling = float(self.params.get("sensitivity_ceiling", 1e3))
        if ceiling < floor:
            raise ValueError("sensitivity_ceiling must be >= sensitivity_floor")
        alpha_max = float(self.params.get("alpha_max", 0.9))
        if not 0.0 < alpha_max < 1.0:
            raise ValueError("alpha_max must be in (0, 1)")
        if int(self.params.get("min_optimizer_steps", 1)) < 1:
            raise ValueError("min_optimizer_steps must be positive")
        quality = str(self.params.get("quality_weighting", "constant"))
        if quality not in _QUALITY_MODES:
            raise ValueError(f"quality_weighting must be one of {sorted(_QUALITY_MODES)}")
        configured_weights = self.params.get("task_weights")
        if configured_weights is not None and not isinstance(configured_weights, Mapping):
            raise ValueError("task_weights must be a mapping or null")

    def validate_runtime(self) -> None:
        self._validate_parameters()

    @staticmethod
    def _pairs(items: Mapping[str, Any]) -> dict[str, dict[str, tuple[str, Any]]]:
        pairs: dict[str, dict[str, tuple[str, Any]]] = {}
        for name, value in items.items():
            module, side = _parameter_identity(name)
            if side in pairs.setdefault(module, {}):
                raise ValueError(f"Duplicate LoRA {side} tensor for module {module}")
            pairs[module][side] = (name, value)
        incomplete = {module: sorted({"A", "B"} - set(pair)) for module, pair in pairs.items()}
        incomplete = {module: sides for module, sides in incomplete.items() if sides}
        if incomplete:
            raise ValueError(f"Incomplete LoRA A/B pairs: {incomplete}")
        if not pairs:
            raise ValueError("No federated LoRA A/B parameters were found")
        return pairs

    def _build_groups(self, state: Mapping[str, Any]) -> dict[str, tuple[_GroupMember, ...]]:
        pairs = self._pairs(state)
        groups: dict[str, tuple[_GroupMember, ...]] = {}
        for module, pair in sorted(pairs.items()):
            a_name, a_value = pair["A"]
            b_name, b_value = pair["B"]
            a_shape, b_shape = _shape(a_value), _shape(b_value)
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(f"Invalid LoRA shapes for {module}: A={a_shape}, B={b_shape}")
            if self.estimator == "rank_gate":
                for rank in range(a_shape[0]):
                    groups[f"{module}::rank_{rank}"] = (
                        _GroupMember(a_name, axis=0, index=rank),
                        _GroupMember(b_name, axis=1, index=rank),
                    )
            else:
                groups[module] = (_GroupMember(a_name), _GroupMember(b_name))
        return groups

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        del clients
        parameters = dict(model.named_federated_parameters())
        self._groups = self._build_groups(parameters)
        self._local_pairs = self._pairs(parameters)

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        self._groups = self._build_groups(initial_state)
        tasks = sorted({str(item.task) for item in clients})
        if not tasks:
            raise ValueError("The proposed method requires task identities")
        configured = self.params.get("task_weights")
        if configured is None:
            self.task_weights = {task: 1.0 / len(tasks) for task in tasks}
        else:
            raw = {str(key): float(value) for key, value in configured.items()}
            if set(raw) != set(tasks):
                raise ValueError(
                    f"task_weights must exactly match the tasks in the system profile: {tasks}"
                )
            if (
                any(not math.isfinite(value) or value < 0.0 for value in raw.values())
                or sum(raw.values()) <= 0.0
            ):
                raise ValueError("task_weights must be non-negative with a positive sum")
            total = sum(raw.values())
            self.task_weights = {task: raw[task] / total for task in tasks}
        for task in tasks:
            self.task_memory.setdefault(task, {})

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        return {
            "reset_sensitivity_for": context.client_id,
            "sensitivity_groups": sorted(self._groups),
        }

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        expected = state.get("reset_sensitivity_for")
        if expected is not None and expected != context.client_id:
            raise ValueError("Sensitivity runtime state belongs to a different client")
        self._local_sensitivity = {}
        self._local_sensitivity_steps = 0
        self._expected_local_groups = {str(group) for group in state.get("sensitivity_groups", ())}

    def _ema(self, key: str, value: Any, beta: float, *, zero_initialized: bool) -> None:
        previous = self._local_sensitivity.get(key)
        if previous is None:
            self._local_sensitivity[key] = (1.0 - beta) * value if zero_initialized else value
        else:
            self._local_sensitivity[key] = beta * previous + (1.0 - beta) * value

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        del context
        if not self._local_pairs:
            self._local_pairs = self._pairs(dict(model.named_federated_parameters()))
        pairs = self._local_pairs
        beta = float(self.params.get("sensitivity_beta", 0.95))
        adam_beta2 = float(self.params.get("adam_beta2", 0.999))
        observed = False
        for module, pair in sorted(pairs.items()):
            _, a = pair["A"]
            _, b = pair["B"]
            if a.grad is None or b.grad is None:
                continue
            observed = True
            a_value, b_value = a.detach().float(), b.detach().float()
            a_grad, b_grad = a.grad.detach().float(), b.grad.detach().float()
            if self.estimator == "module_gate":
                # Both expressions estimate dL/dz for a virtual gate z*B*A.
                # Their mean is used only for finite-precision symmetry.
                h = ((a_grad * a_value).sum() + (b_grad * b_value).sum()) * 0.5
                self._ema(module, h.square(), beta, zero_initialized=False)
            elif self.estimator == "rank_gate":
                if a_value.ndim != 2 or b_value.ndim != 2:
                    raise ValueError(f"Rank-gate requires matrix LoRA tensors for {module}")
                h_a = (a_grad * a_value).sum(dim=1)
                h_b = (b_grad * b_value).sum(dim=0)
                if h_a.shape != h_b.shape:
                    raise ValueError(f"LoRA rank mismatch for {module}")
                scores = ((h_a + h_b) * 0.5).square()
                for rank in range(int(scores.numel())):
                    self._ema(
                        f"{module}::rank_{rank}",
                        scores[rank],
                        beta,
                        zero_initialized=False,
                    )
            else:
                squared_sum = a_grad.square().sum() + b_grad.square().sum()
                count = a_grad.numel() + b_grad.numel()
                self._ema(
                    module,
                    squared_sum / count,
                    adam_beta2,
                    zero_initialized=True,
                )
        if not observed:
            raise RuntimeError("No LoRA gradients were available for sensitivity estimation")
        self._local_sensitivity_steps += 1

    def _normalized_sensitivity(self) -> tuple[dict[str, float], dict[str, float]]:
        if not self._local_sensitivity or self._local_sensitivity_steps <= 0:
            raise RuntimeError("Local training produced no LoRA sensitivity statistics")
        correction = 1.0
        if self.estimator == "adam_v":
            beta2 = float(self.params.get("adam_beta2", 0.999))
            correction = max(1.0 - beta2**self._local_sensitivity_steps, 1e-30)
        keys = sorted(self._local_sensitivity)
        values = [self._local_sensitivity[key] for key in keys]
        if values and all(hasattr(value, "detach") for value in values):
            # One device synchronization for all groups, rather than one .item()
            # per module/rank at the end of every local job.
            import torch

            packed = torch.stack([value.detach().float() for value in values])
            scalars = (packed / correction).clamp_min(0.0).cpu().tolist()
            raw = {key: float(value) for key, value in zip(keys, scalars, strict=True)}
        else:
            raw = {
                key: max(0.0, float(value) / correction)
                for key, value in zip(keys, values, strict=True)
            }
        for group in self._expected_local_groups:
            raw.setdefault(group, 0.0)
        if any(not math.isfinite(value) for value in raw.values()):
            raise FloatingPointError("Non-finite LoRA sensitivity was produced")
        mean_value = sum(raw.values()) / max(1, len(raw))
        if mean_value <= 0.0:
            return ({key: 0.0 for key in raw}, raw)
        floor = float(self.params.get("sensitivity_floor", 1e-6))
        ceiling = float(self.params.get("sensitivity_ceiling", 1e3))
        normalized = {
            key: min(ceiling, max(floor, value / mean_value)) if value > 0.0 else 0.0
            for key, value in raw.items()
        }
        return normalized, raw

    def local_step(
        self, model: Any, step: int, total_steps: int, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        del model, context
        if step != total_steps:
            return {}
        sensitivity, raw = self._normalized_sensitivity()
        positive = [value for value in raw.values() if value > 0.0]
        return {
            "group_sensitivity": sensitivity,
            "sensitivity_estimator": self.estimator,
            "sensitivity_steps": self._local_sensitivity_steps,
            "sensitivity_raw_mean": sum(raw.values()) / len(raw),
            "sensitivity_raw_max": max(raw.values()),
            "sensitivity_positive_groups": len(positive),
        }

    def prepare_upload(self, update: Update, context: Any) -> Update:
        del context
        sensitivity = update.metadata.get("group_sensitivity")
        if not isinstance(sensitivity, Mapping) or not sensitivity:
            raise ValueError("The proposed method requires non-empty group_sensitivity metadata")
        parsed = {str(key): float(value) for key, value in sensitivity.items()}
        if any(not math.isfinite(value) or value < 0.0 for value in parsed.values()):
            raise ValueError("group_sensitivity values must be finite and non-negative")
        if update.metadata.get("sensitivity_estimator") != self.estimator:
            raise ValueError("Client/server sensitivity estimator mismatch")
        if int(update.metadata.get("sensitivity_steps", -1)) != update.optimizer_steps:
            raise ValueError("Sensitivity statistics do not cover every local optimizer step")
        update.metadata["group_sensitivity"] = parsed
        return update

    @staticmethod
    def _selected(value: Any, member: _GroupMember) -> Any:
        if member.axis is None:
            return value
        if not hasattr(value, "select"):
            raise TypeError("Rank-gate server fusion requires tensor-valued LoRA state")
        return value.select(member.axis, int(member.index))

    def _group_mean_square(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        members: tuple[_GroupMember, ...],
    ) -> Any:
        total: Any = 0.0
        count = 0
        for member in members:
            difference = _difference(left[member.parameter], right[member.parameter])
            selected = self._selected(difference, member)
            term, size = _sum_squares_and_count(selected)
            total = total + term
            count += size
        if count <= 0:
            raise ValueError("Sensitivity group is empty")
        return total / count

    @staticmethod
    def _as_float(value: Any) -> float:
        return float(value.detach().float().item()) if hasattr(value, "detach") else float(value)

    def _weighted_distances(
        self, update: Update, current: Mapping[str, Any], sensitivity: Mapping[str, float]
    ) -> tuple[float, float]:
        drift_total: Any = 0.0
        update_total: Any = 0.0
        for group, members in self._groups.items():
            importance = sensitivity[group]
            drift_total = drift_total + importance * self._group_mean_square(
                current, update.base_state, members
            )
            update_total = update_total + importance * self._group_mean_square(
                update.local_state, update.base_state, members
            )
        return self._as_float(drift_total), self._as_float(update_total)

    def _quality(self, update: Update) -> float:
        mode = str(self.params.get("quality_weighting", "constant"))
        if mode == "constant":
            return 1.0
        if mode == "num_samples":
            return float(update.num_samples)
        if mode == "sqrt_num_samples":
            return math.sqrt(update.num_samples)
        return float(update.optimizer_steps)

    def _historical_precision(self, group: str) -> float:
        prior = float(self.params.get("omega_prior", 1.0))
        return prior + sum(
            weight * self.task_memory.get(task, {}).get(group, 0.0)
            for task, weight in self.task_weights.items()
        )

    def _fuse(
        self,
        current: Mapping[str, Any],
        local: Mapping[str, Any],
        alphas: Mapping[str, float],
    ) -> LoRAState:
        if set(current) != set(local):
            raise ValueError("Current and client LoRA state keys differ")
        memberships: dict[str, list[tuple[_GroupMember, str]]] = {}
        for group, members in self._groups.items():
            for member in members:
                memberships.setdefault(member.parameter, []).append((member, group))
        result: LoRAState = {}
        for name in sorted(current):
            entries = memberships.get(name)
            if not entries:
                raise ValueError(f"Federated parameter is not assigned to a LoRA group: {name}")
            if entries[0][0].axis is None:
                result[name] = _blend_value(current[name], local[name], alphas[entries[0][1]])
                continue
            value = current[name]
            if not hasattr(value, "new_tensor"):
                raise TypeError("Rank-gate server fusion requires tensor-valued LoRA state")
            axis = int(entries[0][0].axis)
            entries = sorted(entries, key=lambda item: int(item[0].index))
            weights = value.new_tensor([alphas[group] for _, group in entries])
            broadcast_shape = [1] * value.ndim
            broadcast_shape[axis] = len(entries)
            weights = weights.view(broadcast_shape)
            result[name] = value + weights * (local[name] - value)
        return result

    def _rejection(
        self,
        update: Update,
        server_context: ServerContext,
        reason: str,
        metadata: Mapping[str, Any],
    ) -> list[ServerMutation]:
        return [
            ServerMutation(
                clone_state(server_context.global_state),
                0.0,
                [update.update_id],
                increment_version=False,
                metadata={"rejected": reason, **dict(metadata)},
            )
        ]

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        sensitivity = {
            str(key): float(value)
            for key, value in update.metadata.get("group_sensitivity", {}).items()
        }
        if set(sensitivity) != set(self._groups):
            missing = sorted(set(self._groups) - set(sensitivity))
            extra = sorted(set(sensitivity) - set(self._groups))
            raise ValueError(
                f"Sensitivity groups do not match LoRA state; missing={missing}, extra={extra}"
            )
        version_staleness = server_context.version - update.base_version
        if version_staleness < 0:
            raise ValueError("Update base_version cannot be newer than the server")
        if update.task not in self.task_weights:
            raise ValueError(f"Unknown task identity in update: {update.task}")
        common = {
            "version_staleness": version_staleness,
            "sensitivity_estimator": self.estimator,
            "sensitivity_groups": len(sensitivity),
        }
        minimum_steps = int(self.params.get("min_optimizer_steps", 1))
        if update.optimizer_steps < minimum_steps:
            return self._rejection(update, server_context, "insufficient_optimizer_steps", common)
        drift_energy, update_energy = self._weighted_distances(
            update, server_context.global_state, sensitivity
        )
        common.update({"drift_energy": drift_energy, "update_energy": update_energy})
        if not math.isfinite(drift_energy) or not math.isfinite(update_energy):
            return self._rejection(update, server_context, "non_finite_energy", common)
        if update_energy <= float(self.params.get("min_update_energy", 1e-12)):
            return self._rejection(update, server_context, "negligible_update_energy", common)

        epsilon_update = float(self.params.get("epsilon_update", 1e-12))
        functional_staleness = math.sqrt(drift_energy / (update_energy + epsilon_update))
        capped = min(functional_staleness, float(self.params.get("delta_max", 10.0)))
        reliability = math.exp(-float(self.params.get("gamma", 1.0)) * capped)
        quality = self._quality(update)
        if not math.isfinite(quality) or quality <= 0.0:
            return self._rejection(update, server_context, "invalid_quality_weight", common)
        if reliability <= 0.0:
            return self._rejection(update, server_context, "zero_reliability", common)
        lambda_history = float(self.params.get("lambda_history", 1.0))
        epsilon_precision = float(self.params.get("epsilon_precision", 1e-12))
        alpha_max = float(self.params.get("alpha_max", 0.9))
        alphas: dict[str, float] = {}
        for group, importance in sensitivity.items():
            local_precision = quality * reliability * importance
            historical_precision = self._historical_precision(group)
            denominator = local_precision + lambda_history * historical_precision
            alpha = local_precision / (denominator + epsilon_precision)
            alphas[group] = min(alpha_max, max(0.0, alpha))

        new_state = self._fuse(server_context.global_state, update.local_state, alphas)
        memory_beta = float(self.params.get("memory_beta", 0.9))
        task_memory = self.task_memory.setdefault(update.task, {})
        for group, importance in sensitivity.items():
            evidence = quality * reliability * importance
            task_memory[group] = (
                memory_beta * task_memory.get(group, 0.0) + (1.0 - memory_beta) * evidence
            )

        alpha_values = list(alphas.values())
        metadata = {
            **common,
            "functional_staleness": functional_staleness,
            "reliability": reliability,
            "quality_weight": quality,
            "mean_group_alpha": sum(alpha_values) / len(alpha_values),
            "min_group_alpha": min(alpha_values),
            "max_group_alpha": max(alpha_values),
            "task_memory": update.task,
        }
        return [
            ServerMutation(
                new_state,
                metadata["mean_group_alpha"],
                [update.update_id],
                metadata=metadata,
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": copy.deepcopy(self.params),
            "task_memory": copy.deepcopy(self.task_memory),
            "task_weights": copy.deepcopy(self.task_weights),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self._validate_parameters()
        self.task_memory = {
            str(task): {str(group): float(value) for group, value in memory.items()}
            for task, memory in state.get("task_memory", {}).items()
        }
        self.task_weights = {
            str(task): float(value) for task, value in state.get("task_weights", {}).items()
        }
