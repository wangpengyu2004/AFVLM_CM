"""Rank-Gate sensitivity-aware asynchronous LoRA aggregation.

The proposed method operates only on PEFT LoRA A/B parameters. It introduces
no trainable parameter, never materializes a dense LoRA update, and performs no
SVD or rank alignment. Rank sensitivity is collected from the existing local
backward pass; the server uses it for functional staleness and computes one
shared aggregation coefficient for both A and B of each LoRA module.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, clone_state

_LORA_PARAMETER = re.compile(r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$")


@dataclass(frozen=True, slots=True)
class LoRAModuleMetadata:
    module_name: str
    a_name: str
    b_name: str
    rank: int
    in_features: int
    out_features: int
    scaling: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "a_name": self.a_name,
            "b_name": self.b_name,
            "rank": self.rank,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "scaling": self.scaling,
        }


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "shape"):
        return tuple(int(item) for item in value.shape)
    if isinstance(value, list):
        if not value:
            return (0,)
        child = _shape(value[0])
        if any(_shape(item) != child for item in value):
            raise ValueError("Ragged LoRA tensors are not supported")
        return (len(value), *child)
    return ()


def _parameter_identity(name: str) -> tuple[str, str]:
    match = _LORA_PARAMETER.match(name)
    if match is None:
        raise ValueError(
            f"Rank-Gate supports LoRA-only federated state; unsupported trainable parameter: {name}"
        )
    return match.group("module"), match.group("side")


def _state_pairs(state: Mapping[str, Any]) -> dict[str, dict[str, tuple[str, Any]]]:
    pairs: dict[str, dict[str, tuple[str, Any]]] = {}
    for name, value in state.items():
        module, side = _parameter_identity(name)
        if side in pairs.setdefault(module, {}):
            raise ValueError(f"Duplicate LoRA {side} tensor for module {module}")
        pairs[module][side] = (name, value)
    missing = {
        module: sorted({"A", "B"} - set(pair))
        for module, pair in pairs.items()
        if set(pair) != {"A", "B"}
    }
    if missing:
        raise ValueError(f"Incomplete LoRA A/B pairs: {missing}")
    if not pairs:
        raise ValueError("No federated LoRA A/B parameters were found")
    return pairs


def _active_adapter(module: Any, adapters: set[str]) -> str:
    configured: list[str] = []
    active = getattr(module, "active_adapter", None)
    if isinstance(active, str):
        configured.append(active)
    elif isinstance(active, Sequence):
        configured.extend(str(item) for item in active)
    active_many = getattr(module, "active_adapters", None)
    if isinstance(active_many, str):
        configured.append(active_many)
    elif isinstance(active_many, Sequence):
        configured.extend(str(item) for item in active_many)
    configured.append("default")
    for candidate in configured:
        if candidate in adapters:
            return candidate
    if len(adapters) == 1:
        return next(iter(adapters))
    raise ValueError(f"Cannot select one active LoRA adapter from {sorted(adapters)}")


def iter_lora_modules(model: Any) -> Iterator[tuple[str, Any, Any, float]]:
    """Yield PEFT LoRA module name, A, B, and scaling for the active adapter."""

    root = getattr(model, "model", model)
    if not hasattr(root, "named_modules"):
        raise TypeError("iter_lora_modules requires a PEFT model or model adapter")
    found = 0
    for module_name, module in root.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        scaling = getattr(module, "scaling", None)
        if lora_a is None or lora_b is None or scaling is None:
            continue
        adapters = set(lora_a.keys()) & set(lora_b.keys()) & set(scaling.keys())
        if not adapters:
            continue
        adapter = _active_adapter(module, adapters)
        a = lora_a[adapter].weight
        b = lora_b[adapter].weight
        a_shape, b_shape = _shape(a), _shape(b)
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
            raise ValueError(f"LoRA rank mismatch for {module_name}: A={a_shape}, B={b_shape}")
        found += 1
        yield module_name, a, b, float(scaling[adapter])
    if found == 0:
        raise ValueError("No active PEFT LoRA modules were found")


def _discover_model_modules(model: Any) -> dict[str, tuple[Any, Any, LoRAModuleMetadata]]:
    trainable = dict(model.named_federated_parameters())
    pairs = _state_pairs(trainable)
    by_parameter_ids = {
        (id(pair["A"][1]), id(pair["B"][1])): module for module, pair in pairs.items()
    }
    discovered: dict[str, tuple[Any, Any, LoRAModuleMetadata]] = {}
    for _, a, b, scaling in iter_lora_modules(model):
        module = by_parameter_ids.get((id(a), id(b)))
        if module is None:
            continue
        a_name, a_value = pairs[module]["A"]
        b_name, b_value = pairs[module]["B"]
        a_shape, b_shape = _shape(a_value), _shape(b_value)
        metadata = LoRAModuleMetadata(
            module_name=module,
            a_name=a_name,
            b_name=b_name,
            rank=a_shape[0],
            in_features=a_shape[1],
            out_features=b_shape[0],
            scaling=scaling,
        )
        discovered[module] = (a, b, metadata)
    if set(discovered) != set(pairs):
        missing = sorted(set(pairs) - set(discovered))
        raise ValueError(f"Federated LoRA modules were not found in the PEFT model: {missing}")
    return discovered


class RankSensitivityTracker:
    """Track bias-corrected rank-gate sensitivity without joining the graph."""

    def __init__(
        self,
        model: Any,
        beta: float = 0.95,
        warmup_steps: int = 1,
        clip_max: float = 10.0,
        eps: float = 1e-12,
    ) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("Rank sensitivity beta must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("Rank sensitivity warmup_steps must be non-negative")
        if clip_max <= 0.0 or eps <= 0.0:
            raise ValueError("Rank sensitivity clip_max and eps must be positive")
        self.beta = float(beta)
        self.warmup_steps = int(warmup_steps)
        self.clip_max = float(clip_max)
        self.eps = float(eps)
        self.modules = _discover_model_modules(model)
        self.ema = {
            name: a.detach().float().new_zeros(metadata.rank)
            for name, (a, _, metadata) in self.modules.items()
        }
        self.count = {name: 0 for name in self.modules}

    def update(self, optimizer_step: int) -> None:
        """Collect once after accumulated backward and before optimizer.step."""

        if optimizer_step < self.warmup_steps:
            return
        torch = __import__("torch")
        with torch.no_grad():
            for name, (a, b, _) in self.modules.items():
                if a.grad is None or b.grad is None:
                    continue
                h_a = (a.grad.detach().float() * a.detach().float()).sum(dim=1)
                h_b = (b.grad.detach().float() * b.detach().float()).sum(dim=0)
                if h_a.shape != h_b.shape:
                    raise ValueError(f"LoRA rank mismatch while tracking {name}")
                current = (0.5 * (h_a + h_b)).square()
                self.ema[name].mul_(self.beta)
                self.ema[name].add_(current, alpha=1.0 - self.beta)
                self.count[name] += 1

    def finalize(self) -> tuple[dict[str, Any], dict[str, float], bool]:
        """Return globally normalized rank and module sensitivity."""

        corrected = {
            name: (
                value / (1.0 - self.beta ** self.count[name])
                if self.count[name] > 0
                else value.clone()
            )
            for name, value in self.ema.items()
        }
        all_values = [corrected[name].reshape(-1) for name in sorted(corrected)]
        if not all_values:
            raise RuntimeError("Rank sensitivity tracker has no LoRA modules")
        torch = __import__("torch")
        packed = torch.cat(all_values)
        mean_value = float(packed.mean().detach().cpu())
        valid = math.isfinite(mean_value) and mean_value > self.eps
        if valid:
            rank_sensitivity = {
                name: (value / (mean_value + self.eps))
                .clamp_(0.0, self.clip_max)
                .detach()
                .float()
                .cpu()
                for name, value in corrected.items()
            }
        else:
            rank_sensitivity = {
                name: torch.ones_like(value, dtype=torch.float32, device="cpu")
                for name, value in corrected.items()
            }
        module_sensitivity = {
            name: float(value.mean().item()) for name, value in rank_sensitivity.items()
        }
        return rank_sensitivity, module_sensitivity, valid

    def metadata(self) -> dict[str, dict[str, Any]]:
        return {name: metadata.as_dict() for name, (_, _, metadata) in sorted(self.modules.items())}


def _vector_dot(left: Any, right: Any) -> Any:
    if hasattr(left, "detach") and hasattr(right, "detach"):
        return (left.detach().double().reshape(-1) * right.detach().double().reshape(-1)).sum()
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("Vector lengths differ")
        return sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    raise TypeError("Rank functional distance requires tensor or list vectors")


def rank_functional_distance_sq(
    b1: Any,
    a1: Any,
    b0: Any,
    a0: Any,
    scaling: float,
) -> Any:
    """Squared distance between two scaled rank-1 LoRA functions."""

    value = float(scaling) ** 2 * (
        _vector_dot(b1, b1) * _vector_dot(a1, a1)
        + _vector_dot(b0, b0) * _vector_dot(a0, a0)
        - 2.0 * _vector_dot(b1, b0) * _vector_dot(a1, a0)
    )
    return value.clamp_min(0.0) if hasattr(value, "clamp_min") else max(0.0, float(value))


def _row(matrix: Any, rank: int) -> Any:
    return matrix[rank]


def _column(matrix: Any, rank: int) -> Any:
    if hasattr(matrix, "select"):
        return matrix.select(1, rank)
    return [row[rank] for row in matrix]


def _sensitivity_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        return [float(item) for item in value.detach().float().cpu().reshape(-1).tolist()]
    return [float(item) for item in value]


def _as_float(value: Any) -> float:
    return float(value.detach().double().cpu().item()) if hasattr(value, "detach") else float(value)


def _blend_value(current: Any, client: Any, alpha: float) -> Any:
    if isinstance(current, list) and isinstance(client, list):
        if len(current) != len(client):
            raise ValueError("LoRA tensor shapes differ")
        return [
            _blend_value(left, right, alpha) for left, right in zip(current, client, strict=True)
        ]
    return current + alpha * (client - current)


def compute_functional_staleness(
    base_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    client_state: Mapping[str, Any],
    rank_sensitivity: Mapping[str, Any],
    lora_metadata: Mapping[str, Mapping[str, Any]],
    eps: float = 1e-12,
) -> tuple[float, float, float]:
    """Compute sensitivity-weighted rank-functional server drift and local update."""

    server_drift: Any = 0.0
    local_update: Any = 0.0
    if set(rank_sensitivity) != set(lora_metadata):
        raise ValueError("Rank sensitivity and LoRA metadata modules differ")
    for module_name in sorted(lora_metadata):
        metadata = lora_metadata[module_name]
        a_name, b_name = str(metadata["a_name"]), str(metadata["b_name"])
        a_s, b_s = base_state[a_name], base_state[b_name]
        a_t, b_t = current_state[a_name], current_state[b_name]
        a_k, b_k = client_state[a_name], client_state[b_name]
        rank = int(metadata["rank"])
        values = _sensitivity_values(rank_sensitivity[module_name])
        if len(values) != rank:
            raise ValueError(
                f"Rank sensitivity size mismatch for {module_name}: {len(values)} != {rank}"
            )
        normalizer = int(metadata["out_features"]) * int(metadata["in_features"])
        if normalizer <= 0:
            raise ValueError(f"Invalid LoRA module size for {module_name}")
        scaling = float(metadata["scaling"])
        for index, importance in enumerate(values):
            d_server = rank_functional_distance_sq(
                _column(b_t, index),
                _row(a_t, index),
                _column(b_s, index),
                _row(a_s, index),
                scaling,
            )
            d_local = rank_functional_distance_sq(
                _column(b_k, index),
                _row(a_k, index),
                _column(b_s, index),
                _row(a_s, index),
                scaling,
            )
            server_drift = server_drift + importance * d_server / normalizer
            local_update = local_update + importance * d_local / normalizer
    drift = _as_float(server_drift)
    update = _as_float(local_update)
    relative = math.sqrt(drift / (update + eps))
    return drift, update, relative


def staleness_reliability(
    relative_staleness: float,
    gamma: float = 1.0,
    delta_max: float = 10.0,
) -> float:
    return math.exp(-float(gamma) * min(float(relative_staleness), float(delta_max)))


def compute_module_alpha(
    module_sensitivity: float,
    reliability: float,
    history_precision: float,
    client_weight: float = 1.0,
    history_lambda: float = 1.0,
    alpha_max: float = 0.5,
    eps: float = 1e-12,
) -> float:
    client_precision = client_weight * reliability * module_sensitivity
    server_precision = history_lambda * history_precision
    alpha = client_precision / (client_precision + server_precision + eps)
    return min(max(alpha, 0.0), alpha_max)


def _parameter_delta_norm(state: Mapping[str, Any]) -> float:
    total: Any = 0.0
    for value in state.values():
        if hasattr(value, "detach"):
            total = total + value.detach().double().square().sum()
        else:
            stack = [value]
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                else:
                    total = total + float(item) ** 2
    return math.sqrt(max(_as_float(total), 0.0))


def _numeric_summary(values: Sequence[float], clip_max: float | None = None) -> dict[str, Any]:
    """Build JSON-safe diagnostics without adding a numerical dependency."""

    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "zero_fraction": None,
            "clipped_fraction": None,
        }
    mean = math.fsum(numbers) / len(numbers)
    variance = math.fsum((value - mean) ** 2 for value in numbers) / len(numbers)
    return {
        "count": len(numbers),
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(numbers),
        "max": max(numbers),
        "zero_fraction": sum(value == 0.0 for value in numbers) / len(numbers),
        "clipped_fraction": (
            sum(value >= clip_max for value in numbers) / len(numbers)
            if clip_max is not None
            else None
        ),
    }


def _sensitivity_diagnostics(
    rank_sensitivity: Mapping[str, Sequence[float]],
    module_sensitivity: Mapping[str, float],
    observations: Mapping[str, int],
    clip_max: float,
) -> dict[str, Any]:
    rank_rows = [
        {"module": module, "rank": index, "sensitivity": float(value)}
        for module, values in sorted(rank_sensitivity.items())
        for index, value in enumerate(values)
    ]
    top_ranks = sorted(
        rank_rows,
        key=lambda item: (-item["sensitivity"], item["module"], item["rank"]),
    )[:10]
    top_modules = [
        {"module": module, "sensitivity": float(value)}
        for module, value in sorted(
            module_sensitivity.items(), key=lambda item: (-item[1], item[0])
        )[:10]
    ]
    return {
        "rank_by_module": copy.deepcopy(dict(rank_sensitivity)),
        "module_by_module": copy.deepcopy(dict(module_sensitivity)),
        "observations_by_module": dict(observations),
        "rank_summary": _numeric_summary(
            [item["sensitivity"] for item in rank_rows], clip_max=clip_max
        ),
        "module_summary": _numeric_summary(list(module_sensitivity.values())),
        "observation_summary": _numeric_summary(list(observations.values())),
        "top_ranks": top_ranks,
        "top_modules": top_modules,
    }


@register_method("ours")
class Ours(Method):
    """Rank-Gate sensitivity-aware asynchronous LoRA aggregation."""

    name = "ours"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_staleness=True,
        requires_local_state=True,
        requires_group_sensitivity=True,
    )
    allowed_params = {
        "sensitivity_beta",
        "sensitivity_warmup_steps",
        "sensitivity_clip_max",
        "gamma",
        "delta_max",
        "history_beta",
        "history_lambda",
        "base_precision",
        "alpha_max",
        "eps",
        "min_update_energy",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.task_memory: dict[str, dict[str, float]] = {}
        self.task_weights: dict[str, float] = {}
        self._server_metadata: dict[str, dict[str, Any]] = {}
        self._model_metadata: dict[str, dict[str, Any]] = {}
        self._tracker: RankSensitivityTracker | None = None
        self._validate_parameters()

    def _value(self, key: str, default: float | int) -> float:
        return float(self.params.get(key, default))

    def _validate_parameters(self) -> None:
        beta = self._value("sensitivity_beta", 0.95)
        history_beta = self._value("history_beta", 0.95)
        if not 0.0 <= beta < 1.0 or not 0.0 <= history_beta < 1.0:
            raise ValueError("sensitivity_beta and history_beta must be in [0, 1)")
        if int(self.params.get("sensitivity_warmup_steps", 1)) < 0:
            raise ValueError("sensitivity_warmup_steps must be non-negative")
        positive = {
            "sensitivity_clip_max": 10.0,
            "gamma": 1.0,
            "delta_max": 10.0,
            "history_lambda": 1.0,
            "base_precision": 1.0,
            "alpha_max": 0.5,
            "eps": 1e-12,
            "min_update_energy": 1e-16,
        }
        for key, default in positive.items():
            value = self._value(key, default)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} must be positive")
        if self._value("alpha_max", 0.5) > 1.0:
            raise ValueError("alpha_max must be <= 1")

    def validate_runtime(self) -> None:
        self._validate_parameters()

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        del clients
        discovered = _discover_model_modules(model)
        self._model_metadata = {
            name: metadata.as_dict() for name, (_, _, metadata) in sorted(discovered.items())
        }

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        pairs = _state_pairs(initial_state)
        for module, pair in pairs.items():
            a_name, a = pair["A"]
            b_name, b = pair["B"]
            a_shape, b_shape = _shape(a), _shape(b)
            if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
                raise ValueError(f"Invalid LoRA shapes for {module}: A={a_shape}, B={b_shape}")
            self._server_metadata[module] = {
                "module_name": module,
                "a_name": a_name,
                "b_name": b_name,
                "rank": a_shape[0],
                "in_features": a_shape[1],
                "out_features": b_shape[0],
                "scaling": None,
            }
        if self._model_metadata:
            if set(self._model_metadata) != set(self._server_metadata):
                raise ValueError("Model/server LoRA module names differ")
            for module, server_item in self._server_metadata.items():
                model_item = self._model_metadata[module]
                for key in ("a_name", "b_name", "rank", "in_features", "out_features"):
                    if model_item[key] != server_item[key]:
                        raise ValueError(f"Model/server LoRA metadata mismatch: {module}.{key}")
                server_item["scaling"] = float(model_item["scaling"])
        tasks = sorted({str(client.task) for client in clients})
        if not tasks:
            raise ValueError("Rank-Gate requires client task identities")
        self.task_weights = {task: 1.0 / len(tasks) for task in tasks}
        self.task_memory = {
            task: {
                module: float(self.task_memory.get(task, {}).get(module, 0.0))
                for module in self._server_metadata
            }
            for task in tasks
        }

    def client_runtime_state(self, context: Any) -> dict[str, Any]:
        return {"reset_rank_sensitivity_for": context.client_id}

    def load_client_runtime_state(self, state: Mapping[str, Any], context: Any) -> None:
        expected = state.get("reset_rank_sensitivity_for")
        if expected is not None and expected != context.client_id:
            raise ValueError("Rank sensitivity runtime state belongs to another client")
        self._tracker = None

    def transform_gradients(self, model: Any, context: Mapping[str, Any]) -> None:
        if self._tracker is None:
            self._tracker = RankSensitivityTracker(
                model,
                beta=self._value("sensitivity_beta", 0.95),
                warmup_steps=int(self.params.get("sensitivity_warmup_steps", 1)),
                clip_max=self._value("sensitivity_clip_max", 10.0),
                eps=self._value("eps", 1e-12),
            )
        self._tracker.update(int(context.get("optimizer_step", 0)))

    def local_step(
        self, model: Any, step: int, total_steps: int, context: Mapping[str, Any]
    ) -> dict[str, Any]:
        del model, context
        if step != total_steps:
            return {}
        if self._tracker is None:
            raise RuntimeError("Local training produced no Rank-Gate tracker")
        rank_sensitivity, module_sensitivity, valid = self._tracker.finalize()
        return {
            "rank_sensitivity": rank_sensitivity,
            "module_sensitivity": module_sensitivity,
            "sensitivity_valid": valid,
            "sensitivity_observations": dict(self._tracker.count),
            "lora_metadata": self._tracker.metadata(),
        }

    @staticmethod
    def _metadata_signature(metadata: Mapping[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        return tuple(
            (
                name,
                str(item["a_name"]),
                str(item["b_name"]),
                int(item["rank"]),
                int(item["in_features"]),
                int(item["out_features"]),
                float(item["scaling"]),
            )
            for name, item in sorted(metadata.items())
        )

    def _validate_client_update(self, update: Update) -> None:
        rank_raw = update.metadata.get("rank_sensitivity")
        module_raw = update.metadata.get("module_sensitivity")
        observation_raw = update.metadata.get("sensitivity_observations")
        metadata_raw = update.metadata.get("lora_metadata")
        if not isinstance(rank_raw, Mapping) or not isinstance(module_raw, Mapping):
            raise ValueError("Rank-Gate update requires rank and module sensitivity")
        if not isinstance(observation_raw, Mapping):
            raise ValueError("Rank-Gate update requires sensitivity observation counts")
        if not isinstance(metadata_raw, Mapping):
            raise ValueError("Rank-Gate update requires LoRA metadata")
        metadata = {
            str(name): {str(key): value for key, value in item.items()}
            for name, item in metadata_raw.items()
        }
        if set(metadata) != set(self._server_metadata):
            raise ValueError("Client/server LoRA module names differ")
        for name, expected in self._server_metadata.items():
            actual = metadata[name]
            for key in ("a_name", "b_name", "rank", "in_features", "out_features"):
                if actual[key] != expected[key]:
                    raise ValueError(f"Client/server LoRA metadata mismatch: {name}.{key}")
            if not math.isfinite(float(actual["scaling"])):
                raise ValueError(f"Non-finite LoRA scaling for {name}")
        if any(item["scaling"] is not None for item in self._server_metadata.values()):
            if self._metadata_signature(metadata) != self._metadata_signature(
                self._server_metadata
            ):
                raise ValueError("LoRA scaling or shape differs across clients")
        else:
            self._server_metadata = copy.deepcopy(metadata)

        rank_sensitivity: dict[str, list[float]] = {}
        module_sensitivity: dict[str, float] = {}
        observations: dict[str, int] = {}
        if (
            set(rank_raw) != set(metadata)
            or set(module_raw) != set(metadata)
            or set(observation_raw) != set(metadata)
        ):
            raise ValueError("Sensitivity modules do not match LoRA modules")
        for name, item in metadata.items():
            values = _sensitivity_values(rank_raw[name])
            if len(values) != int(item["rank"]):
                raise ValueError(f"Rank sensitivity size mismatch for {name}")
            if any(not math.isfinite(value) or value < 0.0 for value in values):
                raise ValueError(f"Invalid rank sensitivity for {name}")
            module_value = float(module_raw[name])
            expected_mean = sum(values) / len(values)
            if not math.isfinite(module_value) or module_value < 0.0:
                raise ValueError(f"Invalid module sensitivity for {name}")
            if not math.isclose(module_value, expected_mean, rel_tol=1e-5, abs_tol=1e-7):
                raise ValueError(f"Module sensitivity is not the rank mean for {name}")
            observation_value = observation_raw[name]
            if isinstance(observation_value, bool):
                raise ValueError(f"Invalid sensitivity observation count for {name}")
            observation_count = int(observation_value)
            if observation_count < 0 or observation_count != float(observation_value):
                raise ValueError(f"Invalid sensitivity observation count for {name}")
            rank_sensitivity[name] = values
            module_sensitivity[name] = module_value
            observations[name] = observation_count
        update.metadata["rank_sensitivity"] = rank_sensitivity
        update.metadata["module_sensitivity"] = module_sensitivity
        update.metadata["sensitivity_observations"] = observations
        update.metadata["lora_metadata"] = metadata
        update.metadata["sensitivity_valid"] = bool(update.metadata.get("sensitivity_valid", False))
        update.metadata["_rank_gate_validated"] = True

    def prepare_upload(self, update: Update, context: Any) -> Update:
        del context
        self._validate_client_update(update)
        return update

    def _historical_precision(self, module: str) -> float:
        return self._value("base_precision", 1.0) + sum(
            weight * self.task_memory.get(task, {}).get(module, 0.0)
            for task, weight in self.task_weights.items()
        )

    def _fuse_modules(
        self,
        current: Mapping[str, Any],
        client: Mapping[str, Any],
        alphas: Mapping[str, float],
    ) -> LoRAState:
        if set(current) != set(client):
            raise ValueError("Current and client LoRA state keys differ")
        result = clone_state(current)
        covered: set[str] = set()
        for module, metadata in self._server_metadata.items():
            alpha = float(alphas[module])
            for key in ("a_name", "b_name"):
                name = str(metadata[key])
                result[name] = _blend_value(current[name], client[name], alpha)
                covered.add(name)
        if covered != set(current):
            raise ValueError("Some federated parameters are outside Rank-Gate LoRA modules")
        return result

    def _reject(
        self,
        update: Update,
        context: ServerContext,
        reason: str,
        metadata: Mapping[str, Any],
    ) -> list[ServerMutation]:
        return [
            ServerMutation(
                clone_state(context.global_state),
                0.0,
                [update.update_id],
                increment_version=False,
                metadata={"rejected": reason, **dict(metadata)},
            )
        ]

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        if not update.metadata.get("_rank_gate_validated"):
            self._validate_client_update(update)
        if update.task not in self.task_weights:
            raise ValueError(f"Unknown task identity: {update.task}")
        version_staleness = server_context.version - update.base_version
        if version_staleness < 0:
            raise ValueError("Update base_version cannot exceed server version")
        rank_sensitivity = update.metadata["rank_sensitivity"]
        module_sensitivity = update.metadata["module_sensitivity"]
        observations = update.metadata["sensitivity_observations"]
        drift, local, relative = compute_functional_staleness(
            update.base_state,
            server_context.global_state,
            update.local_state,
            rank_sensitivity,
            self._server_metadata,
            eps=self._value("eps", 1e-12),
        )
        parameter_norm = _parameter_delta_norm(update.delta)
        task_memory_before = {
            module: float(self.task_memory[update.task][module])
            for module in sorted(self._server_metadata)
        }
        historical_precision_before = {
            module: self._historical_precision(module)
            for module in sorted(self._server_metadata)
        }
        diagnostics = {
            "schema_version": 1,
            "client": {
                "update_id": update.update_id,
                "client_id": update.client_id,
                "task": update.task,
                "local_round": update.local_round,
                "base_version": update.base_version,
                "receive_version": server_context.version,
                "arrival_time": update.arrival_time,
                "num_samples": update.num_samples,
                "optimizer_steps": update.optimizer_steps,
            },
            "sensitivity": {
                "valid": bool(update.metadata["sensitivity_valid"]),
                **_sensitivity_diagnostics(
                    rank_sensitivity,
                    module_sensitivity,
                    observations,
                    clip_max=self._value("sensitivity_clip_max", 10.0),
                ),
            },
            "functional_staleness": {
                "version_staleness": version_staleness,
                "server_drift": drift,
                "local_update": local,
                "parameter_delta_norm": parameter_norm,
                "relative_staleness": relative,
                "reliability": None,
            },
            "aggregation": {
                "accepted": False,
                "rejection_reason": None,
                "module_alpha_by_module": {},
                "alpha_summary": _numeric_summary([]),
            },
            "memory": {
                "task": update.task,
                "task_memory_before": task_memory_before,
                "task_memory_after": dict(task_memory_before),
                "historical_precision_before": historical_precision_before,
                "historical_precision_after": dict(historical_precision_before),
                "task_memory_before_summary": _numeric_summary(
                    list(task_memory_before.values())
                ),
                "task_memory_after_summary": _numeric_summary(
                    list(task_memory_before.values())
                ),
            },
        }
        common = {
            "version_staleness": version_staleness,
            "server_drift": drift,
            "local_update": local,
            "parameter_delta_norm": parameter_norm,
            "relative_staleness": relative,
            "sensitivity_valid": update.metadata["sensitivity_valid"],
            "ours_diagnostics": diagnostics,
        }
        if any(not math.isfinite(value) for value in (drift, local, relative, parameter_norm)):
            diagnostics["aggregation"]["rejection_reason"] = "non_finite_update"
            return self._reject(update, server_context, "non_finite_update", common)
        reliability = staleness_reliability(
            relative,
            gamma=self._value("gamma", 1.0),
            delta_max=self._value("delta_max", 10.0),
        )
        diagnostics["functional_staleness"]["reliability"] = reliability
        minimum = self._value("min_update_energy", 1e-16)
        if local < minimum and parameter_norm < minimum:
            diagnostics["aggregation"]["rejection_reason"] = "no_effective_update"
            return self._reject(update, server_context, "no_effective_update", common)

        alphas = {
            module: compute_module_alpha(
                module_sensitivity=module_sensitivity[module],
                reliability=reliability,
                history_precision=self._historical_precision(module),
                client_weight=1.0,
                history_lambda=self._value("history_lambda", 1.0),
                alpha_max=self._value("alpha_max", 0.5),
                eps=self._value("eps", 1e-12),
            )
            for module in sorted(self._server_metadata)
        }
        new_state = self._fuse_modules(server_context.global_state, update.local_state, alphas)

        history_beta = self._value("history_beta", 0.95)
        task_memory = self.task_memory.setdefault(update.task, {})
        for module, sensitivity in module_sensitivity.items():
            task_memory[module] = (
                history_beta * task_memory.get(module, 0.0)
                + (1.0 - history_beta) * reliability * sensitivity
            )

        task_memory_after = {
            module: float(self.task_memory[update.task][module])
            for module in sorted(self._server_metadata)
        }
        historical_precision_after = {
            module: self._historical_precision(module)
            for module in sorted(self._server_metadata)
        }
        diagnostics["aggregation"] = {
            "accepted": True,
            "rejection_reason": None,
            "module_alpha_by_module": dict(alphas),
            "alpha_summary": _numeric_summary(list(alphas.values())),
        }
        diagnostics["memory"].update(
            {
                "task_memory_after": task_memory_after,
                "historical_precision_after": historical_precision_after,
                "task_memory_after_summary": _numeric_summary(
                    list(task_memory_after.values())
                ),
            }
        )

        alpha_values = list(alphas.values())
        metadata = {
            **common,
            "reliability": reliability,
            "module_alphas": alphas,
            "mean_module_alpha": sum(alpha_values) / len(alpha_values),
            "min_module_alpha": min(alpha_values),
            "max_module_alpha": max(alpha_values),
            "task_memory": update.task,
        }
        return [
            ServerMutation(
                new_state,
                metadata["mean_module_alpha"],
                [update.update_id],
                metadata=metadata,
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": copy.deepcopy(self.params),
            "task_memory": copy.deepcopy(self.task_memory),
            "task_weights": copy.deepcopy(self.task_weights),
            "lora_metadata": copy.deepcopy(self._server_metadata),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self._validate_parameters()
        self.task_memory = {
            str(task): {str(module): float(value) for module, value in memory.items()}
            for task, memory in state.get("task_memory", {}).items()
        }
        self.task_weights = {
            str(task): float(value) for task, value in state.get("task_weights", {}).items()
        }
        self._server_metadata = {
            str(module): {str(key): value for key, value in metadata.items()}
            for module, metadata in state.get("lora_metadata", {}).items()
        }
