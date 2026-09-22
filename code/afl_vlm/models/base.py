"""Model adapter contract and backend-independent trainable-state algebra."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

ScalarOrTensor: TypeAlias = Any
LoRAState: TypeAlias = dict[str, ScalarOrTensor]
MetricDict: TypeAlias = dict[str, float]


@dataclass(slots=True)
class TrainConfig:
    local_epochs: int
    batch_size: int
    gradient_accumulation: int
    learning_rate: float
    max_text_length: int
    seed: int
    loss_hook: Any | None = None
    gradient_hook: Any | None = None
    step_hook: Any | None = None
    progress_hook: Any | None = None
    collect_mean_gradient: bool = False
    context: dict[str, Any] = field(default_factory=dict)
    planned_optimizer_steps: int = 1
    max_local_steps: int | None = None
    distributed_rank: int = 0
    distributed_world_size: int = 1


@dataclass(slots=True)
class TrainResult:
    delta: LoRAState
    losses: list[float]
    optimizer_steps: int
    mean_gradient: LoRAState | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def clone_state(state: Mapping[str, ScalarOrTensor]) -> LoRAState:
    result: LoRAState = {}
    for name, value in state.items():
        if hasattr(value, "detach") and hasattr(value, "clone"):
            result[name] = value.detach().clone()
        else:
            result[name] = copy.deepcopy(value)
    return result


def state_to_device(
    state: Mapping[str, ScalarOrTensor], device: Any, dtype: Any | None = None
) -> LoRAState:
    """Move a trainable state once at a runtime boundary."""
    result: LoRAState = {}
    for name, value in state.items():
        if hasattr(value, "to"):
            kwargs = {"device": device}
            if dtype is not None:
                kwargs["dtype"] = dtype
            result[name] = value.detach().to(**kwargs)
        else:
            result[name] = copy.deepcopy(value)
    return result


def nested_to_device(value: Any, device: Any) -> Any:
    """Move tensors inside method/checkpoint containers without changing structure."""
    if hasattr(value, "detach") and hasattr(value, "to"):
        return value.detach().to(device=device)
    if isinstance(value, Mapping):
        return {key: nested_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [nested_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(nested_to_device(item, device) for item in value)
    return copy.deepcopy(value)


def _binary_value(left: ScalarOrTensor, right: ScalarOrTensor, fn: Any) -> ScalarOrTensor:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("State vectors have different lengths")
        return [_binary_value(a, b, fn) for a, b in zip(left, right, strict=True)]
    return fn(left, right)


def add_scaled(
    state: Mapping[str, ScalarOrTensor], delta: Mapping[str, ScalarOrTensor], scale: float
) -> LoRAState:
    if set(state) != set(delta):
        raise ValueError("State and delta keys differ")
    return {
        key: _binary_value(state[key], delta[key], lambda a, b: a + scale * b)
        for key in sorted(state)
    }


def subtract(left: Mapping[str, ScalarOrTensor], right: Mapping[str, ScalarOrTensor]) -> LoRAState:
    if set(left) != set(right):
        raise ValueError("State keys differ")
    return {key: _binary_value(left[key], right[key], lambda a, b: a - b) for key in sorted(left)}


def scale_state(state: Mapping[str, ScalarOrTensor], scale: float) -> LoRAState:
    return {key: _map_value(value, lambda x: x * scale) for key, value in state.items()}


def mean_states(states: Iterable[Mapping[str, ScalarOrTensor]]) -> LoRAState:
    items = list(states)
    if not items:
        raise ValueError("Cannot average an empty state collection")
    result = scale_state(items[0], 1.0 / len(items))
    for item in items[1:]:
        result = add_scaled(result, item, 1.0 / len(items))
    return result


def weighted_mean_states(
    states: Iterable[Mapping[str, ScalarOrTensor]], weights: Iterable[float]
) -> LoRAState:
    """Average compatible states with normalized non-negative weights."""
    items = list(states)
    raw_weights = [float(weight) for weight in weights]
    if not items or len(items) != len(raw_weights):
        raise ValueError("States and weights must be non-empty and have equal length")
    if any(weight < 0 for weight in raw_weights) or sum(raw_weights) <= 0:
        raise ValueError("State weights must be non-negative with a positive sum")
    total = sum(raw_weights)
    result = scale_state(items[0], raw_weights[0] / total)
    for item, weight in zip(items[1:], raw_weights[1:], strict=True):
        result = add_scaled(result, item, weight / total)
    return result


def _map_value(value: ScalarOrTensor, fn: Any) -> ScalarOrTensor:
    if isinstance(value, list):
        return [_map_value(item, fn) for item in value]
    return fn(value)


def map_state(state: Mapping[str, ScalarOrTensor], fn: Any) -> LoRAState:
    return {key: _map_value(value, fn) for key, value in state.items()}


def zip_states(
    left: Mapping[str, ScalarOrTensor], right: Mapping[str, ScalarOrTensor], fn: Any
) -> LoRAState:
    if set(left) != set(right):
        raise ValueError("State keys differ")
    return {key: _binary_value(left[key], right[key], fn) for key in sorted(left)}


def zeros_like(state: Mapping[str, ScalarOrTensor]) -> LoRAState:
    return map_state(state, lambda value: value * 0)


def _flat_values(value: ScalarOrTensor) -> Iterable[float]:
    if isinstance(value, list):
        for item in value:
            yield from _flat_values(item)
    elif hasattr(value, "detach"):
        for item in value.detach().float().cpu().reshape(-1).tolist():
            yield float(item)
    else:
        yield float(value)


def _sum_squares(value: ScalarOrTensor, tensor_totals: dict[str, Any]) -> float:
    if isinstance(value, list):
        return sum(_sum_squares(item, tensor_totals) for item in value)
    if hasattr(value, "detach"):
        term = value.detach().float().square().sum()
        key = str(term.device)
        tensor_totals[key] = tensor_totals.get(key, 0.0) + term
        return 0.0
    scalar = float(value)
    return scalar * scalar


def _dot_value(
    left: ScalarOrTensor,
    right: ScalarOrTensor,
    tensor_totals: dict[str, Any],
) -> float:
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError("State vectors have different lengths")
        return sum(_dot_value(a, b, tensor_totals) for a, b in zip(left, right, strict=True))
    if hasattr(left, "detach") and hasattr(right, "detach"):
        if left.shape != right.shape:
            raise ValueError("State tensors have different shapes")
        term = (left.detach().float() * right.detach().float()).sum()
        key = str(term.device)
        tensor_totals[key] = tensor_totals.get(key, 0.0) + term
        return 0.0
    return float(left) * float(right)


def state_norm(state: Mapping[str, ScalarOrTensor]) -> float:
    tensor_totals: dict[str, Any] = {}
    scalar_total = sum(_sum_squares(state[key], tensor_totals) for key in sorted(state))
    total = scalar_total + sum(float(value.item()) for value in tensor_totals.values())
    return math.sqrt(total)


def state_dot(left: Mapping[str, ScalarOrTensor], right: Mapping[str, ScalarOrTensor]) -> float:
    """Return the Euclidean inner product of two compatible trainable states."""
    if set(left) != set(right):
        raise ValueError("State keys differ")
    tensor_totals: dict[str, Any] = {}
    scalar_total = sum(_dot_value(left[key], right[key], tensor_totals) for key in sorted(left))
    return scalar_total + sum(float(value.item()) for value in tensor_totals.values())


def state_cosine(left: Mapping[str, ScalarOrTensor], right: Mapping[str, ScalarOrTensor]) -> float:
    dot = state_dot(left, right)
    denominator = state_norm(left) * state_norm(right)
    return 0.0 if denominator == 0 else dot / denominator


def state_hash(state: Mapping[str, ScalarOrTensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        for value in _flat_values(state[key]):
            digest.update(struct.pack("!d", value))
    return digest.hexdigest()


class ModelAdapter(ABC):
    """Only trainable adapter state crosses the federation boundary."""

    @abstractmethod
    def load(self, config: Mapping[str, Any]) -> None:
        """Load the base model and initialize its trainable adapter."""

    @abstractmethod
    def snapshot_trainable(self) -> LoRAState:
        """Return a detached clone of trainable adapter parameters."""

    @abstractmethod
    def load_trainable(self, state: Mapping[str, ScalarOrTensor]) -> None:
        """Restore trainable adapter parameters."""

    @abstractmethod
    def local_train(
        self, task_batch_stream: Iterable[Any], train_config: TrainConfig, task_name: str
    ) -> TrainResult:
        """Train locally and return an update relative to the loaded state."""

    @abstractmethod
    def evaluate(
        self,
        task_adapter: Any,
        sample_ids: list[str],
        mode: str,
        progress_hook: Any | None = None,
    ) -> MetricDict:
        """Evaluate the current state on a task-specific adapter."""

    def set_evaluation_context(self, task: str, client_id: str | None = None) -> None:
        """Select optional task/client routing without changing model state."""
        return None

    def get_federated_state(self) -> LoRAState:
        return self.snapshot_trainable()

    def load_federated_state(self, state: Mapping[str, ScalarOrTensor]) -> None:
        self.load_trainable(state)

    @staticmethod
    def compute_federated_delta(
        local_state: Mapping[str, ScalarOrTensor],
        base_state: Mapping[str, ScalarOrTensor],
    ) -> LoRAState:
        return subtract(local_state, base_state)

    def apply_federated_state(self, state: Mapping[str, ScalarOrTensor]) -> None:
        self.load_trainable(state)
