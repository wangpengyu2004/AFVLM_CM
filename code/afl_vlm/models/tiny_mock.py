"""Deterministic CPU-only model used to verify mechanisms and accounting."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from typing import Any

from afl_vlm.models.base import (
    LoRAState,
    ModelAdapter,
    TrainConfig,
    TrainResult,
    clone_state,
    subtract,
)
from afl_vlm.models.registry import register_model


def _stable_unit(text: str) -> float:
    raw = hashlib.sha256(text.encode("utf-8")).digest()[:8]
    return int.from_bytes(raw, "big") / (2**64 - 1)


@register_model("tiny_mock")
class TinyMockAdapter(ModelAdapter):
    """A two-dimensional non-linear task model with controlled cross-task effects."""

    def __init__(self) -> None:
        self._state: LoRAState = {"adapter": [0.0, 0.0, 0.0]}
        self._config: dict[str, Any] = {}

    def load(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)
        dimension = int(config.get("mock_dimension", 3))
        self._state = {"adapter": [0.0] * dimension}

    def snapshot_trainable(self) -> LoRAState:
        return clone_state(self._state)

    def load_trainable(self, state: Mapping[str, Any]) -> None:
        self._state = clone_state(state)

    def _target(self, task_name: str, sample_ids: list[str]) -> list[float]:
        dimension = len(self._state["adapter"])
        sign = 1.0 if task_name in {"fast", "hateful_memes"} else -1.0
        jitter = sum(_stable_unit(sample_id) - 0.5 for sample_id in sample_ids) / max(
            len(sample_ids), 1
        )
        base = [sign * (1.15 - 0.2 * index) for index in range(dimension)]
        # The shared second axis makes ordinary transfer possible while opposing
        # first/third axes make cross-task drift observable.
        if dimension > 1:
            base[1] = 0.45
        return [value + 0.08 * jitter for value in base]

    def local_train(
        self, task_batch_stream: Iterable[Any], train_config: TrainConfig, task_name: str
    ) -> TrainResult:
        samples = list(task_batch_stream)
        sample_ids = [str(getattr(sample, "id", sample)) for sample in samples]
        start = self.snapshot_trainable()
        vector = list(self._state["adapter"])
        target = self._target(task_name, sample_ids)
        rng = random.Random(train_config.seed)
        losses: list[float] = []
        # Mock learning rate is deliberately rescaled so a five-step CPU run has
        # visible effects while retaining the configured real-model learning rate.
        step_size = float(self._config.get("mock_step_size", 0.18))
        for _step in range(train_config.local_steps):
            noise = [(rng.random() - 0.5) * 0.006 for _ in vector]
            gradient = [
                value - wanted + noise_i
                for value, wanted, noise_i in zip(vector, target, noise, strict=True)
            ]
            vector = [
                value - step_size * grad for value, grad in zip(vector, gradient, strict=True)
            ]
            losses.append(
                sum((value - wanted) ** 2 for value, wanted in zip(vector, target, strict=True))
                / len(vector)
            )
        self._state = {"adapter": vector}
        return TrainResult(
            delta=subtract(self._state, start),
            losses=losses,
            optimizer_steps=train_config.local_steps,
        )

    def evaluate(self, task_adapter: Any, sample_ids: list[str], mode: str) -> dict[str, float]:
        target = self._target(task_adapter.task_key, sample_ids)
        vector = self._state["adapter"]
        loss = sum(
            (value - wanted) ** 2 for value, wanted in zip(vector, target, strict=True)
        ) / len(vector)
        score = 1.0 / (1.0 + loss)
        metrics: dict[str, float] = {"loss": loss, "score": score}
        if task_adapter.task_key == "fast":
            metrics.update({"accuracy": score, "auc": min(1.0, score + 0.025)})
        else:
            metrics.update({"exact": score, "official_accuracy": score})
        return metrics
