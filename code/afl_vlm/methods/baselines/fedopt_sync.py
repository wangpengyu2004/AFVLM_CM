"""Synchronous FedAdam and FedYogi server optimizers from FedOpt."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, add_scaled, weighted_mean_states


def _map(value: Any, function: Any) -> Any:
    if isinstance(value, list):
        return [_map(item, function) for item in value]
    return function(value)


def _zip_map(left: Any, right: Any, function: Any) -> Any:
    if isinstance(left, list) and isinstance(right, list):
        return [_zip_map(a, b, function) for a, b in zip(left, right, strict=True)]
    return function(left, right)


def _zeros(state: Mapping[str, Any]) -> LoRAState:
    return {key: _map(value, lambda item: item * 0.0) for key, value in state.items()}


def _combine(left: Mapping[str, Any], right: Mapping[str, Any], function: Any) -> LoRAState:
    return {key: _zip_map(left[key], right[key], function) for key in left}


class _FedOptSync(Method):
    schedule_mode = "synchronous"
    allowed_params = {"server_lr", "beta1", "beta2", "tau", "sample_weighted"}
    optimizer = "base"

    def __init__(self, params=None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 0.01))
        self.beta1 = float(self.params.get("beta1", 0.9))
        self.beta2 = float(self.params.get("beta2", 0.99))
        self.tau = float(self.params.get("tau", 1e-3))
        self.sample_weighted = bool(self.params.get("sample_weighted", True))
        if self.server_lr <= 0 or self.tau <= 0:
            raise ValueError("FedOpt server_lr and tau must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("FedOpt beta1 and beta2 must be in [0, 1)")
        self.rounds: dict[int, list[Update]] = defaultdict(list)
        self.first_moment: LoRAState | None = None
        self.second_moment: LoRAState | None = None

    def _second_moment(self, previous: Any, squared: Any) -> Any:
        if self.optimizer == "adam":
            return self.beta2 * previous + (1.0 - self.beta2) * squared
        difference = previous - squared
        sign = (
            difference.sign()
            if hasattr(difference, "sign")
            else (1 if difference > 0 else (-1 if difference < 0 else 0))
        )
        return previous - (1.0 - self.beta2) * squared * sign

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        self.rounds[update.local_round].append(copy.deepcopy(update))
        pending = self.rounds[update.local_round]
        if len(pending) < server_context.expected_clients:
            return []
        del self.rounds[update.local_round]
        weights = [
            float(item.metadata.get("num_examples", 1.0)) if self.sample_weighted else 1.0
            for item in pending
        ]
        pseudo_gradient = weighted_mean_states((item.delta for item in pending), weights)
        if self.first_moment is None:
            self.first_moment = _zeros(pseudo_gradient)
            self.second_moment = _zeros(pseudo_gradient)
        assert self.second_moment is not None
        self.first_moment = _combine(
            self.first_moment,
            pseudo_gradient,
            lambda moment, value: self.beta1 * moment + (1.0 - self.beta1) * value,
        )
        squared = {
            key: _map(value, lambda item: item * item) for key, value in pseudo_gradient.items()
        }
        self.second_moment = _combine(self.second_moment, squared, self._second_moment)
        direction = _combine(
            self.first_moment,
            self.second_moment,
            lambda moment, variance: (
                moment
                / (
                    (variance.sqrt() if hasattr(variance, "sqrt") else math.sqrt(variance))
                    + self.tau
                )
            ),
        )
        return [
            ServerMutation(
                new_state=add_scaled(server_context.global_state, direction, self.server_lr),
                applied_weight=self.server_lr,
                contributing_update_ids=[item.update_id for item in pending],
                metadata={
                    "synchronous_round": update.local_round,
                    "server_optimizer": self.optimizer,
                    "sample_weights": weights,
                },
            )
        ]

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if self.rounds:
            incomplete = {round_id: len(items) for round_id, items in self.rounds.items()}
            raise RuntimeError(f"Incomplete synchronous FedOpt rounds: {incomplete}")
        return []

    def state_dict(self) -> dict[str, Any]:
        return {
            "params": copy.deepcopy(self.params),
            "rounds": copy.deepcopy(dict(self.rounds)),
            "first_moment": copy.deepcopy(self.first_moment),
            "second_moment": copy.deepcopy(self.second_moment),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.rounds = defaultdict(list, copy.deepcopy(dict(state.get("rounds", {}))))
        self.first_moment = copy.deepcopy(state.get("first_moment"))
        self.second_moment = copy.deepcopy(state.get("second_moment"))


@register_method("fedadam_sync")
class FedAdamSyncMethod(_FedOptSync):
    name = "fedadam_sync"
    optimizer = "adam"


@register_method("fedyogi_sync")
class FedYogiSyncMethod(_FedOptSync):
    name = "fedyogi_sync"
    optimizer = "yogi"
