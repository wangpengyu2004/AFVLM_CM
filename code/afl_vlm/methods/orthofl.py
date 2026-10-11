"""OrthoFL: Orthogonal Calibration for Asynchronous Federated Learning.

Source: https://arxiv.org/html/2502.15940v1 (2025 preprint), Eqs. 4--8.
Adaptation: orthogonalize concatenated LoRA A/B coordinates per transformer
layer, never frozen backbone weights. Client and global trajectories are
separate; calibration changes the NEXT client download, not the global update.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import (
    ClientContext,
    MethodCapabilities,
    ServerContext,
    ServerMutation,
    Update,
)
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, add_scaled, clone_state, state_dot, subtract


@register_method("orthofl")
class OrthoFL(Method):
    name = "orthofl"
    capabilities = MethodCapabilities(
        mode="asynchronous", requires_local_state=True, requires_staleness=True
    )
    allowed_params = {"beta", "a", "eps"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        beta, a, eps = (
            float(self.params.get(k, d)) for k, d in (("beta", 0.6), ("a", 0.5), ("eps", 1e-12))
        )
        if (
            not all(math.isfinite(v) for v in (beta, a, eps))
            or not 0 < beta <= 1
            or a < 0
            or eps <= 0
        ):
            raise ValueError("OrthoFL requires 0 < beta <= 1, a >= 0, eps > 0")
        self.client_models: dict[str, LoRAState] = {}
        self.global_references: dict[str, LoRAState] = {}
        self.reference_versions: dict[str, int] = {}
        self.downloads: dict[tuple[str, int], LoRAState] = {}

    def prepare_download(self, global_state: LoRAState, context: ClientContext) -> LoRAState:
        client = context.client_id
        if client not in self.client_models:
            self.client_models[client] = clone_state(global_state)
            self.global_references[client] = clone_state(global_state)
            self.reference_versions[client] = context.base_version
        start = clone_state(self.client_models[client])
        self.downloads[(client, context.local_round)] = clone_state(start)
        return start

    def prepare_upload(self, update: Update, context: ClientContext) -> Update:
        start = self.downloads.pop((context.client_id, context.local_round))
        update.base_state = start
        update.delta = subtract(update.local_state, start)
        return update

    @staticmethod
    def layer_groups(state: LoRAState) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in sorted(state):
            match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            # For non-transformer trainables, each parameter is its own layer.
            layer = f"layer{match[1]}" if match else name
            groups.setdefault(layer, []).append(name)
        return groups

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        age = server_context.version - self.reference_versions.get(
            update.client_id, update.base_version
        )
        if age < 0:
            raise ValueError("OrthoFL received a future base version")
        reference = self.global_references.get(update.client_id, update.base_state)
        global_shift = subtract(server_context.global_state, reference)
        perpendicular = clone_state(global_shift)
        residuals = {}
        for layer, names in self.layer_groups(global_shift).items():
            local = {n: update.delta[n] for n in names}
            shift = {n: global_shift[n] for n in names}
            norm2 = state_dot(local, local)
            coefficient = (
                state_dot(shift, local) / norm2
                if norm2 > float(self.params.get("eps", 1e-12))
                else 0.0
            )
            orthogonal = add_scaled(shift, local, -coefficient)
            perpendicular.update(orthogonal)
            residuals[layer] = state_dot(orthogonal, local)
        weight = float(self.params.get("beta", 0.6)) * (age + 1) ** (
            -float(self.params.get("a", 0.5))
        )
        # Eq. 4: global receives the RAW uploaded model, not the calibrated one.
        new_global = add_scaled(
            server_context.global_state,
            subtract(update.local_state, server_context.global_state),
            weight,
        )
        # Eqs. 5--8: calibrate only with other clients' PRE-arrival drift.
        self.client_models[update.client_id] = add_scaled(update.local_state, perpendicular, 1.0)
        self.global_references[update.client_id] = clone_state(new_global)
        self.reference_versions[update.client_id] = server_context.version + 1
        return [
            ServerMutation(
                new_global,
                weight,
                [update.update_id],
                metadata={
                    "orthofl": True,
                    "paper_staleness": age + 1,
                    "orthogonality_residuals": residuals,
                    "projection_space": "layer_concatenated_federated_parameters",
                    "calibration_target": "next_client_model",
                    "global_receives_raw_local_model": True,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                **super().state_dict(),
                "client_models": self.client_models,
                "global_references": self.global_references,
                "reference_versions": self.reference_versions,
                "downloads": self.downloads,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.client_models = copy.deepcopy(state["client_models"])
        self.global_references = copy.deepcopy(state["global_references"])
        self.reference_versions = copy.deepcopy(state["reference_versions"])
        self.downloads = copy.deepcopy(state["downloads"])
