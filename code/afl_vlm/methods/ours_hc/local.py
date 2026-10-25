"""Instance-only LoRA hooks and a private gate optimizer for ours_hc.

The existing LoRA AdamW loop, batching, accumulation and total step budget stay
unchanged. Only this method's loaded model receives these extensions.
"""

from __future__ import annotations

import math
from functools import wraps
from types import MethodType
from typing import Any

from afl_vlm.methods.ours import ModuleSensitivityTracker, _discover_model_modules
from afl_vlm.methods.ours_hc.gates import make_gate


def phase_boundaries(total: int, warmup: int, tail_fraction: float) -> tuple[int, int]:
    if total < 3:
        raise ValueError("ours_hc needs >=3 optimizer steps for warmup/joint/fixed-tail phases")
    tail = min(total - 1, max(1, math.ceil(total * tail_fraction)))
    end_joint = total - tail
    return min(warmup, max(0, end_joint - 1)), end_joint


def _nonreentrant_forward(original: Any) -> Any:
    @wraps(original)
    def forward(*args: Any, **kwargs: Any) -> Any:
        import torch
        from torch.utils.checkpoint import checkpoint

        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        return checkpoint(original, *args, use_reentrant=False, **kwargs)

    return forward


class LocalGates:
    def __init__(self, model: Any, params: dict[str, Any]) -> None:
        self.model, self.params = model, dict(params)
        self.modules = _discover_model_modules(model)
        self.gates = {}
        network = model._network()
        by_a = {id(a): name for name, (a, _, _) in self.modules.items()}
        for module in list(network.modules()):
            weight = getattr(module, "weight", None)
            name = by_a.get(id(weight))
            if name is None:
                continue
            gate = make_gate(self.modules[name][2].rank, weight, self.params)
            module.add_module("_hc_gate", gate)
            self.gates[name] = gate
            module.register_forward_hook(self._hook(gate))
        if set(self.gates) != set(self.modules):
            raise ValueError("Could not install all language-side Hard Concrete gates")
        # Private gate parameters stay in DDP, but never enter the common LoRA
        # optimizer, model snapshots, downloads, global checkpoints or state store.
        original = model.named_federated_parameters

        def federated(this: Any) -> list:
            del this
            return [(name, parameter) for name, parameter in original() if "._hc_gate." not in name]

        model.named_federated_parameters = MethodType(federated, model)
        if model._config.get("gradient_checkpointing", True):
            # Reentrant checkpoints plus a gate used in L0 outside a checkpoint
            # can trigger DDP's ready-twice error. This method alone uses the
            # supported nonreentrant implementation, including keyword arguments.
            network.gradient_checkpointing_disable()
            backbone = network.get_base_model()
            for layer in backbone.get_model().layers:
                layer.forward = _nonreentrant_forward(layer.forward)
        self.optimizer = None
        self.tracker = None
        self.steps = 0
        self.trajectory = []
        self.total = 0

    @staticmethod
    def _hook(gate: Any) -> Any:
        def apply(module: Any, inputs: Any, output: Any) -> Any:
            del module, inputs
            return gate(output)

        return apply

    def start(self, total: int) -> None:
        import torch

        self.warmup, self.end_joint = phase_boundaries(
            total,
            int(self.params.get("gate_warmup_steps", 1)),
            float(self.params.get("fixed_tail_fraction", 0.2)),
        )
        for gate in self.gates.values():
            gate.reset()
            if self.warmup == 0:
                gate.phase = "joint"
        self.optimizer = torch.optim.AdamW(
            [gate.log_alpha for gate in self.gates.values()],
            lr=float(self.params.get("gate_learning_rate", 0.2)),
            weight_decay=0.0,
            eps=1e-8,
        )
        self.steps, self.total, self.trajectory = 0, total, []
        self.tracker = None

    def penalty(self) -> Any:
        value = sum(gate.probability().sum() for gate in self.gates.values())
        return value * (
            float(self.params.get("l0_lambda", 1e-4))
            if self.warmup <= self.steps < self.end_joint
            else 0.0
        )

    def observe(self) -> None:
        if self.steps < self.end_joint:
            return
        if self.tracker is None:
            self.tracker = ModuleSensitivityTracker(
                self.model,
                warmup_steps=0,
                uniform_mix=float(self.params.get("sensitivity_uniform_mix", 0.75)),
                eps=float(self.params.get("eps", 1e-12)),
            )
        self.tracker.update(self.steps)

    def step(self, step: int) -> None:
        import torch

        if step != self.steps + 1 or self.optimizer is None:
            raise RuntimeError("Hard Concrete hooks must run once per real optimizer step")
        phase = (
            "warmup"
            if self.steps < self.warmup
            else ("joint" if self.steps < self.end_joint else "fixed")
        )
        if phase == "joint":
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            packed = (
                torch.stack(
                    (
                        sum(gate.probability().sum() for gate in self.gates.values()),
                        torch.stack(
                            [gate.log_alpha.isfinite().all() for gate in self.gates.values()]
                        )
                        .all()
                        .float(),
                    )
                )
                .cpu()
                .tolist()
            )
            expected = float(packed[0])
        if not packed[1] or not math.isfinite(expected):
            raise ValueError("Non-finite Hard Concrete probability")
        self.trajectory.append(
            {"optimizer_step": step, "phase": phase, "expected_active_components": expected}
        )
        self.steps = step
        if step == self.end_joint:
            for gate in self.gates.values():
                gate.freeze()
        elif step == self.warmup:
            for gate in self.gates.values():
                gate.phase = "joint"

    def finish(self) -> dict:
        import torch

        if self.steps != self.total or self.tracker is None:
            raise RuntimeError("Incomplete Hard Concrete training or fixed-mask sensitivity tail")
        sensitivity, valid = self.tracker.finalize()
        names = sorted(self.gates)
        with torch.no_grad():
            values = torch.stack([self.gates[name].fixed_z for name in names]).cpu().tolist()
            probabilities = (
                torch.stack([self.gates[name].probability() for name in names])
                .detach()
                .cpu()
                .tolist()
            )
        selected = {
            name: [j for j, value in enumerate(row) if value > 0]
            for name, row in zip(names, values, strict=True)
        }
        result = {
            "module_sensitivity": sensitivity,
            "sensitivity_valid": valid,
            "sensitivity_observations": dict(self.tracker.count),
            "sensitivity_transform": self.tracker.diagnostics(),
            "lora_metadata": self.tracker.metadata(),
            "hc_selection": selected,
            "hc_gate_values": dict(zip(names, values, strict=True)),
            "hc_local": {
                "relearn_every_upload": True,
                "warmup_steps": self.warmup,
                "joint_steps": self.end_joint - self.warmup,
                "fixed_tail_steps": self.total - self.end_joint,
                "module_gate_summary": {
                    name: {
                        "expected_active_components": math.fsum(probability),
                        "retained_gate_mean": math.fsum(row) / len(selected[name])
                        if selected[name]
                        else 0.0,
                    }
                    for name, row, probability in zip(names, values, probabilities, strict=True)
                },
                "trajectory": self.trajectory,
            },
        }
        for gate in self.gates.values():
            gate.phase = "off"
        return result


def install(model: Any, params: dict[str, Any]) -> LocalGates:
    if not hasattr(model, "_ours_hc"):
        model._ours_hc = LocalGates(model, params)
    elif model._ours_hc.params != dict(params):
        raise ValueError("Do not reuse a loaded Hard Concrete model with a different gate recipe")
    return model._ours_hc
