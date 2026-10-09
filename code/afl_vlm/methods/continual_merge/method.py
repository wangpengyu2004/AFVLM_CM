"""Continual merging baselines adapted to AFVLM-CM asynchronous LoRA updates.

Not exact reproductions: original independently trained common-origin experts
are replaced by download-relative functional increments; deployment uses the
same fixed-rank LoRA and frozen LLaVA as the other methods. See
docs/continual_merging.md for equations, citations and all adaptation boundaries.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method, staleness_weight
from afl_vlm.methods.continual_merge.algebra import (
    compact_coordinates,
    lora_pairs,
    opcm_filter,
    projection_loss,
    refactor,
    spectrum,
)
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState, clone_state


class ContinualLoRAMerge(Method):
    capabilities = MethodCapabilities(mode="asynchronous", requires_staleness=True)
    common_params = {"merge_weight", "staleness", "eps"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.arrivals = 0
        self.norm_sum = 0.0
        self.opcm_scale = 1.0
        weight, eps = float(self.params.get("merge_weight", 0.5)), self.eps
        if not math.isfinite(weight) or not 0 < weight <= 1:
            raise ValueError("merge_weight must be in (0, 1]")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        stale_config = self.params.get("staleness", {"type": "constant"})
        if any(
            not math.isfinite(staleness_weight(t, stale_config))
            or not 0 < staleness_weight(t, stale_config) <= 1
            for t in (0, 1, 100)
        ):
            raise ValueError("staleness weighting must be finite and in (0, 1]")

    @property
    def eps(self) -> float:
        return float(self.params.get("eps", 1e-8))

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        config = getattr(model, "_config", {})
        lora = config.get("lora", {})
        if lora.get("train_mm_projector", False) or lora.get("bias", "none") != "none":
            raise ValueError(f"{self.name} requires LoRA-only weights (no projector/bias training)")
        if lora.get("use_dora", False) or lora.get("use_rslora", False):
            raise ValueError(f"{self.name} currently supports ordinary LoRA only")

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        import torch

        # Common functional origin is W0: ordinary PEFT initializes every B=0.
        # Fail rather than silently assuming a zero origin for pre-tuned adapters.
        for a_name, b_name in lora_pairs(initial_state).values():
            a, b = initial_state[a_name], initial_state[b_name]
            if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
                raise TypeError("Continual merging requires torch LoRA tensors")
            if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
                raise ValueError("Invalid LoRA A/B shapes")
            if a.shape[0] > min(a.shape[1], b.shape[0]):
                raise ValueError("LoRA rank must not exceed projection dimensions")
            if not bool(torch.isfinite(a).all() & torch.isfinite(b).all()):
                raise ValueError("Non-finite initial LoRA state")
            if bool(torch.count_nonzero(b)):
                raise ValueError(
                    "Continual baselines require zero-B initialization, "
                    "not a pre-tuned adapter"
                )

    def merge_core(self, old: Any, incoming: Any, seed: int) -> tuple[Any, dict[str, Any]]:
        raise NotImplementedError

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        import torch

        stale = server_context.version - update.base_version
        weight = float(self.params.get("merge_weight", 0.5)) * staleness_weight(
            stale, self.params.get("staleness", {"type": "constant"})
        )
        states = (server_context.global_state, update.local_state, update.base_state)
        pairs = lora_pairs(states[0])
        if any(set(state) != set(states[0]) for state in states[1:]):
            raise ValueError("Global, local and immutable download LoRA state keys differ")
        output = clone_state(states[0])
        pieces: list[tuple[Any, ...]] = []
        records: dict[str, Any] = {}
        incoming_energy = 0.0
        raw_energy = 0.0
        # All matrix algebra stays on the server state's device, generally CUDA.
        with torch.no_grad():
            for index, (module, (a_name, b_name)) in enumerate(pairs.items()):
                a, b = states[0][a_name], states[0][b_name]
                qo, qi, cores = compact_coordinates(
                    [(state[a_name], state[b_name]) for state in states], b.device
                )
                old, local, base = cores
                incoming = local - base  # includes LoRA product cross terms, not dB @ dA
                if float(incoming.square().sum()) <= self.eps**2:
                    merged, details = old.clone(), {"zero_increment": True}
                    if self.name == "opcm_lora":
                        # Global OPCM scaling includes every old module, including
                        # modules for which this client contributed no increment.
                        merged = self.opcm_scale * old
                else:
                    merged, details = self.merge_core(old, incoming, update.seed + index)
                if not bool(torch.isfinite(merged).all()):
                    raise FloatingPointError(f"Non-finite {self.name} merge in {module}")
                incoming_energy += float(incoming.square().sum())
                raw_energy += float(merged.square().sum())
                pieces.append((module, a_name, b_name, qo, qi, old, merged))
                records[module] = details
            new_count = self.arrivals + 1
            new_norm_sum = self.norm_sum + math.sqrt(incoming_energy)
            # OPCM adaptive norm control is global over all modules, not per-layer.
            scale = 1.0
            if self.name == "opcm_lora" and incoming_energy > self.eps**2:
                mean_norm = new_norm_sum / new_count
                scale = math.sqrt(raw_energy) / max(mean_norm, self.eps)
                scale = max(scale, self.eps)
            for module, a_name, b_name, qo, qi, old, merged in pieces:
                if incoming_energy <= self.eps**2 or (
                    self.name != "opcm_lora" and records[module].get("zero_increment", False)
                ):
                    # Keep unchanged A/B coordinates. OPCM may still rescale an
                    # individually unchanged module through its global norm rule.
                    records[module].update(
                        {"compression_relative_error": 0.0, "retained_energy_ratio": 1.0}
                    )
                    continue
                candidate = old + weight * (merged / scale - old)
                a, b, diagnostics = refactor(
                    candidate,
                    qo,
                    qi,
                    states[0][a_name].shape[0],
                    states[0][a_name],
                    states[0][b_name],
                    self.eps,
                )
                output[a_name], output[b_name] = a, b
                records[module].update(diagnostics)
            self.arrivals = new_count
            self.norm_sum = new_norm_sum
            if self.name == "opcm_lora" and incoming_energy > self.eps**2:
                # Scaling and rank compression are explicit AFVLM adaptations;
                # the original recurrence's lambda refers to the raw unblended merge.
                self.opcm_scale = scale
        return [
            ServerMutation(
                output,
                weight,
                [update.update_id],
                metadata={
                    "continual_merge": self.name,
                    "adaptation": "download_relative_functional_increment_fixed_rank",
                    "staleness": stale,
                    "merge_weight": weight,
                    "merge_count": self.arrivals,
                    "incoming_functional_norm": math.sqrt(incoming_energy),
                    "opcm_scale": scale if self.name == "opcm_lora" else None,
                    "modules": records,
                },
            )
        ]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "params": self.params,
                "arrivals": self.arrivals,
                "norm_sum": self.norm_sum,
                "opcm_scale": self.opcm_scale,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        # Revalidate checkpoint hyperparameters instead of bypassing the constructor.
        self.__init__(state["params"])
        self.arrivals = int(state["arrivals"])
        self.norm_sum = float(state["norm_sum"])
        self.opcm_scale = float(state["opcm_scale"])
        if self.arrivals < 0 or not math.isfinite(self.norm_sum) or self.norm_sum < 0:
            raise ValueError("Invalid continual merging checkpoint counters")
        if not math.isfinite(self.opcm_scale) or self.opcm_scale <= 0:
            raise ValueError("Invalid OPCM scale")


@register_method("opcm_lora")
class OPCMLoRA(ContinualLoRAMerge):
    """OPCM: Merging Models on the Fly Without Retraining, NeurIPS 2025.

    Retains singular-basis diagonal/principal-block removal and adaptive norm
    control. Zero-singular null completion, rebasing, relaxation and rank-r
    deployment differ from the dense independent-expert algorithm.
    """

    name = "opcm_lora"
    allowed_params = ContinualLoRAMerge.common_params | {"retain"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        if not 0 < float(self.params.get("retain", 0.5)) <= 1:
            raise ValueError("OPCM retain must be in (0, 1]")

    def merge_core(self, old: Any, incoming: Any, seed: int) -> tuple[Any, dict[str, Any]]:
        clean, rank = opcm_filter(incoming, old, float(self.params.get("retain", 0.5)), self.eps)
        return self.opcm_scale * old + clean, {
            "protected_rank": rank,
            "filtered_functional_norm": float(clean.norm()),
            "removed_functional_norm": float((incoming - clean).norm()),
        }


@register_method("dop_lora")
class DOPLoRA(ContinualLoRAMerge):
    """DOP: Continual Model Merging without Data, NeurIPS 2025, Eq. (2)-(3).

    Optimize both singular-value-weighted left/right projection losses using the
    analytical two-objective MGDA coefficient. Compact-coordinate Adam is an
    explicit optimizer adaptation (not invariant to full-matrix Adam coordinates).
    """

    name = "dop_lora"
    allowed_params = ContinualLoRAMerge.common_params | {
        "inner_steps",
        "inner_lr",
        "retain",
        "mgda_ema_beta",
        "mgda_initial_alpha",
        "normalize_gradients",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_inner(self.params)
        if not 0 < float(self.params.get("retain", 1.0)) <= 1:
            raise ValueError("DOP retain must be in (0, 1]")
        if not 0 <= float(self.params.get("mgda_ema_beta", 0.99)) < 1:
            raise ValueError("mgda_ema_beta must be in [0, 1)")
        if not 0 <= float(self.params.get("mgda_initial_alpha", 0.5)) <= 1:
            raise ValueError("mgda_initial_alpha must be in [0, 1]")

    def merge_core(self, old: Any, incoming: Any, seed: int) -> tuple[Any, dict[str, Any]]:
        import torch

        new = old + incoming  # arrival-time rebased expert target
        retain = float(self.params.get("retain", 1.0))
        left = spectrum(old, retain, eps=self.eps)
        right = spectrum(new, retain, eps=self.eps)
        with torch.enable_grad():
            value = ((old + new) / 2).detach().clone().requires_grad_(True)
            optimizer = torch.optim.Adam([value], lr=float(self.params.get("inner_lr", 1e-4)))
            alpha = value.new_tensor(float(self.params.get("mgda_initial_alpha", 0.5)))
            for _ in range(int(self.params.get("inner_steps", 50))):
                ls = projection_loss(value, old, *left)
                lp = projection_loss(value, new, *right)
                gs = torch.autograd.grad(ls, value, retain_graph=True)[0]
                gp = torch.autograd.grad(lp, value, retain_graph=True)[0]
                if self.params.get("normalize_gradients", True):
                    gs = gs / ls.detach().clamp_min(self.eps)
                    gp = gp / lp.detach().clamp_min(self.eps)
                difference = gs - gp
                solution = (
                    (gp * (gp - gs)).sum() / difference.square().sum().clamp_min(self.eps**2)
                ).clamp(0, 1)
                beta = float(self.params.get("mgda_ema_beta", 0.99))
                alpha = beta * alpha + (1 - beta) * solution.detach()
                optimizer.zero_grad(set_to_none=True)
                (alpha * ls + (1 - alpha) * lp).backward()
                optimizer.step()
            merged = value.detach()
        return merged, {
            "protected_rank": len(left[1]),
            "new_rank": len(right[1]),
            "mgda_alpha": float(alpha),
            "inner_steps": int(self.params.get("inner_steps", 50)),
            "stability_loss": float(projection_loss(merged, old, *left)),
            "plasticity_loss": float(projection_loss(merged, new, *right)),
        }


@register_method("nufilt_lora")
class NUFILTLoRA(ContinualLoRAMerge):
    """NUFILT, ICLR 2026: filter Eq. (8), residual gate Eq. (15)-(17).

    Follows the official FilterLoRA's dimensionally consistent new @ (P + BA),
    not the shorthand additive residual in paper Eq. (10). Fits only a server
    matrix surrogate; no model forward/backward or raw server data is needed.
    """

    name = "nufilt_lora"
    allowed_params = ContinualLoRAMerge.common_params | {
        "inner_steps",
        "inner_lr",
        "filter_rank",
        "projection_rank",
        "adaptation_rank",
        "init_std",
        "stability_weight",
        "plasticity_weight",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        _validate_inner(self.params)
        for key, default in (("filter_rank", 8), ("projection_rank", 8), ("adaptation_rank", 8)):
            if int(self.params.get(key, default)) <= 0:
                raise ValueError(f"{key} must be positive")
        for key, default in (
            ("init_std", 0.02),
            ("stability_weight", 1.0),
            ("plasticity_weight", 1.0),
        ):
            number = float(self.params.get(key, default))
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{key} must be finite and positive")

    def merge_core(self, old: Any, incoming: Any, seed: int) -> tuple[Any, dict[str, Any]]:
        import torch

        new = old + incoming
        vo = spectrum(old, rank_limit=int(self.params.get("filter_rank", 8)), eps=self.eps)[2]
        vn = spectrum(new, rank_limit=int(self.params.get("projection_rank", 8)), eps=self.eps)[2]
        filtered = new - (new @ vo) @ vo.T
        initial = old + filtered
        dimension = old.shape[1]
        rank = min(dimension, int(self.params.get("adaptation_rank", 8)))
        generator = torch.Generator(device=old.device).manual_seed(seed % (2**63 - 1))
        with torch.enable_grad():
            a = (
                torch.randn(rank, dimension, device=old.device, generator=generator)
                * float(self.params.get("init_std", 0.02))
            ).requires_grad_(True)
            b = torch.zeros(dimension, rank, device=old.device, requires_grad=True)
            optimizer = torch.optim.Adam([a, b], lr=float(self.params.get("inner_lr", 1e-3)))
            for _ in range(int(self.params.get("inner_steps", 50))):
                merged = initial + (new @ b) @ a
                ls = ((merged - old) @ vo).square().sum()
                lp = ((merged - new) @ vn).square().sum()
                objective = (
                    float(self.params.get("stability_weight", 1.0)) * ls
                    + float(self.params.get("plasticity_weight", 1.0)) * lp
                )
                optimizer.zero_grad(set_to_none=True)
                objective.backward()
                optimizer.step()
            merged = (initial + (new @ b) @ a).detach()
        return merged, {
            "protected_rank": vo.shape[1],
            "new_rank": vn.shape[1],
            "adaptation_rank": rank,
            "inner_steps": int(self.params.get("inner_steps", 50)),
            "filtered_functional_norm": float(filtered.norm()),
            "stability_loss": float(((merged - old) @ vo).square().sum()),
            "plasticity_loss": float(((merged - new) @ vn).square().sum()),
            "residual_norm": float((merged - initial).norm()),
        }


def _validate_inner(params: Mapping[str, Any]) -> None:
    if int(params.get("inner_steps", 50)) <= 0:
        raise ValueError("inner_steps must be positive")
    rate = float(params.get("inner_lr", 1e-4))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("inner_lr must be finite and positive")
