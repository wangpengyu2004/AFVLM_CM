"""Private component-value regression. No router is attached to the VLM/DDP graph."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def conflict_topk(scores: Any, observations: dict, k: int, penalty: float,
                  eps: float, seed: int) -> tuple[Any, dict]:
    """Normalize signed local value, subtract ONLY valid task-specific conflict.

    Stable seeded tie-breaking never consumes the main model/dropout RNG. No
    neutral prior, random replacement, exclusivity constraint or extra v label.
    """
    import torch

    with torch.no_grad():
        scores = scores.detach().float()
        if scores.ndim != 1 or not bool(torch.isfinite(scores).all()):
            raise ValueError("Invalid private router scores")
        if not 0 < k <= scores.numel():
            raise ValueError("Invalid calibrated rank")
        conflict = scores.new_tensor(observations["penalty"])
        valid = torch.tensor(observations["valid"], device=scores.device, dtype=torch.bool)
        if (conflict.shape != scores.shape or valid.shape != scores.shape
                or not bool(torch.isfinite(conflict).all())
                or bool(((conflict < 0) | (conflict > 1)).any())):
            raise ValueError("Invalid task-specific conflict observations")
        normalized = scores / (scores.abs().max() + eps)
        applied = torch.where(valid, conflict, torch.zeros_like(conflict))
        final = normalized - penalty * applied
        generator = torch.Generator(device=scores.device).manual_seed(seed)
        order = torch.randperm(scores.numel(), generator=generator, device=scores.device)
        chosen = order[torch.argsort(final[order], descending=True, stable=True)[:k]]
        original = order[torch.argsort(normalized[order], descending=True, stable=True)[:k]]
        changed = len(set(chosen.cpu().tolist()) - set(original.cpu().tolist()))
        return chosen, {
            "scores": scores.cpu().tolist(), "normalized_scores": normalized.cpu().tolist(),
            "penalty": applied.cpu().tolist(), "valid": valid.cpu().tolist(),
            "peer_count": observations["peer_count"],
            "own_direction_valid": observations["own_direction_valid"],
            "final_scores": final.cpu().tolist(), "penalty_strength": penalty,
            "selected_valid_comparison_fraction": float(valid[chosen].float().mean().cpu()),
            "candidate_valid_comparison_fraction": float(valid.float().mean().cpu()),
            "changed_components": changed, "router_only_indices": sorted(original.cpu().tolist()),
        }


def component_features(a: Any, b: Any, eps: float) -> Any:
    """One component per row: normalized A/B directions and their original norms."""
    import torch

    with torch.no_grad():
        x, y = a.detach().float(), b.detach().float().T
        na, nb = x.norm(dim=1, keepdim=True), y.norm(dim=1, keepdim=True)
        return torch.cat((x / na.clamp_min(eps), y / nb.clamp_min(eps), na, nb), dim=1)


def component_targets(b: Any, gradient: Any) -> Any:
    """Signed first-order deletion cost; read before the main optimizer step."""
    return -(b.detach().float() * gradient.detach().float()).sum(dim=0)


class PrivateRouter:
    def __init__(
        self, dimensions: list[tuple[int, int]], device: Any, seed: int, params: dict,
        module_metadata: Mapping[str, Mapping] | None = None,
    ) -> None:
        import torch

        self.params, self.scales, self.observed = dict(params), {}, {}
        # Missing field preserves published/frozen dimension-sharing recipes.
        self.sharing = str(params.get("router_sharing", "dimension"))
        if self.sharing not in {"dimension", "layer"}:
            raise ValueError("router_sharing must be dimension or layer")
        self.module_heads = {}
        layout = {self.key(din, dout): (din, dout) for din, dout in sorted(set(dimensions))}
        if self.sharing == "layer":
            if not module_metadata:
                raise ValueError("Per-layer Router requires real LoRA module metadata")
            layout, layer_dimensions = {}, {}
            for name, metadata in sorted(module_metadata.items()):
                din, dout = metadata["in_features"], metadata["out_features"]
                layer = self.layer_id(name)
                if layer in layer_dimensions and layer_dimensions[layer] != (din, dout):
                    raise ValueError(
                        "Per-layer Router requires equal feature dimensions within each layer; "
                        "the current LLaVA Q/K/V/O recipe satisfies this constraint"
                    )
                layer_dimensions[layer] = (din, dout)
                key = self.key(din, dout, name, self.sharing)
                self.module_heads[name] = key
                layout[key] = (din, dout)
            if set(dimensions) != set(layout.values()):
                raise ValueError("Router module metadata and feature dimensions disagree")
        self.head_dimensions = layout
        self.steps = 0
        self.heads = torch.nn.ModuleDict()
        devices = [device.index] if device.type == "cuda" else []
        # Initialization must not consume the VLM's random stream.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for key, (din, dout) in layout.items():
                hidden = int(params.get("router_hidden_size", 32))
                self.heads[key] = torch.nn.Sequential(
                    torch.nn.Linear(din + dout + 2, hidden, device=device, dtype=torch.float32),
                    torch.nn.SiLU(),
                    torch.nn.Linear(hidden, 1, device=device, dtype=torch.float32),
                )
        self.optimizer = torch.optim.AdamW(
            self.heads.parameters(),
            lr=float(params.get("router_learning_rate", 0.001)),
            weight_decay=0.0,
            eps=1e-8,
        )

    @staticmethod
    def layer_id(module_name: str) -> int:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
        if match is None:
            raise ValueError(f"Cannot determine Transformer layer from LoRA module: {module_name}")
        return int(match[1])

    @staticmethod
    def key(din: int, dout: int, module_name: str | None = None,
            sharing: str = "dimension") -> str:
        if sharing == "dimension":
            return f"d{din}_{dout}"
        if sharing != "layer" or module_name is None:
            raise ValueError("Per-layer Router scoring requires a module identity")
        return f"layer{PrivateRouter.layer_id(module_name)}_d{din}_{dout}"

    def predict(self, a: Any, b: Any, module_name: str | None = None) -> Any:
        import torch

        with torch.no_grad():
            if self.sharing == "layer" and module_name not in self.module_heads:
                raise ValueError("Per-layer Router scoring requires a registered LoRA module")
            return self.heads[self.key(a.shape[1], b.shape[0], module_name, self.sharing)](
                component_features(a, b, float(self.params.get("eps", 1e-12)))
            ).flatten()

    def learn(self, observations: dict) -> dict:
        """Detached pre-step observations only; invalid/zero-label modules are not targets."""
        import os

        import torch
        import torch.nn.functional as functional

        beta = float(self.params.get("router_label_beta", 0.9))
        eps = float(self.params.get("eps", 1e-12))
        losses, validities, magnitudes = [], [], []
        rank_weight = float(self.params.get("router_rank_lambda", 0.0))
        rank_gap = float(self.params.get("router_rank_min_gap", 0.1))
        rank_losses, rank_counts = [], []
        # The primary replica owns diagnostics only. No extra DDP collectives,
        # stochastic operations, model passes, gradient edits or loss terms.
        diagnostics = os.environ.get("AFVLM_PRIMARY_PROCESS", "1") == "1"
        ranking_names, ranking_values = [], []
        with torch.enable_grad():
            for name, (key, features, labels) in observations.items():
                if self.sharing == "layer" and self.module_heads.get(name) != key:
                    raise ValueError(
                        "Router supervision was assigned to the wrong Transformer layer"
                    )
                labels, features = labels.detach(), features.detach()
                magnitude = labels.abs().mean()
                valid = magnitude > float(self.params.get("router_label_min", 1e-12))
                previous = self.scales.get(name, torch.zeros_like(magnitude))
                seen = self.observed.get(name, torch.zeros_like(valid))
                scale = torch.where(
                    valid,
                    torch.where(seen, beta * previous + (1 - beta) * magnitude, magnitude),
                    previous,
                ).detach()
                self.scales[name], self.observed[name] = scale, seen | valid
                target = (labels / (scale + eps)).detach()
                predicted = self.heads[key](features).flatten()
                if diagnostics:
                    from afl_vlm.methods.ours_router.ranking import ranking_tensor

                    ranking_names.append(name)
                    ranking_values.append(ranking_tensor(predicted, labels, valid))
                loss = functional.huber_loss(
                    predicted,
                    target,
                    reduction="mean",
                    delta=float(self.params.get("router_huber_delta", 1.0)),
                )
                losses.append(loss * valid)
                validities.append(valid.float())
                magnitudes.append(magnitude)
                if rank_weight > 0:
                    from afl_vlm.methods.ours_router.losses import pairwise_rank_loss

                    rank_loss, pairs = pairwise_rank_loss(predicted, target, rank_gap, valid)
                    rank_losses.append(rank_loss)
                    rank_counts.append(pairs)
            count = torch.stack(validities).sum()
            loss = torch.stack(losses).sum() / count.clamp_min(1)
            regression = loss
            rank_loss = loss.detach().new_zeros(())
            pair_count = rank_modules = rank_loss
            if rank_weight > 0:
                pairs = torch.stack(rank_counts)
                rank_modules = (pairs > 0).sum()
                pair_count = pairs.sum()
                rank_loss = torch.stack(rank_losses).sum() / rank_modules.clamp_min(1)
                loss = regression + rank_weight * rank_loss
            summary = torch.stack((loss.detach(), count, torch.stack(magnitudes).mean(),
                                   regression.detach(), rank_loss.detach(), pair_count,
                                   rank_modules))
            if diagnostics:
                # One small transfer per complete step, not per rank/module.
                combined = torch.cat((summary, torch.stack(ranking_values).flatten())).cpu()
                summary = combined[:7]
            else:
                summary = summary.cpu()
            if not bool(torch.isfinite(summary).all()):
                raise ValueError("Non-finite router supervision")
            self.optimizer.zero_grad(set_to_none=True)
            if summary[1] > 0:
                if not bool(torch.isfinite(loss)):
                    raise ValueError("Non-finite private router loss")
                loss.backward()  # Only this small private MLP, never the VLM.
                torch.nn.utils.clip_grad_norm_(
                    self.heads.parameters(),
                    float(self.params.get("router_max_grad_norm", 1.0)),
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.steps += 1
            # These gradients have already been consumed. Keeping them until
            # this client participates again costs one full private Router per
            # idle client on EVERY DDP replica. Adam moments/weights are kept.
            self.optimizer.zero_grad(set_to_none=True)
        report = {
            "router_loss": float(summary[0]),
            "router_valid_modules": int(summary[1]),
            "label_abs_mean": float(summary[2]),
            "router_lifetime_steps": self.steps,
            "router_regression_loss": float(summary[3]),
            "router_rank_loss": float(summary[4]),
            "router_rank_pairs": int(summary[5]),
            "router_rank_valid_modules": int(summary[6]),
            "router_rank_lambda": rank_weight,
            "router_rank_min_gap": rank_gap,
        }
        if diagnostics:
            from afl_vlm.methods.ours_router.ranking import FIELDS, ranking_report

            ranking = ranking_report(ranking_names, combined[7:].reshape(-1, len(FIELDS)).tolist())
            report["router_ranking"] = ranking
            for metric in ("spearman", "topk_overlap", "top1_overlap"):
                report[f"router_{metric}_mean"] = ranking[f"{metric}_mean"]
                report[f"router_{metric}_valid_modules"] = ranking[f"{metric}_valid_modules"]
        return report

    def state_dict(self) -> dict:
        result = {
            "heads": self.heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scales": self.scales,
            "observed": self.observed,
            "steps": self.steps,
        }
        if self.sharing == "layer":
            result["router_layout"] = self.layout_state()
        return result

    def layout_state(self) -> dict:
        return {"schema_version": 1, "sharing": self.sharing,
                "hidden_size": int(self.params.get("router_hidden_size", 32)),
                "hidden_layers": 1, "activation": "SiLU", "module_heads": dict(self.module_heads),
                "head_dimensions": dict(self.head_dimensions)}

    def load_state_dict(self, state: dict) -> None:
        if self.sharing == "layer":
            if state.get("router_layout") != self.layout_state():
                raise ValueError(
                    "Private Router layer layout differs; start a new experiment rather than "
                    "silently converting dimension-sharing weights"
                )
        elif "router_layout" in state:
            raise ValueError("Cannot load per-layer private Router into dimension-sharing mode")
        self.heads.load_state_dict(state["heads"])
        self.optimizer.load_state_dict(state["optimizer"])
        device = next(self.heads.parameters()).device
        self.scales = {name: value.to(device).detach() for name, value in state["scales"].items()}
        self.observed = {
            name: value.to(device).detach() for name, value in state["observed"].items()
        }
        self.steps = state["steps"]
