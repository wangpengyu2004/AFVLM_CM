"""Private component-value regression. No router is attached to the VLM/DDP graph."""

from __future__ import annotations

from typing import Any


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
        self, dimensions: list[tuple[int, int]], device: Any, seed: int, params: dict
    ) -> None:
        import torch

        self.params, self.scales, self.observed = dict(params), {}, {}
        self.steps = 0
        self.heads = torch.nn.ModuleDict()
        devices = [device.index] if device.type == "cuda" else []
        # Initialization must not consume the VLM's random stream.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for din, dout in sorted(set(dimensions)):
                hidden = int(params.get("router_hidden_size", 32))
                self.heads[self.key(din, dout)] = torch.nn.Sequential(
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
    def key(din: int, dout: int) -> str:
        return f"d{din}_{dout}"

    def predict(self, a: Any, b: Any) -> Any:
        import torch

        with torch.no_grad():
            return self.heads[self.key(a.shape[1], b.shape[0])](
                component_features(a, b, float(self.params.get("eps", 1e-12)))
            ).flatten()

    def learn(self, observations: dict) -> dict:
        """Detached pre-step observations only; invalid/zero-label modules are not targets."""
        import torch
        import torch.nn.functional as functional

        beta = float(self.params.get("router_label_beta", 0.9))
        eps = float(self.params.get("eps", 1e-12))
        losses, validities, magnitudes = [], [], []
        with torch.enable_grad():
            for name, (key, features, labels) in observations.items():
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
                loss = functional.huber_loss(
                    predicted,
                    target,
                    reduction="mean",
                    delta=float(self.params.get("router_huber_delta", 1.0)),
                )
                losses.append(loss * valid)
                validities.append(valid.float())
                magnitudes.append(magnitude)
            count = torch.stack(validities).sum()
            loss = torch.stack(losses).sum() / count.clamp_min(1)
            summary = torch.stack((loss.detach(), count, torch.stack(magnitudes).mean())).cpu()
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
        return {
            "router_loss": float(summary[0]),
            "router_valid_modules": int(summary[1]),
            "label_abs_mean": float(summary[2]),
            "router_lifetime_steps": self.steps,
        }

    def state_dict(self) -> dict:
        return {
            "heads": self.heads.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scales": self.scales,
            "observed": self.observed,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.heads.load_state_dict(state["heads"])
        self.optimizer.load_state_dict(state["optimizer"])
        device = next(self.heads.parameters()).device
        self.scales = {name: value.to(device).detach() for name, value in state["scales"].items()}
        self.observed = {
            name: value.to(device).detach() for name, value in state["observed"].items()
        }
        self.steps = state["steps"]
