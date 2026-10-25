"""Hard Concrete equations 10--13 (Louizos et al., ICLR 2018).

Imported without torch so method/config discovery never loads a model or CUDA.
The gate distribution log_alpha is unrelated to LoRA's lora_alpha scaling.
"""

from __future__ import annotations

import math
from typing import Any


def active_probability(log_alpha: Any, temperature: float, lower: float, upper: float) -> Any:
    return (log_alpha - temperature * math.log(-lower / upper)).sigmoid()


def deterministic_gate(log_alpha: Any, lower: float, upper: float) -> Any:
    # Eq.13 deliberately has no temperature and is not an exact expectation.
    return (log_alpha.sigmoid() * (upper - lower) + lower).clamp(0.0, 1.0)


def sample_gate(
    log_alpha: Any, uniform: Any, temperature: float, lower: float, upper: float
) -> Any:
    uniform = uniform.clamp(1e-6, 1.0 - 1e-6)
    relaxed = ((uniform.log() - (-uniform).log1p() + log_alpha) / temperature).sigmoid()
    return (relaxed * (upper - lower) + lower).clamp(0.0, 1.0)


def make_gate(rank: int, reference: Any, params: dict[str, Any]) -> Any:
    import torch

    temperature = float(params.get("gate_temperature", 2.0 / 3.0))
    lower, upper = float(params.get("gate_lower", -0.1)), float(params.get("gate_upper", 1.1))
    probability = float(params.get("gate_initial_active_probability", 0.9))
    initial = math.log(probability / (1 - probability)) + temperature * math.log(-lower / upper)

    class HardConcreteGate(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_alpha = torch.nn.Parameter(
                torch.full((rank,), initial, dtype=torch.float32, device=reference.device)
            )
            self.register_buffer("fixed_z", reference.new_ones(rank).float(), persistent=False)
            self.phase = "off"

        def probability(self) -> Any:
            return active_probability(self.log_alpha, temperature, lower, upper)

        def deterministic(self) -> Any:
            return deterministic_gate(self.log_alpha, lower, upper)

        def reset(self) -> None:
            with torch.no_grad():
                self.log_alpha.fill_(initial)
                self.fixed_z.fill_(1)
            self.log_alpha.grad = None
            self.phase = "warmup"

        def freeze(self) -> None:
            with torch.no_grad():
                self.fixed_z.copy_(self.deterministic())
            self.phase = "fixed"

        def forward(self, latent: Any) -> Any:
            if not self.training or self.phase == "off":
                return latent
            if self.phase == "joint":
                z = sample_gate(
                    self.log_alpha, torch.rand_like(self.log_alpha), temperature, lower, upper
                )
            elif self.phase == "fixed":
                z = self.fixed_z + 0.0 * self.log_alpha
            else:
                z = torch.ones_like(self.log_alpha) + 0.0 * self.log_alpha
            # Keep the gate in DDP's graph even in warmup/fixed phases. Its zero
            # gradient does not update it then. Do not change requires_grad mid-DDP.
            return latent * z.to(dtype=latent.dtype)

    return HardConcreteGate()
