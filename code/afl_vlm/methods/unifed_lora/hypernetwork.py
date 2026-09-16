"""Small server hypernetwork that emits a bounded LoRA conditioning gate."""

from __future__ import annotations

from typing import Any


class LoRAHypernetwork:
    def __init__(
        self,
        descriptor_size: int,
        hidden_size: int,
        learning_rate: float,
        seed: int,
        device: Any = "cpu",
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.module = torch.nn.Sequential(
            torch.nn.Linear(descriptor_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Tanh(),
        ).to(device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)

    def gate(self, descriptor: list[float]) -> Any:
        tensor = self.torch.tensor(
            descriptor,
            dtype=self.torch.float32,
            device=next(self.module.parameters()).device,
        )
        return self.module(tensor).detach().squeeze()

    def fit(self, descriptor: list[float], target: Any) -> Any:
        tensor = self.torch.tensor(
            descriptor,
            dtype=self.torch.float32,
            device=next(self.module.parameters()).device,
        )
        prediction = self.module(tensor).squeeze()
        target_tensor = (
            target.detach().to(device=prediction.device, dtype=prediction.dtype)
            if hasattr(target, "detach")
            else self.torch.tensor(float(target), device=prediction.device)
        )
        loss = (prediction - target_tensor) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def state_dict(self) -> dict[str, Any]:
        return {
            "module": {key: value.detach() for key, value in self.module.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
        }
