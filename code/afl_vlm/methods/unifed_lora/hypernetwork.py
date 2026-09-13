"""Small server hypernetwork that emits a bounded LoRA conditioning gate."""

from __future__ import annotations

from typing import Any


class LoRAHypernetwork:
    def __init__(
        self, descriptor_size: int, hidden_size: int, learning_rate: float, seed: int
    ) -> None:
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.module = torch.nn.Sequential(
            torch.nn.Linear(descriptor_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Tanh(),
        )
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)

    def gate(self, descriptor: list[float]) -> float:
        tensor = self.torch.tensor(descriptor, dtype=self.torch.float32)
        return float(self.module(tensor).detach().item())

    def fit(self, descriptor: list[float], target: float) -> float:
        tensor = self.torch.tensor(descriptor, dtype=self.torch.float32)
        prediction = self.module(tensor).squeeze()
        loss = (prediction - float(target)) ** 2
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def state_dict(self) -> dict[str, Any]:
        return {
            "module": {
                key: value.detach().cpu() for key, value in self.module.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
        }
