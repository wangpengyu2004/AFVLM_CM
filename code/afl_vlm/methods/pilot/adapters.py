"""Pilot visual adapters and cross-task mixture used only by Pilot."""

from __future__ import annotations

from typing import Any


def build_pilot_connector(
    base_connector: Any, hidden_size: int, tasks: list[str], clients: list[str], bottleneck: int
) -> Any:
    import torch
    from torch import nn

    class ResidualAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down = nn.Linear(hidden_size, bottleneck, bias=False)
            self.up = nn.Linear(bottleneck, hidden_size, bias=False)

        def forward(self, value: Any) -> Any:
            return self.up(torch.nn.functional.gelu(self.down(value)))

    class PilotConnector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_connector = base_connector
            self.task_adapters = nn.ModuleDict({task: ResidualAdapter() for task in tasks})
            self.client_adapters = nn.ModuleDict(
                {client.replace("/", "__"): ResidualAdapter() for client in clients}
            )
            self.cross_task_adapters = nn.ModuleDict({task: ResidualAdapter() for task in tasks})
            self.router = nn.Linear(hidden_size, len(tasks), bias=False)
            self.current_task = tasks[0]
            self.current_client = clients[0]
            self.auxiliary_losses: dict[str, Any] = {}

        def set_context(self, task: str, client: str) -> None:
            self.current_task, self.current_client = task, client

        def forward(self, image_features: Any) -> Any:
            value = self.base_connector(image_features)
            task_value = self.task_adapters[self.current_task](value)
            client_value = self.client_adapters[self.current_client.replace("/", "__")](value)
            logits = self.router(value.mean(dim=-2))
            weights = torch.softmax(logits, dim=-1)
            experts = torch.stack([self.cross_task_adapters[task](value) for task in tasks], dim=-2)
            mixture = (experts * weights.unsqueeze(-2).unsqueeze(-1)).sum(dim=-2)
            self.auxiliary_losses = {
                "difference": (task_value @ client_value.transpose(-1, -2)).pow(2).mean(),
                "router_z": torch.logsumexp(logits, dim=-1).pow(2).mean(),
                "load_balance": (weights.mean(dim=0) - 1.0 / len(tasks)).pow(2).mean(),
            }
            return value + task_value + client_value + mixture

    return PilotConnector()
