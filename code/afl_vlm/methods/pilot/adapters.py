"""Pilot (AAAI 2025): pretrained connectors and token-wise CT-MoA.

Only Pilot installs this wrapper. Round 0 is stage 1; later rounds are stage 2,
a budget-matched adaptation instead of adding extra first-round local epochs.
Private connectors/router never enter shared aggregation.
"""

from __future__ import annotations

import copy
from typing import Any


def build_pilot_connector(
    base_connector: Any, tasks: list[str], normalize_difference: bool = True
) -> Any:
    import torch
    from torch import nn

    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Pilot requires unique task descriptors")
    first_linear = next(layer for layer in base_connector.modules() if isinstance(layer, nn.Linear))

    class PilotConnector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base_connector = base_connector
            self.base_connector.requires_grad_(False)
            self.task_adapters = nn.ModuleDict(
                {task: copy.deepcopy(base_connector) for task in tasks}
            )
            self.private_adapter = copy.deepcopy(base_connector)
            self.cross_task_adapters = nn.ModuleDict(
                {task: copy.deepcopy(base_connector) for task in tasks}
            )
            self.router = nn.Linear(first_linear.in_features, len(tasks), bias=False).to(
                device=first_linear.weight.device, dtype=first_linear.weight.dtype
            )
            self.current_task, self.current_client = tasks[0], None
            self.stage = 1
            self.auxiliary_losses: dict[str, Any] = {}

        def set_context(self, task: str, client: str | None) -> None:
            if task not in self.task_adapters:
                raise ValueError(f"Unknown Pilot task: {task}")
            self.current_task, self.current_client = task, client

        def configure_local(
            self, task: str, client: str, stage: int, private: dict[str, Any] | None
        ) -> None:
            self.set_context(task, client)
            self.requires_grad_(False)
            self.stage = stage
            if private:
                self.load_private(private)
            else:
                self.private_adapter.load_state_dict(self.base_connector.state_dict())
            if stage == 2 and (not private or int(private["stage"]) == 1):
                for adapter in self.cross_task_adapters.values():
                    adapter.load_state_dict(self.task_adapters[task].state_dict())
                with torch.no_grad():
                    self.router.weight.zero_()
            self.stage = stage
            self.task_adapters[task].requires_grad_(True)
            if stage == 1:
                self.private_adapter.requires_grad_(True)
            else:
                self.router.requires_grad_(True)
                for other in tasks:
                    if other != task:
                        self.cross_task_adapters[other].requires_grad_(True)

        def export_private(self) -> dict[str, Any]:
            def snapshot(module: Any) -> dict[str, Any]:
                return {
                    key: value.detach().float().clone()
                    for key, value in module.state_dict().items()
                }

            return {
                "stage": self.stage,
                "adapter": snapshot(self.private_adapter),
                "cross": {
                    task: snapshot(module) for task, module in self.cross_task_adapters.items()
                },
                "router": snapshot(self.router),
            }

        def load_private(self, state: dict[str, Any]) -> None:
            self.private_adapter.load_state_dict(state["adapter"])
            for task, values in state["cross"].items():
                self.cross_task_adapters[task].load_state_dict(values)
            self.router.load_state_dict(state["router"])
            self.stage = int(state["stage"])

        def forward(self, image_features: Any) -> Any:
            task_value = self.task_adapters[self.current_task](image_features)
            self.auxiliary_losses = {}
            # Explicit shared deployment view for global primary evaluation.
            if self.current_client is None:
                return task_value
            private_value = self.private_adapter(image_features)
            if self.stage == 1:
                # Eq. (3): feature-space, not token-space, orthogonality.
                left, right = task_value.float(), private_value.float()
                if normalize_difference:
                    left = torch.nn.functional.normalize(left, dim=-1)
                    right = torch.nn.functional.normalize(right, dim=-1)
                # Average independent images, not cross-image feature products.
                self.auxiliary_losses["difference"] = (
                    (left.transpose(-1, -2) @ right).square().sum(dim=(-2, -1)).mean()
                )
                return task_value + private_value
            logits = self.router(image_features).float()
            weights = logits.softmax(dim=-1)
            expert_values = [
                task_value
                if task == self.current_task
                else self.task_adapters[task](image_features)
                + self.cross_task_adapters[task](image_features)
                for task in tasks
            ]
            experts = torch.stack(expert_values, dim=-2)
            routed = (experts * weights.to(experts.dtype).unsqueeze(-1)).sum(dim=-2)
            probabilities = weights.reshape(-1, len(tasks))
            assignments = (
                torch.nn.functional.one_hot(probabilities.argmax(dim=-1), len(tasks))
                .float()
                .mean(dim=0)
            )
            self.auxiliary_losses = {
                "load_balance": len(tasks) * (assignments * probabilities.mean(dim=0)).sum(),
                "router_z": logits.logsumexp(dim=-1).square().mean(),
            }
            return routed + private_value

    return PilotConnector()
