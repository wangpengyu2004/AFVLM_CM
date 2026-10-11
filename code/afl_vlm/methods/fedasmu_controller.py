"""FedASMU Eqs. 5--6: LSTM cold-start policy and per-client Q refinement.

Epoch slots become normalized optimizer slots for one-epoch VLM tuning. One
categorical request specializes the binary sequence under the one-request
constraint. Dispatch-time policy gradients handle asynchronous rewards without
keeping live autograd graphs while other policy updates mutate the controller.
"""

from __future__ import annotations

import copy
import random
from typing import Any


class RequestController:
    def __init__(self, params: dict[str, Any], device: Any) -> None:
        import torch

        self.params = dict(params)
        self.bins = int(params.get("request_bins", 8))
        hidden = int(params.get("rl_hidden_size", 16))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(params.get("controller_seed", 42)))
            self.lstm = torch.nn.LSTM(3, hidden, batch_first=True)
            self.head = torch.nn.Linear(hidden, 1)
        self.lstm.to(device)
        self.head.to(device)
        self.device = device
        self.clients: dict[str, Any] = {}
        self.baseline = 0.0
        self.initial_decisions: dict[tuple[str, int], Any] = {}

    def select(
        self, client: str, round_index: int, seed: int, total_steps: int, version: int
    ) -> dict[str, Any]:
        import torch

        rng = random.Random(seed)
        eps = float(self.params.get("rl_epsilon", 0.1))
        valid = list(range(min(self.bins, total_steps - 1)))
        if not valid:
            return {"step": None, "kind": "too_short"}
        features = [
            [(i + 1) / (len(valid) + 1), 1 / (total_steps + 1), version / (version + 1)]
            for i in valid
        ]
        row = self.clients.get(client)
        if row is None:
            x = torch.tensor([features], dtype=torch.float32, device=self.device)
            sequence, _ = self.lstm(x)
            probabilities = self.head(sequence).reshape(-1).softmax(0)
            behavior = (1 - eps) * probabilities + eps / len(valid)
            slot = rng.choices(valid, weights=behavior.detach().cpu().tolist(), k=1)[0]
            parameters = [*self.lstm.parameters(), *self.head.parameters()]
            gradients = torch.autograd.grad(behavior[slot].log(), parameters)
            self.initial_decisions[(client, round_index)] = [g.detach().clone() for g in gradients]
            row = {"slot": slot, "q": [[0.0] * 3 for _ in range(self.bins)], "updates": 0}
            self.clients[client] = row
            kind, previous_slot, action = "lstm", slot, 1
        else:
            previous_slot = min(row["slot"], len(valid) - 1)
            actions = [
                a for a, change in enumerate((-1, 0, 1)) if 0 <= previous_slot + change < len(valid)
            ]
            if rng.random() < eps:
                action = rng.choice(actions)
            else:
                best = max(row["q"][previous_slot][a] for a in actions)
                action = rng.choice([a for a in actions if row["q"][previous_slot][a] == best])
            slot = previous_slot + (-1, 0, 1)[action]
            kind = "q_learning"
        step = max(1, min(total_steps - 1, round((slot + 1) / (len(valid) + 1) * total_steps)))
        return {
            "kind": kind,
            "slot": slot,
            "previous_slot": previous_slot,
            "action": action,
            "step": step,
            "valid_slots": len(valid),
        }

    def observe(
        self, client: str, round_index: int, decision: dict[str, Any], reward: float
    ) -> None:
        import torch

        if decision["kind"] == "too_short":
            return
        row = self.clients[client]
        if decision["kind"] == "lstm":
            gradients = self.initial_decisions.pop((client, round_index))
            rate = float(self.params.get("reward_baseline_rate", 0.1))
            self.baseline = (1 - rate) * self.baseline + rate * reward
            with torch.no_grad():
                for p, g in zip(
                    [*self.lstm.parameters(), *self.head.parameters()], gradients, strict=True
                ):
                    p.add_(
                        g.to(device=p.device),
                        alpha=float(self.params.get("rl_lr", 0.001)) * (reward - self.baseline),
                    )
        else:
            previous, action, slot = decision["previous_slot"], decision["action"], decision["slot"]
            valid_actions = [
                a
                for a, change in enumerate((-1, 0, 1))
                if 0 <= slot + change < decision["valid_slots"]
            ]
            target = reward + float(self.params.get("q_discount", 0.9)) * max(
                row["q"][slot][a] for a in valid_actions
            )
            old = row["q"][previous][action]
            row["q"][previous][action] = old + float(self.params.get("q_lr", 0.1)) * (target - old)
        row["slot"] = decision["slot"]
        row["updates"] += 1

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy(
            {
                "lstm": self.lstm.state_dict(),
                "head": self.head.state_dict(),
                "clients": self.clients,
                "baseline": self.baseline,
                "initial_decisions": self.initial_decisions,
            }
        )

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.lstm.load_state_dict(state["lstm"])
        self.head.load_state_dict(state["head"])
        self.clients = copy.deepcopy(state["clients"])
        self.baseline = float(state["baseline"])
        self.initial_decisions = copy.deepcopy(state["initial_decisions"])
