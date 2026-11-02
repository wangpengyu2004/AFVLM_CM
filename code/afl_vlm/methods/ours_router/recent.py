"""Successful per-client selection windows and irreversible planned exits.

Only small index lists live here. No model tensors, in-flight leases, global
arrival-based expiry, frequency voting, or component sensitivity are added.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any


class RecentSelectionRegistry:
    schema_version = 2
    outcomes = {"accepted", "rejected", "cancelled", "failed", "terminated"}

    def __init__(self, ranks: Mapping[str, int], clients: Mapping[str, str], size: int = 3):
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("selection_window_size must be a positive integer")
        self.ranks, self.clients, self.size = dict(ranks), dict(clients), size
        if not self.ranks or any(type(r) is not int or r < 1 for r in self.ranks.values()):
            raise ValueError("Invalid selection rank map")
        self.planned: dict[tuple[str, int], str] = {}
        self.resolved: dict[tuple[str, int], dict] = {}
        self.recent: dict[str, list[dict]] = {client: [] for client in self.clients}
        self.exits: dict[str, dict] = {}
        self.finished_tasks: dict[str, dict] = {}

    def configure(self, events: list[Any]) -> None:
        plan = {}
        for event in events:
            key = (event.client_id, event.local_round)
            if (
                event.client_id not in self.clients
                or self.clients[event.client_id] != event.task
                or type(event.local_round) is not int
                or event.local_round < 0
                or key in plan
            ):
                raise ValueError("Invalid/duplicate selection Plan identity")
            plan[key] = event.task
        if self.planned and self.planned != plan:
            raise ValueError("Cannot change the selection Plan during a run")
        self.planned = plan

    def validate_selection(self, selected: Mapping) -> None:
        if set(selected) != set(self.ranks):
            raise ValueError("Selection must record every global module")
        for module, rank in self.ranks.items():
            indices = selected[module]
            if (
                not isinstance(indices, list)
                or any(type(j) is not int or not 0 <= j < rank for j in indices)
                or len(indices) != len(set(indices))
            ):
                raise ValueError("Selection indices must be unique integers in the global rank")

    def client_union(self, client: str) -> dict[str, list[int]]:
        return {
            m: sorted({j for row in self.recent[client] for j in row["selected_indices"][m]})
            for m in self.ranks
        }

    def task_unions(self) -> tuple[dict, dict]:
        """A task stays active until ALL its clients finish; only then freeze it."""
        tasks = sorted(set(self.clients.values()))
        active = {q: {m: set() for m in self.ranks} for q in tasks}
        frozen = copy.deepcopy(active)
        for client, task in self.clients.items():
            if task in self.finished_tasks:
                continue
            values = self.client_union(client)
            for module in self.ranks:
                active[task][module].update(values[module])
        for task, row in self.finished_tasks.items():
            for module in self.ranks:
                frozen[task][module].update(row["selected_indices"][module])
        return active, frozen

    def resolve(
        self,
        client: str,
        local_round: int,
        outcome: str,
        *,
        accepted_count: int,
        virtual_time: float | None,
        job_id: str | None = None,
        selected: Mapping | None = None,
    ) -> dict:
        key = (client, local_round)
        if (
            type(local_round) is not int
            or local_round < 0
            or key not in self.planned
            or outcome not in self.outcomes
        ):
            raise ValueError("Only known planned jobs can be explicitly resolved")
        if key in self.resolved:
            if self.resolved[key]["outcome"] != outcome:
                raise ValueError("A terminal job cannot change outcome")
            return {
                "already_resolved": True,
                "client_finished": False,
                "client_id": client,
                "local_round": local_round,
                "outcome": outcome,
            }
        if client in self.exits:
            raise ValueError("An exited client cannot be reactivated")
        if outcome == "accepted":
            if not isinstance(job_id, str) or not job_id:
                raise ValueError("Accepted selections require a real job ID")
            self.validate_selection(selected or {})
            if any(row.get("job_id") == job_id for row in self.resolved.values()):
                raise ValueError("Duplicate successful job ID")
            record = {
                "job_id": job_id,
                "local_round": local_round,
                "selected_indices": copy.deepcopy(dict(selected)),
            }
            self.recent[client] = sorted(
                self.recent[client] + [record], key=lambda row: row["local_round"]
            )[-self.size :]
        elif selected is not None:
            raise ValueError("Unsuccessful jobs cannot supply successful selections")
        self.resolved[key] = {"outcome": outcome, "job_id": job_id}
        done = all(k in self.resolved for k in self.planned if k[0] == client)
        report = {
            "already_resolved": False,
            "client_id": client,
            "local_round": local_round,
            "outcome": outcome,
            "client_finished": done,
            "recent_jobs": copy.deepcopy(self.recent[client]),
            "recent_union": self.client_union(client),
        }
        if done:
            self.exits[client] = {
                "task": self.clients[client],
                "selected_indices": self.client_union(client),
                "source_jobs": copy.deepcopy(self.recent[client]),
                "accepted_updates_at_exit": accepted_count,
                "virtual_time": virtual_time,
            }
            report["frozen_exit"] = copy.deepcopy(self.exits[client])
        task = self.clients[client]
        task_clients = sorted(c for c, q in self.clients.items() if q == task)
        report["task_finished"] = False
        if task not in self.finished_tasks and all(c in self.exits for c in task_clients):
            self.finished_tasks[task] = {
                "selected_indices": {
                    m: sorted({j for c in task_clients for j in self.client_union(c)[m]})
                    for m in self.ranks
                },
                "source_clients": task_clients,
                "accepted_updates_at_finish": accepted_count,
                "virtual_time": virtual_time,
            }
            report["task_finished"] = True
            report["frozen_task"] = copy.deepcopy(self.finished_tasks[task])
        return report

    def task_finished(self, task: str) -> bool:
        return task in self.finished_tasks

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "schema_version": self.schema_version,
                "rank_map": self.ranks,
                "client_task": self.clients,
                "window_size": self.size,
                "planned": self.planned,
                "resolved": self.resolved,
                "recent_jobs": self.recent,
                "exits": self.exits,
                "finished_tasks": self.finished_tasks,
            }
        )

    def load_state_dict(self, state: Mapping) -> None:
        if (
            state.get("schema_version") != self.schema_version
            or state.get("rank_map") != self.ranks
            or state.get("client_task") != self.clients
            or state.get("window_size") != self.size
            or set(state.get("recent_jobs", {})) != set(self.clients)
        ):
            raise ValueError(
                "Incompatible recent-selection state; restart instead of silently migrating"
            )
        trial = RecentSelectionRegistry(self.ranks, self.clients, self.size)
        trial.planned = copy.deepcopy(state["planned"])
        for (client, number), task in trial.planned.items():
            if self.clients.get(client) != task or type(number) is not int or number < 0:
                raise ValueError("Invalid restored Plan identity")
        trial.resolved = copy.deepcopy(state["resolved"])
        if not set(trial.resolved) <= set(trial.planned):
            raise ValueError("Restored resolutions include an unknown job")
        if any(row.get("outcome") not in self.outcomes for row in trial.resolved.values()):
            raise ValueError("Invalid restored terminal outcome")
        trial.recent = copy.deepcopy(state["recent_jobs"])
        seen_ids = set()
        for client, rows in trial.recent.items():
            numbers = [row["local_round"] for row in rows]
            if len(rows) > self.size or numbers != sorted(set(numbers)):
                raise ValueError("Invalid restored per-client window")
            for row in rows:
                trial.validate_selection(row["selected_indices"])
                resolved = trial.resolved.get((client, row["local_round"]), {})
                if (
                    resolved.get("outcome") != "accepted"
                    or resolved.get("job_id") != row["job_id"]
                    or row["job_id"] in seen_ids
                ):
                    raise ValueError("Restored selection has no unique accepted job")
                seen_ids.add(row["job_id"])
        trial.exits = copy.deepcopy(state["exits"])
        for client, row in trial.exits.items():
            if client not in self.clients or row.get("task") != self.clients[client]:
                raise ValueError("Invalid restored exited client")
            trial.validate_selection(row["selected_indices"])
            if (
                row["selected_indices"] != trial.client_union(client)
                or row["source_jobs"] != trial.recent[client]
                or not all(k in trial.resolved for k in trial.planned if k[0] == client)
            ):
                raise ValueError("Restored exit must freeze the final successful window")
        for client in self.clients:
            keys = [k for k in trial.planned if k[0] == client]
            if keys and all(k in trial.resolved for k in keys) and client not in trial.exits:
                raise ValueError("Fully resolved restored client is missing its frozen exit")
        trial.finished_tasks = copy.deepcopy(state.get("finished_tasks", {}))
        expected_finished = {
            q
            for q in self.clients.values()
            if all(c in trial.exits for c, task in self.clients.items() if task == q)
        }
        if set(trial.finished_tasks) != expected_finished:
            raise ValueError("Restored task completion does not match client terminal states")
        for task, row in trial.finished_tasks.items():
            clients = sorted(c for c, q in self.clients.items() if q == task)
            trial.validate_selection(row["selected_indices"])
            expected = {
                m: sorted({j for c in clients for j in trial.client_union(c)[m]})
                for m in self.ranks
            }
            if row["selected_indices"] != expected or row["source_clients"] != clients:
                raise ValueError("Restored task mask must freeze the complete final task union")
        self.planned, self.resolved, self.recent, self.exits, self.finished_tasks = (
            trial.planned,
            trial.resolved,
            trial.recent,
            trial.exits,
            trial.finished_tasks,
        )
