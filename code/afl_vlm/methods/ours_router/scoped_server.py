"""Method-private transactions: overlap switches the original sensitivity rule."""

from __future__ import annotations

import copy
import math
from typing import Any

from afl_vlm.federation.types import ServerMutation
from afl_vlm.methods.ours_hc import payload
from afl_vlm.methods.ours_layer_protection import select_layers, sum_extra
from afl_vlm.methods.ours_router import scoped
from afl_vlm.models.base import clone_state


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def finish_tasks(
    method: Any, registry: Any, history: dict, counts: dict, finished: dict, accepted_count: int
) -> dict:
    newly_finished = {}
    for task in method.task_weights:
        if task not in finished and registry.task_finished(task):
            snapshot = (
                select_layers(
                    history[task],
                    1 - method.value("sensitivity_uniform_mix", 0.75),
                    method.value("eps", 1e-12),
                )
                if method.params.get("history_age_enabled", True)
                else {
                    "history": copy.deepcopy(history[task]),
                    "selected_layers": [],
                    "layer_mask": {},
                }
            )
            snapshot.update(finished_at=accepted_count, observed_updates=counts[task])
            snapshot["selected_indices"] = copy.deepcopy(
                registry.finished_tasks[task]["selected_indices"]
            )
            finished[task] = snapshot
            newly_finished[task] = copy.deepcopy(snapshot)
    return newly_finished


def resolve_terminal(
    method: Any, client: str, local_round: int, outcome: str, virtual_time: float | None
) -> dict:
    if outcome == "accepted":
        raise ValueError("Accepted jobs must be resolved by a committed server mutation")
    registry = copy.deepcopy(method.selection_registry)
    report = registry.resolve(
        client,
        local_round,
        outcome,
        accepted_count=method.accepted_count,
        virtual_time=virtual_time,
    )
    if report["already_resolved"]:
        return report
    finished = copy.deepcopy(method.finished)
    report["finished_tasks"] = finish_tasks(
        method, registry, method.history, method.history_counts, finished, method.accepted_count
    )
    method.selection_registry, method.finished = registry, finished
    method.resolved = set(registry.resolved)
    method.downloads.pop((client, local_round), None)
    method.choices.pop((client, local_round), None)
    return report


def arrive(method: Any, update: Any, context: Any) -> list[ServerMutation]:
    scoped.validate(update, method.modules)
    key = (update.client_id, update.local_round)
    registry = method.selection_registry
    if (
        type(update.local_round) is not int
        or key not in method.downloads
        or update.update_id in method.seen
        or key in registry.resolved
        or registry.planned.get(key) != update.task
    ):
        raise ValueError("Duplicate, resolved, or unknown scoped router job")
    version, base = method.downloads[key]
    if (
        version != update.base_version
        or version > context.version
        or method.client_tasks.get(update.client_id) != update.task
    ):
        raise ValueError("Scoped router version/task mismatch")
    drift, energy = payload.functional_energies(base, context.global_state, update)
    if not all(math.isfinite(v) and v >= 0 for v in (drift, energy)):
        raise ValueError("Non-finite scoped functional energy")
    eps, lam = method.value("eps", 1e-12), method.value("history_lambda", 1)
    relative = math.sqrt(drift / (energy + eps))
    if not math.isfinite(relative):
        raise ValueError("Non-finite scoped staleness")
    rho = math.exp(-method.value("gamma", 1) * relative)
    frequency = (
        (len(method.window) / len(method.task_weights) + 1) / (method.window.count(update.task) + 1)
    ) ** method.value("frequency_power", 0.5)
    frequency = min(
        method.value("frequency_max", 1.5), max(method.value("frequency_min", 0.5), frequency)
    )
    boosts = {
        q: method.value("history_age_boost", 3)
        * -math.expm1(
            -(method.accepted_count - row["finished_at"])
            / method.value("history_age_time_scale", 18)
        )
        for q, row in method.finished.items()
    }
    _, demands = (
        sum_extra(method.finished, method.task_weights, boosts, tuple(method.modules))
        if method.finished and method.params.get("history_age_enabled", True)
        else ({}, {})
    )
    selected, c = update.metadata["hc_selection"], update.metadata["module_sensitivity"]
    details = scoped.precision(
        registry,
        update.task,
        selected,
        method.history,
        method.task_weights,
        method.finished,
        demands,
        method.value("history_strength", 1),
    )
    alphas = {}
    for name, row in details.items():
        row["effective_frequency"] = [
            frequency if matched else 1.0 for matched in row["sensitivity_mask"]
        ]
        alphas[name] = scoped.rank_alpha(
            float(c[name]),
            rho,
            row["effective_frequency"],
            row["omega"],
            sensitivity_mask=row["sensitivity_mask"],
            eta=method.value("server_step_size", 1),
            lam=lam,
            eps=eps,
            maximum=float(method.params["alpha_max"])
            if method.params.get("alpha_max") is not None
            else None,
        )
        row["alpha_proposed"] = list(alphas[name])
    accepted = energy >= method.value("min_update_energy", 1e-16) and any(
        any(values) for values in alphas.values()
    )
    state = (
        scoped.fuse(context.global_state, update, alphas)
        if accepted
        else clone_state(context.global_state)
    )
    if not payload.finite_state(state):
        raise ValueError("Non-finite scoped fusion; selection/history not committed")

    # Only small algorithm state is staged. Never deepcopy in-flight LoRA snapshots.
    next_registry = copy.deepcopy(registry)
    history, counts = copy.deepcopy(method.history), dict(method.history_counts)
    window, finished = list(method.window), copy.deepcopy(method.finished)
    accepted_count = method.accepted_count + int(accepted)
    if accepted:
        for name in method.modules:
            history[update.task][name] = (
                float(c[name])
                if not counts[update.task]
                else (
                    method.value("history_beta", 0.9) * history[update.task][name]
                    + (1 - method.value("history_beta", 0.9)) * float(c[name])
                )
            )
        counts[update.task] += 1
        window = (window + [update.task])[-int(method.params.get("frequency_window_size", 18)) :]
    terminal = next_registry.resolve(
        update.client_id,
        update.local_round,
        "accepted" if accepted else "rejected",
        accepted_count=accepted_count,
        virtual_time=update.arrival_time,
        job_id=update.update_id,
        selected=selected if accepted else None,
    )
    terminal["finished_tasks"] = finish_tasks(
        method, next_registry, history, counts, finished, accepted_count
    )
    for name, row in details.items():
        row["alpha"] = list(alphas[name]) if accepted else [0.0] * len(alphas[name])
    diagnostic = {
        "method": method.name,
        "rank_aggregation_mode": "overlap_switch",
        "update_id": update.update_id,
        "client_id": update.client_id,
        "task": update.task,
        "local_round": update.local_round,
        "base_version": version,
        "server_version_before": context.version,
        "arrival_time": update.arrival_time,
        "optimizer_steps": update.optimizer_steps,
        "accepted": accepted,
        "version_staleness": context.version - version,
        "functional_drift": drift,
        "functional_local_energy": energy,
        "functional_energy_normalization": "unchanged_original_low_rank_distance",
        "relative_staleness": relative,
        "reliability": rho,
        "frequency_weight": frequency,
        "frequency_counts_before": {q: method.window.count(q) for q in method.task_weights},
        "frequency_counts_after": {q: window.count(q) for q in method.task_weights},
        "history_accepted_updates_before": method.accepted_count,
        "history_accepted_updates_after": accepted_count,
        "history_age_boosts_before": boosts,
        "selected_indices": copy.deepcopy(selected),
        "selected_rank": {m: len(v) for m, v in selected.items()},
        "module_sensitivity": dict(c),
        "sensitivity_transform": update.metadata["sensitivity_transform"],
        "sensitivity_valid": update.metadata["sensitivity_valid"],
        "sensitivity_observations": update.metadata["sensitivity_observations"],
        "history_before": copy.deepcopy(method.history),
        "history_after": history,
        # Legacy scalar columns remain explicitly selected-position means.
        "omega_base": {m: mean(row["omega_base"]) for m, row in details.items()},
        "omega_extra": {m: mean(row["omega_age"]) for m, row in details.items()},
        "omega": {m: mean(row["omega"]) for m, row in details.items()},
        "module_alpha": {m: mean(row["alpha"]) for m, row in details.items()},
        "summary_scope": "mean_over_uploaded_positions",
        "ended_task_demands": demands,
        "finished_task_snapshot": terminal["finished_tasks"].get(update.task),
        "local": update.metadata["hc_local"],
        "communication": update.metadata["hc_communication"],
        "scoped_aggregation": {
            "modules": details,
            "selection_window_size": registry.size,
            "server_step_size": method.value("server_step_size", 1),
            "alpha_max": method.params.get("alpha_max"),
        },
        "job_resolution": terminal,
    }

    def commit() -> None:
        method.history, method.history_counts, method.window = history, counts, window
        method.accepted_count, method.finished = accepted_count, finished
        method.selection_registry = next_registry
        method.resolved = set(next_registry.resolved)
        method.seen.add(update.update_id)
        method.downloads.pop(key, None)
        method.choices.pop(key, None)
        for name in method.modules:
            if method.modules[name]["scaling"] is None:
                method.modules[name]["scaling"] = update.metadata["lora_metadata"][name]["scaling"]

    weight = mean([value for row in details.values() for value in row["alpha"]])
    return [
        ServerMutation(
            state,
            weight if accepted else 0.0,
            [update.update_id],
            increment_version=accepted,
            metadata={
                "router_diagnostics": diagnostic,
                **({} if accepted else {"rejected": "no_effective_update"}),
            },
            on_commit=commit,
        )
    ]
