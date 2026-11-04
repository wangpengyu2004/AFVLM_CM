"""One eta*rho step now; exact applied-direction feedback for FUTURE selection.

No task-frequency multiplier, historical sensitivity denominator, age term,
overlap switch or repeated penalty in the fusion coefficient.
"""

from __future__ import annotations

import copy
import math
from typing import Any

from afl_vlm.federation.types import ServerMutation
from afl_vlm.methods.ours_hc import payload
from afl_vlm.models.base import clone_state


def full_functional_drift(base: dict, current: dict, modules: dict) -> float:
    """Exact full-support BA distance; independent of upload, sensitivity and mask."""
    return float(sum(
        payload.lora_functional_distance_sq(
            current[item["a_name"]], current[item["b_name"]],
            base[item["a_name"]], base[item["b_name"]], float(item["scaling"]),
        ) for item in modules.values()
    ))


def arrive(method: Any, update: Any, context: Any) -> list[ServerMutation]:
    payload.validate(update, method.modules)
    key = (update.client_id, update.local_round)
    if (key not in method.downloads or key not in method.choices
            or update.update_id in method.seen or key in method.resolved):
        raise ValueError("Duplicate/missing conflict-router virtual download")
    version, base = method.downloads[key]
    if (version != update.base_version or version > context.version
            or method.client_tasks.get(update.client_id) != update.task
            or (method.planned and method.planned.get(key) != update.task)):
        raise ValueError("Conflict-router upload has an invalid task/version/Plan identity")
    if method.choices[key]["indices"] != update.metadata["hc_selection"]:
        raise ValueError("Upload support differs from its virtual-download choice")
    modules = copy.deepcopy(method.modules)
    for name in modules:
        if modules[name]["scaling"] is None:
            modules[name]["scaling"] = float(update.metadata["lora_metadata"][name]["scaling"])
    window = method.conflict_window
    if window.modules != modules:
        if window.clock:
            raise ValueError("Conflict functional scaling changed during the run")
        window = window._new()
        window.modules = modules
    window.to(next(iter(context.global_state.values())).device)
    selected_drift, energy = payload.functional_energies(base, context.global_state, update)
    mode = method.params.get("staleness_mode", "selected_relative")
    drift = (full_functional_drift(base, context.global_state, modules)
             if mode == "full_functional_absolute" else selected_drift)
    if not all(math.isfinite(x) and x >= 0 for x in (drift, energy)):
        raise ValueError("Invalid conflict-router functional energy")
    distance = (math.sqrt(drift) if mode == "full_functional_absolute"
                else math.sqrt(drift / (energy + method.value("eps", 1e-12))))
    if not math.isfinite(distance):
        raise ValueError("Non-finite functional staleness")
    rho = math.exp(-method.value("gamma", 1) * distance)
    alpha = method.value("server_step_size", 1) * rho
    accepted = energy >= method.value("min_update_energy", 1e-16) and alpha > 0
    alphas = dict.fromkeys(modules, alpha if accepted else 0.0)
    candidate = (payload.fuse(context.global_state, update, alphas) if accepted
                 else clone_state(context.global_state))
    if not payload.finite_state(candidate):
        raise ValueError("Non-finite conflict-router fusion")
    pending = (window.append(context.global_state, candidate, update.metadata["hc_selection"],
                             update.task, update.update_id, method.accepted_count + 1)
               if accepted else window)
    selection = copy.deepcopy(method.choices[key].get("conflict_selection", {}))
    diagnostic = {
        "method": method.name, "aggregation_mode": "conflict_router",
        "update_id": update.update_id, "client_id": update.client_id, "task": update.task,
        "local_round": update.local_round, "base_version": version,
        "server_version_before": context.version, "arrival_time": update.arrival_time,
        "optimizer_steps": update.optimizer_steps, "accepted": accepted,
        "version_staleness": context.version - version,
        "functional_drift": drift, "functional_local_energy": energy,
        "relative_staleness": distance if mode == "selected_relative" else None,
        "absolute_staleness": distance if mode == "full_functional_absolute" else None,
        "staleness_mode": mode, "selected_functional_drift": selected_drift,
        "reliability": rho,
        "server_step_size": method.value("server_step_size", 1),
        "uniform_alpha": alpha if accepted else 0.0, "module_alpha": alphas,
        "accepted_updates_before": method.accepted_count,
        "accepted_updates_after": method.accepted_count + int(accepted),
        "selected_indices": update.metadata["hc_selection"],
        "selected_rank": {m: len(v) for m, v in update.metadata["hc_selection"].items()},
        "module_sensitivity": update.metadata["module_sensitivity"],
        "sensitivity_transform": update.metadata["sensitivity_transform"],
        "sensitivity_valid": update.metadata["sensitivity_valid"],
        "sensitivity_observations": update.metadata["sensitivity_observations"],
        "conflict_selection": selection,
        "conflict_window_before": window.summary(), "conflict_window_after": pending.summary(),
        "summary_scope": "one_coefficient_for_the_entire_upload",
        "local": update.metadata["hc_local"], "communication": update.metadata["hc_communication"],
    }

    def commit() -> None:
        # Candidate state/model copy is already successful at this point. A
        # rejected upload resolves transport only; it supplies no new evidence.
        method.modules = modules
        method.conflict_window = pending
        method.seen.add(update.update_id)
        method.downloads.pop(key)
        method.choices.pop(key)
        if accepted:
            method.resolved.add(key)
            method.accepted_count += 1

    return [ServerMutation(candidate, alpha if accepted else 0.0, [update.update_id],
                           increment_version=accepted,
                           metadata={"router_diagnostics": diagnostic, **(
                               {} if accepted else {"rejected": "no_effective_update"}
                           )}, on_commit=commit)]
