"""E1: same-receiving-state stale/fresh twin diagnostics."""

from __future__ import annotations

from typing import Any

from afl_vlm.data.partition import local_batch
from afl_vlm.evaluation.branch import BranchEvaluator
from afl_vlm.evaluation.paired_metrics import extra_harm
from afl_vlm.evaluation.probes import evaluate_tasks
from afl_vlm.evaluation.task_metrics import task_damage
from afl_vlm.experiments.common import batch_for, probe_all, train_config_for, update_record
from afl_vlm.federation.client import FederatedClient
from afl_vlm.federation.server import FederatedServer
from afl_vlm.models.base import state_hash


def run(context: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    if not context.method.branch_compatible:
        raise ValueError(f"Method '{context.method.name}' is incompatible with stale_twin")
    tau = int(params.get("tau", 2))
    paths = [list(path) for path in params.get("paths", [["fast", "fast"], ["slow", "slow"]])]
    if any(len(path) != tau for path in paths):
        raise ValueError("Every E1 path length must equal tau")
    initial_state = context.model.snapshot_trainable()
    initial_method = context.method.state_dict()
    slow_samples = context.heldout_samples["slow"]
    diagnostic = FederatedClient("heldout-s", "slow", slow_samples)
    fixed_batch = local_batch(
        slow_samples,
        0,
        context.train_config.local_steps,
        context.train_config.grad_accumulation,
    )
    rows: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths, start=1):
        context.model.load_trainable(initial_state)
        context.method.load_state_dict(initial_method)
        server = FederatedServer(context.model, context.method, len(context.clients))
        pair_id = f"P{path_index}"
        twin_seed = context.derived_seed("e1-twin", pair_id)
        stale = diagnostic.train(
            context.model,
            context.method,
            initial_state,
            0,
            0,
            fixed_batch,
            train_config_for(context, twin_seed),
            update_kind="stale",
            pair_id=pair_id,
        )
        task_occurrences: dict[str, int] = {"fast": 0, "slow": 0}
        for path_step, task_key in enumerate(path):
            candidates = sorted(
                (client for client in context.clients.values() if client.task == task_key),
                key=lambda client: client.id,
            )
            client = candidates[task_occurrences[task_key] % len(candidates)]
            local_round = task_occurrences[task_key] // len(candidates)
            task_occurrences[task_key] += 1
            seed = context.derived_seed("e1-path", pair_id, path_step, client.id)
            update = client.train(
                context.model,
                context.method,
                server.state,
                server.version,
                local_round,
                batch_for(context, client, local_round),
                train_config_for(context, seed),
            )
            state_before = server.state
            server.receive(update)
            context.writer.append("updates.jsonl", update_record(update, state_before))
        receiving_state = server.state
        receiving_hash = state_hash(receiving_state)
        before = probe_all(context)
        fresh = diagnostic.train(
            context.model,
            context.method,
            receiving_state,
            server.version,
            0,
            fixed_batch,
            train_config_for(context, twin_seed),
            update_kind="fresh",
            pair_id=pair_id,
        )
        context.writer.append("updates.jsonl", update_record(stale, receiving_state))
        context.writer.append("updates.jsonl", update_record(fresh, receiving_state))
        evaluator = BranchEvaluator(
            context.model, context.method, float(context.method.params["server_lr"])
        )
        branch_metrics, restored_hash = evaluator.evaluate_branches(
            receiving_state,
            {"stale": stale.delta, "fresh": fresh.delta},
            context.tasks,
            context.probe_ids,
        )
        if restored_hash != receiving_hash:
            raise RuntimeError("E1 branch evaluation did not restore the receiving state")
        affected_task = path[-1]
        paired = extra_harm(
            before[affected_task],
            branch_metrics["stale"][affected_task],
            branch_metrics["fresh"][affected_task],
        )
        row = {
            "experiment": "e1",
            "pair_id": pair_id,
            "path": ",".join(path),
            "affected_task": affected_task,
            "tau": tau,
            "stale_s_gain": -task_damage(before["slow"], branch_metrics["stale"]["slow"]),
            "fresh_s_gain": -task_damage(before["slow"], branch_metrics["fresh"]["slow"]),
            **paired,
        }
        final_before = None
        final_branches = None
        if context.config["evaluation"]["generation_on_selected_checkpoints"]:
            context.model.load_trainable(receiving_state)
            final_before = evaluate_tasks(
                context.model, context.tasks, context.final_ids, mode="final"
            )
            final_branches, restored_hash = evaluator.evaluate_branches(
                receiving_state,
                {"stale": stale.delta, "fresh": fresh.delta},
                context.tasks,
                context.final_ids,
                mode="final",
            )
            if restored_hash != receiving_hash:
                raise RuntimeError("E1 final branch evaluation did not restore receiving state")
        rows.append(row)
        for branch in ("stale", "fresh"):
            context.writer.append(
                "branches.jsonl",
                {
                    "pair_id": pair_id,
                    "receiving_state_id": receiving_hash,
                    "path_tasks": path,
                    "tau": tau,
                    "branch": branch,
                    "task_metrics_before": before,
                    "task_metrics_after": branch_metrics[branch],
                    "final_metrics_before": final_before,
                    "final_metrics_after": final_branches[branch] if final_branches else None,
                    "extra_harm": paired["extra_harm"],
                },
            )
    context.writer.write_json(
        "schedule.json",
        [
            {"pair_id": f"P{index}", "tau": tau, "path_tasks": path}
            for index, path in enumerate(paths, start=1)
        ],
    )
    context.model.load_trainable(initial_state)
    context.method.load_state_dict(initial_method)
    return rows
