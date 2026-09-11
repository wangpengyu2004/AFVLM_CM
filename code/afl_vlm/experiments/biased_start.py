"""E2: controlled biased versus balanced download-start diagnostics."""

from __future__ import annotations

from typing import Any

from afl_vlm.data.partition import local_batch
from afl_vlm.evaluation.paired_metrics import adaptation_gap
from afl_vlm.evaluation.probes import evaluate_tasks
from afl_vlm.evaluation.task_metrics import task_damage
from afl_vlm.experiments.common import batch_for, probe_all, train_config_for, update_record
from afl_vlm.federation.client import FederatedClient
from afl_vlm.federation.server import FederatedServer
from afl_vlm.models.base import add_scaled, state_hash, state_norm


def _build_checkpoint(context: Any, path: list[str]) -> dict[str, Any]:
    server = FederatedServer(context.model, context.method, len(context.clients))
    occurrences = {task: 0 for task in context.tasks}
    for index, task_key in enumerate(path):
        candidates = sorted(
            (client for client in context.clients.values() if client.task == task_key),
            key=lambda client: client.id,
        )
        client = candidates[occurrences[task_key] % len(candidates)]
        local_round = occurrences[task_key] // len(candidates)
        occurrences[task_key] += 1
        update = client.train(
            context.model,
            context.method,
            server.state,
            server.version,
            local_round,
            batch_for(context, client, local_round),
            train_config_for(context, context.derived_seed("e2-path", index, task_key, client.id)),
        )
        state_before = server.state
        server.receive(update)
        context.writer.append("updates.jsonl", update_record(update, state_before))
    return server.state


def _diagnose(context: Any, checkpoint: dict[str, Any], label: str) -> dict[str, Any]:
    context.model.load_trainable(checkpoint)
    before = probe_all(context)
    samples = context.heldout_samples["slow"]
    heldout = FederatedClient("heldout-s", "slow", samples)
    fixed_batch = local_batch(
        samples,
        0,
        context.train_config.local_steps,
        context.train_config.grad_accumulation,
    )
    update = heldout.train(
        context.model,
        context.method,
        checkpoint,
        4,
        0,
        fixed_batch,
        train_config_for(context, context.derived_seed("e2-heldout")),
        update_kind=label,
        pair_id="E2",
    )
    local_state = add_scaled(checkpoint, update.delta, 1.0)
    context.model.load_trainable(local_state)
    local_after = probe_all(context)
    corrected_state = add_scaled(
        checkpoint, update.delta, float(context.method.params["server_lr"])
    )
    context.model.load_trainable(corrected_state)
    corrected_after = probe_all(context)
    final_before = None
    final_local_after = None
    final_corrected_after = None
    if context.config["evaluation"]["generation_on_selected_checkpoints"]:
        context.model.load_trainable(checkpoint)
        final_before = evaluate_tasks(context.model, context.tasks, context.final_ids, mode="final")
        context.model.load_trainable(local_state)
        final_local_after = evaluate_tasks(
            context.model, context.tasks, context.final_ids, mode="final"
        )
        context.model.load_trainable(corrected_state)
        final_corrected_after = evaluate_tasks(
            context.model, context.tasks, context.final_ids, mode="final"
        )
    context.writer.append("updates.jsonl", update_record(update, checkpoint))
    area = sum(update.losses) / len(update.losses)
    return {
        "label": label,
        "checkpoint_hash": state_hash(checkpoint),
        "before": before,
        "local_after": local_after,
        "corrected_after": corrected_after,
        "final_before": final_before,
        "final_local_after": final_local_after,
        "final_corrected_after": final_corrected_after,
        "delta_norm": state_norm(update.delta),
        "loss_curve_area": area,
        "step5_loss": update.losses[-1],
        "fast_damage_after_correction": task_damage(before["fast"], corrected_after["fast"]),
    }


def run(context: Any, params: dict[str, Any]) -> list[dict[str, Any]]:
    if not context.method.branch_compatible:
        raise ValueError(f"Method '{context.method.name}' is incompatible with biased_start")
    biased_path = list(params.get("biased_path", ["fast"] * 4))
    balanced_path = list(params.get("balanced_path", ["fast", "slow", "fast", "slow"]))
    if len(biased_path) != 4 or len(balanced_path) != 4:
        raise ValueError("E2 controlled checkpoints must each contain four updates")
    initial_state = context.model.snapshot_trainable()
    initial_method = context.method.state_dict()
    context.model.load_trainable(initial_state)
    context.method.load_state_dict(initial_method)
    bias = _build_checkpoint(context, biased_path)
    context.model.load_trainable(initial_state)
    context.method.load_state_dict(initial_method)
    balanced = _build_checkpoint(context, balanced_path)
    bias_result = _diagnose(context, bias, "bias")
    balanced_result = _diagnose(context, balanced, "balanced")
    gaps = adaptation_gap(
        bias_result["before"]["slow"],
        balanced_result["before"]["slow"],
        bias_result["local_after"]["slow"],
        balanced_result["local_after"]["slow"],
    )
    row = {
        "experiment": "e2",
        **gaps,
        "bias_delta_norm": bias_result["delta_norm"],
        "balanced_delta_norm": balanced_result["delta_norm"],
        "delta_norm_gap": bias_result["delta_norm"] - balanced_result["delta_norm"],
        "bias_step5_loss": bias_result["step5_loss"],
        "balanced_step5_loss": balanced_result["step5_loss"],
        "bias_fast_damage": bias_result["fast_damage_after_correction"],
        "balanced_fast_damage": balanced_result["fast_damage_after_correction"],
    }
    for result, path in ((bias_result, biased_path), (balanced_result, balanced_path)):
        context.writer.append(
            "branches.jsonl",
            {
                "pair_id": "E2",
                "receiving_state_id": result["checkpoint_hash"],
                "path_tasks": path,
                "tau": 0,
                "branch": result["label"],
                "task_metrics_before": result["before"],
                "task_metrics_after": result["local_after"],
                "final_metrics_before": result["final_before"],
                "final_metrics_after": result["final_local_after"],
                "delta_norm": result["delta_norm"],
                "loss_curve_area": result["loss_curve_area"],
                "fast_damage_after_correction": result["fast_damage_after_correction"],
            },
        )
    context.writer.write_json(
        "schedule.json",
        [
            {"checkpoint": "bias", "path_tasks": biased_path},
            {"checkpoint": "balanced", "path_tasks": balanced_path},
        ],
    )
    context.model.load_trainable(initial_state)
    context.method.load_state_dict(initial_method)
    return [row]
