"""Shared experiment helpers."""

from __future__ import annotations

from typing import Any

from afl_vlm.data.partition import local_batch
from afl_vlm.evaluation.probes import evaluate_tasks
from afl_vlm.models.base import state_cosine, state_norm, subtract


def train_config_for(context: Any, seed: int, local_steps: int | None = None) -> Any:
    config = context.train_config
    return type(config)(
        local_steps=config.local_steps if local_steps is None else local_steps,
        batch_size=config.batch_size,
        grad_accumulation=config.grad_accumulation,
        client_lr=config.client_lr,
        max_text_length=config.max_text_length,
        seed=seed,
    )


def batch_for(
    context: Any, client: Any, local_round: int, local_steps: int | None = None
) -> list[Any]:
    steps = context.train_config.local_steps if local_steps is None else local_steps
    return local_batch(
        client.samples,
        local_round,
        steps,
        context.train_config.grad_accumulation,
    )


def update_record(update: Any, receiving_state: Any) -> dict[str, Any]:
    drift = subtract(receiving_state, update.base_state)
    return {
        "pair_id": update.pair_id,
        "update_id": update.update_id,
        "client_id": update.client_id,
        "task": update.task,
        "local_round": update.local_round,
        "download_version": update.download_version,
        "update_kind": update.update_kind,
        "delta_norm": state_norm(update.delta),
        "server_drift_norm": state_norm(drift),
        "cosine_with_server_drift": state_cosine(update.delta, drift),
        "sample_ids_hash": update.sample_ids_hash,
        "seed": update.seed,
        "optimizer_steps": update.optimizer_steps,
        "losses": update.losses,
    }


def probe_all(context: Any) -> dict[str, dict[str, float]]:
    return evaluate_tasks(context.model, context.tasks, context.probe_ids)
