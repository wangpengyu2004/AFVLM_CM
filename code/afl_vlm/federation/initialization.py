"""Real algorithm-required local-gradient initialization (not a validation mode).

MasFL Algorithm 1 evaluates K stochastic gradients at the unchanged initial
model on every client's own training data. No validation/test data, parameter
update, server aggregation, or fake model is involved. Returned records belong
to algorithm_initialization.json, not updates.jsonl or aggregation counters.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

from afl_vlm.batching import planned_optimizer_steps
from afl_vlm.federation.types import ClientContext, ScheduledEvent
from afl_vlm.federation.worker_pool import TrainJob
from afl_vlm.models.base import TrainConfig, clone_state


def initial_gradient_jobs(
    config: dict[str, Any],
    profiles: list[Any],
    initial_state: dict[str, Any],
) -> list[TrainJob]:
    training = config["training"]
    jobs = []
    for index, spec in enumerate(profiles):
        seed = int.from_bytes(
            hashlib.sha256(f"{config['run']['seed']}:{spec.id}:initial_gradient".encode()).digest()[
                :4
            ],
            "big",
        )
        steps = planned_optimizer_steps(spec.num_samples, int(training["local_epochs"]), training)
        event = ScheduledEvent(
            event_id=-index - 1,
            client_id=spec.id,
            task=spec.task,
            dataset=spec.dataset,
            local_round=-1,
            start_time=0,
            arrival_time=0,
            speed_factor=spec.speed_factor,
            estimated_train_time=0,
            network_delay=spec.network_delay,
            local_steps=steps,
        )
        context = ClientContext(spec.id, spec.task, spec.dataset, spec.num_samples, -1, 0, 0, seed)
        train_config = TrainConfig(
            local_epochs=int(training["local_epochs"]),
            batch_size=int(training["batch_size"]),
            gradient_accumulation=int(training["gradient_accumulation"]),
            learning_rate=float(training["learning_rate"]),
            max_text_length=int(training["max_text_length"]),
            seed=seed,
            collect_mean_gradient=True,
            planned_optimizer_steps=steps,
            batch_size_scope=str(training.get("batch_size_scope", "per_device")),
            reference_world_size=int(training.get("reference_world_size", 8)),
            gradient_estimation_only=True,
        )
        jobs.append(
            TrainJob(
                event,
                context,
                clone_state(initial_state),
                clone_state(initial_state),
                train_config,
                {"initial_gradient_only": True},
                None,
                time.time(),
            )
        )
    return jobs


def initial_gradient_record(update: Any) -> dict[str, Any]:
    if not update.metadata.get("gradient_estimation_only", False):
        raise ValueError("An initialization job unexpectedly optimized model parameters")
    return {
        "client_id": update.client_id,
        "gradient_observations": update.optimizer_steps,
        "batching": update.metadata.get("batching", {}),
        "counted_as_upload": False,
    }
