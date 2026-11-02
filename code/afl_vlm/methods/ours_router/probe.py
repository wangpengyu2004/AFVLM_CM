"""Full-rank capacity probe + private-router warm-up + common version-zero LoRA.

Only conflict_router uses this preparation. The published legacy/overlap probes
keep their original behavior. No additional VLM forward/backward is introduced
for the router; detached pre-step observations use the existing trainer hooks.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from afl_vlm.batching import planned_optimizer_steps
from afl_vlm.methods.ours_router.capacity import spectral_capacity


def initialization_weights(clients: dict) -> dict[str, float]:
    """Equal total task weight; within a task, weight by its client's sample count."""
    totals = Counter()
    for client in clients.values():
        if not client.samples:
            raise ValueError("Capacity probe requires nonempty client training data")
        totals[client.task] += len(client.samples)
    if not totals:
        raise ValueError("Capacity probe requires clients")
    return {c.id: len(c.samples) / (len(totals) * totals[c.task]) for c in clients.values()}


def calibrate(controller: Any, clients: dict, initial: dict, config: Any, output: Path) -> dict:
    import torch
    from tqdm.auto import tqdm

    from afl_vlm.methods.ours_router.local import preserved_rng, stable_seed

    if controller.active or controller.probing or controller.initialized_state is not None:
        raise RuntimeError("Conflict capacity preparation must run once before formal training")
    if any(bool(torch.count_nonzero(initial[item["b_name"]]))
           for item in controller.full_metadata.values()):
        raise ValueError(
            "Full-rank probe capacity assumes zero initial B; do not estimate a nonzero "
            "LoRA endpoint as if it were the base-relative functional update"
        )
    controller.output = output
    weights = initialization_weights(clients)
    aggregate = {name: torch.zeros_like(value, dtype=torch.float32)
                 for name, value in initial.items()}
    records, started = [], time.perf_counter()
    if config.distributed_rank == 0:
        (output / "capacity_calibration.jsonl").write_text("", encoding="utf-8")
    device = next(iter(controller.slots.values()))[2].device
    with preserved_rng(device):
        try:
            for number, client in enumerate(clients.values(), 1):
                client_started = time.perf_counter()
                epochs = int(controller.params.get("probe_local_epochs", 1))
                layout = {"batch_size": config.batch_size,
                          "gradient_accumulation": config.gradient_accumulation,
                          "batch_size_scope": config.batch_size_scope,
                          "reference_world_size": config.reference_world_size}
                steps = planned_optimizer_steps(
                    len(client.samples), epochs, layout, config.distributed_world_size
                )
                context = SimpleNamespace(client_id=client.id, local_round="calibration")
                controller.model.load_trainable(initial)
                controller.client_router = controller.router(client.id, config.seed)
                controller.pending, controller.trajectory = {}, []
                controller.loss_sum, controller.loss_count = None, 0
                controller.probing = True
                recipe = replace(
                    config, local_epochs=epochs, planned_optimizer_steps=steps,
                    seed=stable_seed(config.seed, f"capacity:{client.id}"),
                    context={"client": context},
                    loss_hook=lambda loss, _model, _batch, _context: controller.task_loss(loss),
                    gradient_hook=lambda _model, ctx: controller.observe(
                        int(ctx["optimizer_step"])
                    ),
                    step_hook=lambda _model, step, _total, _context: (controller.step(step) or {}),
                )
                if config.distributed_rank == 0:
                    tqdm.write(f"[ours_router] full-rank probe + router warm-up "
                               f"{number}/{len(clients)}: {client.id}, epochs={epochs}, "
                               f"optimizer_steps={steps}")
                result = controller.model.local_train(client.samples, recipe, client.task)
                if result.optimizer_steps != steps:
                    raise RuntimeError("Incomplete full-rank capacity probe")
                if controller.client_router.steps == 0:
                    raise RuntimeError(f"No valid router supervision during probe: {client.id}")
                trained = dict(controller.model.named_federated_parameters())
                capacities = {}
                for name, item in controller.full_metadata.items():
                    capacities[name] = spectral_capacity(
                        trained[item["a_name"]], trained[item["b_name"]], item["scaling"],
                        float(controller.params.get("capacity_energy_threshold", 0.95)),
                        int(controller.params.get("capacity_min_rank", 1)),
                        controller.params.get("capacity_max_rank"),
                    )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    message = [capacities if config.distributed_rank == 0 else None]
                    torch.distributed.broadcast_object_list(message, src=0)
                    capacities = message[0]
                controller.capacities[client.id] = capacities
                with torch.no_grad():
                    for name, parameter in trained.items():
                        if not bool(torch.isfinite(parameter).all()):
                            raise ValueError("Non-finite full-rank probe parameters")
                        aggregate[name].add_(parameter.detach().float(), alpha=weights[client.id])
                row = {"client_id": client.id, "task": client.task, "probe_epochs": epochs,
                       "seed": recipe.seed, "num_samples": len(client.samples),
                       "learning_rate": recipe.learning_rate, "optimizer_steps": steps,
                       "wall_time_seconds": time.perf_counter() - client_started,
                       "modules": capacities, "counted_as_upload": False,
                       "initialization_weight": weights[client.id],
                       "router_lifetime_steps": controller.client_router.steps,
                       "router_sharing": controller.client_router.sharing,
                       "router_head_count": len(controller.client_router.heads),
                       "router_parameter_count": sum(
                           p.numel() for p in controller.client_router.heads.parameters()
                       ),
                       "trajectory": controller.trajectory}
                records.append(row)
                if config.distributed_rank == 0:
                    target = output / "client_private" / client.id
                    target.mkdir(parents=True, exist_ok=True)
                    torch.save(controller.client_router.state_dict(), target / "router_latest.pt")
                    with (output / "capacity_calibration.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(row) + "\n")
                del trained, result
                controller.model.load_trainable(initial)
        finally:
            controller.probing = False
            controller.pending, controller.trajectory = {}, []
            controller.loss_sum, controller.loss_count = None, 0
            controller.model.load_trainable(initial)
    if not all(bool(torch.isfinite(v).all()) for v in aggregate.values()):
        raise ValueError("Non-finite task-balanced probe initialization")
    controller.initialized_state = aggregate
    return {"kind": "full_rank_probe_router_warmup_and_initialization",
            "extra_work_not_in_plan": True, "uses_validation_or_test": False,
            "server_initialization_unchanged": False,
            "initialization": {"weighting": "equal_tasks_then_sample_weighted_clients",
                               "weights": weights, "server_version": 0,
                               "counted_as_upload": False, "conflict_window_initially_empty": True,
                               "averages_factors_not_effective_BA": True},
            "probe_optimizer_steps_total": sum(row["optimizer_steps"] for row in records),
            "wall_time_seconds": time.perf_counter() - started, "clients": records}
