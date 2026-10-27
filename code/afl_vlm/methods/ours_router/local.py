"""Instance-local variable-rank PEFT factors and persistent client-private routers.

No common model/training function is edited. Only selected rows/columns become
trainable parameters; the original scaling and global indices are preserved.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

from tqdm.auto import tqdm

from afl_vlm.batching import planned_optimizer_steps
from afl_vlm.methods.ours import ModuleSensitivityTracker, _discover_model_modules
from afl_vlm.methods.ours_router.capacity import spectral_capacity
from afl_vlm.methods.ours_router.router import PrivateRouter, component_features, component_targets
from afl_vlm.models.base import clone_state


def stable_seed(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:4], "big")


@contextmanager
def preserved_rng(device: Any):
    import numpy as np
    import torch

    python = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
            yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy_state)


class LocalRouter:
    def __init__(self, model: Any, params: dict) -> None:
        self.model, self.params = model, dict(params)
        self.full_metadata = {
            name: item.as_dict() for name, (_, _, item) in _discover_model_modules(model).items()
        }
        self.slots, self.private, self.capacities = {}, {}, {}
        self.active, self.base, self.choice = False, {}, {}
        self.output = None
        self.pending, self.trajectory, self.loss_sum, self.loss_count = {}, [], None, 0
        by_id = {id(a): name for name, (a, _, _) in _discover_model_modules(model).items()}
        for module in model._network().modules():
            for adapter, linear in getattr(module, "lora_A", {}).items():
                name = by_id.get(id(linear.weight))
                if name is not None:
                    self.slots[name] = (
                        module,
                        adapter,
                        linear.weight,
                        module.lora_B[adapter].weight,
                    )
        if set(self.slots) != set(self.full_metadata):
            raise ValueError("Cannot locate every language-side PEFT factor")
        if len({a.device for _, _, a, _ in self.slots.values()}) != 1:
            raise ValueError("ours_router requires one device per model replica; use run_one DDP")
        original_snapshot, original_load = model.snapshot_trainable, model.load_trainable

        def snapshot(this: Any) -> dict:
            del this
            if not self.active:
                return original_snapshot()
            result = clone_state(self.base)
            for name, (a, b, _) in _discover_model_modules(model).items():
                item, indices = self.full_metadata[name], self.index(name, a.device)
                result[item["a_name"]].index_copy_(0, indices, a.detach().float())
                result[item["b_name"]].index_copy_(1, indices, b.detach().float())
            return result

        def load(this: Any, state: dict) -> None:
            del this
            if self.active:
                raise RuntimeError("Do not replace a fixed local subset during training")
            original_load(state)

        model.snapshot_trainable = MethodType(snapshot, model)
        model.load_trainable = MethodType(load, model)

    def index(self, name: str, device: Any) -> Any:
        import torch

        return torch.tensor(self.choice[name], device=device, dtype=torch.long)

    def _unwrap(self) -> bool:
        distributed = self.model._ddp_enabled
        if distributed:
            old = self.model.model
            self.model.model = old.module
            self.model._ddp_enabled = False
            del old
            gc.collect()
        return distributed

    def _rewrap(self, distributed: bool) -> None:
        if distributed:
            self.model.enable_distributed_data_parallel(int(os.environ["LOCAL_RANK"]))

    def router(self, client: str, seed: int) -> PrivateRouter:
        if client not in self.private:
            dimensions = [
                (m["in_features"], m["out_features"]) for m in self.full_metadata.values()
            ]
            device = next(iter(self.slots.values()))[2].device
            self.private[client] = PrivateRouter(
                dimensions, device, stable_seed(seed, client), self.params
            )
        return self.private[client]

    def select(self, state: dict, context: Any) -> dict:
        import torch

        if context.client_id not in self.capacities:
            raise RuntimeError("Client capacity calibration must precede the virtual Plan")
        router = self.router(context.client_id, context.seed)
        chosen, sources = {}, {}
        with torch.no_grad():
            for name, item in self.full_metadata.items():
                a, b = state[item["a_name"]], state[item["b_name"]]
                k = self.capacities[context.client_id][name]["rank"]
                # Cold routers have no meaningful ranking, including late joiners.
                if router.steps == 0:
                    generator = torch.Generator(device=a.device)
                    generator.manual_seed(stable_seed(context.seed, name))
                    indices = torch.randperm(item["rank"], generator=generator, device=a.device)[:k]
                    sources[name] = "uniform_cold_start"
                else:
                    scores = router.predict(a, b)
                    if not bool(torch.isfinite(scores).all()):
                        raise ValueError("Non-finite private component scores")
                    # Stable tie-breaking in original global index order.
                    indices = torch.argsort(scores, descending=True, stable=True)[:k]
                    sources[name] = "router_topk"
                chosen[name] = sorted(indices.cpu().tolist())
        # One authoritative support, identical across all DDP replicas.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            message = [
                {"indices": chosen, "source": sources}
                if torch.distributed.get_rank() == 0
                else None
            ]
            torch.distributed.broadcast_object_list(message, src=0)
            return message[0]
        return {"indices": chosen, "source": sources}

    def prepare(self, context: Any, selection: dict) -> None:
        import torch

        if self.active:
            raise RuntimeError("Previous compact local model was not restored")
        self.job_started = time.perf_counter()
        self.context = context
        self.client_router = self.router(context.client_id, context.seed)
        self.selection = copy.deepcopy(selection)
        self.choice = self.selection["indices"]
        if set(self.choice) != set(self.full_metadata):
            raise ValueError("Fixed router support is missing a module")
        for name, indices in self.choice.items():
            if (
                not indices
                or indices != sorted(set(indices))
                or any(
                    isinstance(i, bool)
                    or not isinstance(i, int)
                    or not 0 <= i < self.full_metadata[name]["rank"]
                    for i in indices
                )
            ):
                raise ValueError("Fixed router indices must be nonempty, unique and in range")
        self.base = self.model.snapshot_trainable()
        was_ddp = self._unwrap()
        try:
            for name, (module, adapter, a, b) in self.slots.items():
                indices = self.index(name, a.device)
                module.lora_A[adapter].weight = torch.nn.Parameter(
                    a.detach().index_select(0, indices)
                )
                module.lora_B[adapter].weight = torch.nn.Parameter(
                    b.detach().index_select(1, indices)
                )
                module.lora_A[adapter].out_features = len(self.choice[name])
                module.lora_B[adapter].in_features = len(self.choice[name])
                # module.scaling is deliberately untouched: still global alpha/R.
                module.r[adapter] = len(self.choice[name])
            self.active = True
        finally:
            self._rewrap(was_ddp)
        self.tracker = ModuleSensitivityTracker(
            self.model,
            warmup_steps=int(self.params.get("sensitivity_warmup_steps", 1)),
            uniform_mix=float(self.params.get("sensitivity_uniform_mix", 0.75)),
            eps=float(self.params.get("eps", 1e-12)),
        )
        self.pending, self.trajectory, self.loss_sum, self.loss_count = {}, [], None, 0

    def task_loss(self, loss: Any) -> Any:
        value = loss.detach().float()
        self.loss_sum = value.clone() if self.loss_sum is None else self.loss_sum + value
        self.loss_count += 1
        return loss  # No router loss/score is added to the VLM objective.

    def observe(self, step: int) -> None:
        import torch

        self.tracker.update(step)
        self.pending = {}
        with torch.no_grad():
            for name, (a, b, _) in self.tracker.modules.items():
                if b.grad is not None:
                    self.pending[name] = (
                        PrivateRouter.key(a.shape[1], b.shape[0]),
                        component_features(a, b, float(self.params.get("eps", 1e-12))),
                        component_targets(b, b.grad),
                    )

    def step(self, step: int) -> None:
        import torch

        if not self.pending or self.loss_sum is None:
            raise RuntimeError("Router step is missing synchronized pre-update observations")
        report = self.client_router.learn(self.pending)
        report.update(optimizer_step=step)
        loss = torch.stack((self.loss_sum, self.loss_sum.new_tensor(self.loss_count)))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss)
        report["task_loss"] = float((loss[0] / loss[1]).cpu())
        self.trajectory.append(report)
        self.pending, self.loss_sum, self.loss_count = {}, None, 0

    def restore(self) -> None:
        if not self.active:
            return
        was_ddp = self._unwrap()
        for name, (module, adapter, a, b) in self.slots.items():
            module.lora_A[adapter].weight, module.lora_B[adapter].weight = a, b
            module.lora_A[adapter].out_features = self.full_metadata[name]["rank"]
            module.lora_B[adapter].in_features = self.full_metadata[name]["rank"]
            module.r[adapter] = self.full_metadata[name]["rank"]
        self.active, self.base = False, {}
        self.tracker = None
        self._rewrap(was_ddp)

    def finish(self) -> dict:
        sensitivity, valid = self.tracker.finalize()
        result = {
            "module_sensitivity": sensitivity,
            "sensitivity_valid": valid,
            "sensitivity_observations": dict(self.tracker.count),
            "sensitivity_transform": self.tracker.diagnostics(),
            "lora_metadata": copy.deepcopy(self.full_metadata),
            "hc_selection": copy.deepcopy(self.choice),
            "hc_local": {
                "selection_source": self.selection["source"],
                "trajectory": self.trajectory,
                "calibrated_ranks": {
                    name: row["rank"]
                    for name, row in self.capacities[self.context.client_id].items()
                },
                "router_lifetime_steps": self.client_router.steps,
                "router_parameter_count": sum(
                    p.numel() for p in self.client_router.heads.parameters()
                ),
                "selection_fixed_for_job": True,
            },
        }
        if self.output is not None and os.environ.get("AFVLM_PRIMARY_PROCESS", "1") == "1":
            import torch

            target = self.output / "client_private" / self.context.client_id
            target.mkdir(parents=True, exist_ok=True)
            torch.save(self.client_router.state_dict(), target / "router_latest.pt")
        self.restore()
        result["hc_local"]["local_wall_time_seconds"] = time.perf_counter() - self.job_started
        return result

    def calibrate(self, clients: dict, initial: dict, config: Any, output: Path) -> dict:
        import torch

        self.output = output
        records, started = [], time.perf_counter()
        if config.distributed_rank == 0:
            (output / "capacity_calibration.jsonl").write_text("", encoding="utf-8")
        device = next(iter(self.slots.values()))[2].device
        with preserved_rng(device):
            for number, client in enumerate(clients.values(), 1):
                client_started = time.perf_counter()
                epochs = int(self.params.get("probe_local_epochs", 1))
                layout = {
                    "batch_size": config.batch_size,
                    "gradient_accumulation": config.gradient_accumulation,
                    "batch_size_scope": config.batch_size_scope,
                    "reference_world_size": config.reference_world_size,
                }
                steps = planned_optimizer_steps(
                    len(client.samples), epochs, layout, config.distributed_world_size
                )
                recipe = replace(
                    config,
                    local_epochs=epochs,
                    planned_optimizer_steps=steps,
                    seed=stable_seed(config.seed, f"capacity:{client.id}"),
                    context={
                        "client": SimpleNamespace(client_id=client.id, local_round="calibration")
                    },
                )
                self.model.load_trainable(initial)
                if config.distributed_rank == 0:
                    tqdm.write(
                        f"[ours_router] capacity calibration {number}/{len(clients)}: "
                        f"{client.id}, epochs={epochs}, optimizer_steps={steps}"
                    )
                result = self.model.local_train(client.samples, recipe, client.task)
                if result.optimizer_steps != steps:
                    raise RuntimeError("Incomplete client capacity probe")
                trained = dict(self.model.named_federated_parameters())
                capacities = {}
                for name, item in self.full_metadata.items():
                    capacities[name] = spectral_capacity(
                        trained[item["a_name"]],
                        trained[item["b_name"]],
                        item["scaling"],
                        float(self.params.get("capacity_energy_threshold", 0.95)),
                        int(self.params.get("capacity_min_rank", 1)),
                        self.params.get("capacity_max_rank"),
                    )
                # SVD thresholds close to a boundary must not create different DDP shapes.
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    message = [capacities if config.distributed_rank == 0 else None]
                    torch.distributed.broadcast_object_list(message, src=0)
                    capacities = message[0]
                self.capacities[client.id] = capacities
                records.append(
                    {
                        "client_id": client.id,
                        "task": client.task,
                        "probe_epochs": epochs,
                        "seed": recipe.seed,
                        "num_samples": len(client.samples),
                        "learning_rate": recipe.learning_rate,
                        "optimizer_steps": steps,
                        "wall_time_seconds": time.perf_counter() - client_started,
                        "modules": capacities,
                        "counted_as_upload": False,
                    }
                )
                del result, trained
                self.model.load_trainable(initial)
                if config.distributed_rank == 0:
                    with (output / "capacity_calibration.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps(records[-1]) + "\n")
        return {
            "kind": "one_probe_per_client",
            "extra_work_not_in_plan": True,
            "uses_validation_or_test": False,
            "server_initialization_unchanged": True,
            "probe_optimizer_steps_total": sum(row["optimizer_steps"] for row in records),
            "wall_time_seconds": time.perf_counter() - started,
            "clients": records,
        }


def install(model: Any, params: dict) -> LocalRouter:
    if not hasattr(model, "_ours_router"):
        model._ours_router = LocalRouter(model, params)
    elif model._ours_router.params != dict(params):
        raise ValueError("Do not reuse a loaded router model with a different recipe")
    return model._ours_router
