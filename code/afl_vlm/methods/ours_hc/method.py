"""User's independent Hard-Concrete + module-aware asynchronous LoRA design.

Source design: thread01a163c2-00b0-74b3-ab3c-6faef58610a9, final module-level
revision; user26Oct2026 explicitly requires relearning gates on EVERY local job.
No per-rank sensitivity history, SVD, rotation, or task-end age protection.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.ours import _discover_model_modules, _shape, _state_pairs
from afl_vlm.methods.ours_hc import payload
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import clone_state


@register_method("ours_hc")
class OursHC(Method):
    name = "ours_hc"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_staleness=True,
        requires_local_state=True,
        requires_custom_adapter=True,
        requires_group_sensitivity=True,
    )
    allowed_params = {
        "l0_lambda",
        "gate_temperature",
        "gate_lower",
        "gate_upper",
        "gate_learning_rate",
        "gate_initial_active_probability",
        "gate_warmup_steps",
        "fixed_tail_fraction",
        "sensitivity_uniform_mix",
        "gamma",
        "history_beta",
        "history_lambda",
        "frequency_window_size",
        "frequency_power",
        "frequency_min",
        "frequency_max",
        "eps",
        "min_update_energy",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.validate_runtime()
        self.modules, self.client_tasks, self.history, self.history_counts = {}, {}, {}, {}
        self.task_weights, self.downloads, self.seen = {}, {}, set()
        self.window = []
        self.controller = None

    def value(self, key: str, default: float) -> float:
        return float(self.params.get(key, default))

    def validate_runtime(self) -> None:
        defaults = {
            "gate_temperature": 2 / 3,
            "gate_learning_rate": 0.2,
            "gamma": 1.0,
            "history_lambda": 1.0,
            "eps": 1e-12,
            "min_update_energy": 1e-16,
        }
        for key, default in defaults.items():
            if not math.isfinite(self.value(key, default)) or self.value(key, default) <= 0:
                raise ValueError(f"ours_hc {key} must be finite and positive")
        for key, default in (("l0_lambda", 1e-4), ("frequency_power", 0.5)):
            if not math.isfinite(self.value(key, default)) or self.value(key, default) < 0:
                raise ValueError(f"ours_hc {key} must be finite and non-negative")
        for key, default in (
            ("history_beta", 0.9),
            ("gate_initial_active_probability", 0.9),
            ("fixed_tail_fraction", 0.2),
        ):
            if not 0 < self.value(key, default) < 1:
                raise ValueError(f"ours_hc {key} must lie in (0,1)")
        if not 0 <= self.value("sensitivity_uniform_mix", 0.75) <= 1:
            raise ValueError("sensitivity_uniform_mix must lie in [0,1]")
        for key, default, lower in (("gate_warmup_steps", 1, 0), ("frequency_window_size", 18, 1)):
            value = self.params.get(key, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f"ours_hc {key} must be an integer >= {lower}")
        if not (
            math.isfinite(self.value("gate_lower", -0.1))
            and math.isfinite(self.value("gate_upper", 1.1))
            and self.value("gate_lower", -0.1) < 0 < 1 < self.value("gate_upper", 1.1)
        ):
            raise ValueError("Hard Concrete endpoints require lower<0<1<upper")
        if not (
            0 < self.value("frequency_min", 0.5) <= self.value("frequency_max", 1.5)
            and math.isfinite(self.value("frequency_max", 1.5))
        ):
            raise ValueError("Invalid frequency bounds")

    def configure_model(self, model: Any, clients: list[Any]) -> None:
        del clients
        from afl_vlm.methods.ours_hc.local import install

        self.controller = install(model, self.params)

    def configure_server(self, initial_state: dict, clients: list[Any]) -> None:
        self.client_tasks = {client.id: client.task for client in clients}
        tasks = sorted(set(self.client_tasks.values()))
        if not tasks:
            raise ValueError("ours_hc requires task identities")
        self.task_weights = dict.fromkeys(tasks, 1 / len(tasks))
        discovered = _discover_model_modules(self.controller.model) if self.controller else {}
        for module, pair in _state_pairs(initial_state).items():
            an, a = pair["A"]
            bn, b = pair["B"]
            ashape, bshape = _shape(a), _shape(b)
            if len(ashape) != 2 or len(bshape) != 2 or ashape[0] != bshape[1]:
                raise ValueError("Invalid global LoRA shapes")
            self.modules[module] = {
                "module_name": module,
                "a_name": an,
                "b_name": bn,
                "rank": ashape[0],
                "in_features": ashape[1],
                "out_features": bshape[0],
                "scaling": discovered[module][2].scaling if discovered else None,
            }
        self.history = {
            task: {module: self.history.get(task, {}).get(module, 1.0) for module in self.modules}
            for task in tasks
        }
        self.history_counts = {task: self.history_counts.get(task, 0) for task in tasks}

    def prepare_download(self, global_state: dict, context: Any) -> dict:
        key = (context.client_id, context.local_round)
        if key in self.downloads:
            raise ValueError("Duplicate in-flight Hard Concrete client job")
        # The runtime passed an immutable download snapshot; reuse its tensor
        # storage, never mutate it, and release it when this arrival is resolved.
        self.downloads[key] = (context.base_version, global_state)
        return global_state

    def prepare_local_model(self, model: Any, context: Any) -> None:
        del context
        self.controller = model._ours_hc

    def configure_local_training(self, train_config: Any) -> Any:
        if train_config.optimizer != "adamw" or train_config.gradient_estimation_only:
            raise ValueError("ours_hc uses ordinary AdamW local training")
        self.controller.start(train_config.planned_optimizer_steps)
        return train_config

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping) -> Any:
        del model, batch, context
        return base_loss + self.controller.penalty()

    def transform_gradients(self, model: Any, context: Mapping) -> None:
        del model, context
        self.controller.observe()

    def local_step(self, model: Any, step: int, total_steps: int, context: Mapping) -> dict:
        del model, total_steps, context
        self.controller.step(step)
        return {}

    def export_local_metadata(self, model: Any, context: Any) -> dict:
        del model, context
        return self.controller.finish()

    def pack_local_update(self, update: Update, context: Any) -> Update:
        del context
        return payload.pack(update)

    def prepare_upload(self, update: Update, context: Any) -> Update:
        if (update.client_id, update.local_round) != (context.client_id, context.local_round):
            raise ValueError("Hard Concrete upload identity differs from dispatch")
        payload.validate(update, self.modules)
        return update

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        payload.validate(update, self.modules)
        key = (update.client_id, update.local_round)
        if update.update_id in self.seen or key not in self.downloads:
            raise ValueError("Duplicate or missing Hard Concrete download snapshot")
        version, base = self.downloads[key]
        if (
            version != update.base_version
            or server_context.version < version
            or self.client_tasks.get(update.client_id) != update.task
        ):
            raise ValueError("Invalid Hard Concrete base version or task identity")
        for module, item in self.modules.items():
            if item["scaling"] is None:
                item["scaling"] = float(update.metadata["lora_metadata"][module]["scaling"])
        drift, energy = payload.functional_energies(base, server_context.global_state, update)
        if not all(math.isfinite(value) for value in (drift, energy)):
            raise ValueError("Non-finite indexed LoRA update; history/frequency were not changed")
        relative = math.sqrt(drift / (energy + self.value("eps", 1e-12)))
        reliability = math.exp(-self.value("gamma", 1) * relative)
        sensitivity = update.metadata["module_sensitivity"]
        omega = {
            module: sum(
                weight * self.history[task][module] for task, weight in self.task_weights.items()
            )
            for module in self.modules
        }
        count = self.window.count(update.task)
        raw = ((len(self.window) / len(self.task_weights) + 1) / (count + 1)) ** self.value(
            "frequency_power", 0.5
        )
        weight = min(self.value("frequency_max", 1.5), max(self.value("frequency_min", 0.5), raw))
        eps, lam = self.value("eps", 1e-12), self.value("history_lambda", 1)
        alphas = {
            module: reliability
            * weight
            * float(sensitivity[module])
            / (reliability * weight * float(sensitivity[module]) + lam * omega[module] + eps)
            if update.metadata["hc_selection"][module]
            else 0.0
            for module in self.modules
        }
        accepted = (
            all(math.isfinite(value) for value in (drift, energy, relative))
            and energy >= self.value("min_update_energy", 1e-16)
            and sum(alphas.values()) > 0
        )
        diagnostics = {
            "method": self.name,
            "base_version": version,
            "update_id": update.update_id,
            "client_id": update.client_id,
            "task": update.task,
            "optimizer_steps": update.optimizer_steps,
            "arrival_time": update.arrival_time,
            "num_samples": update.num_samples,
            "server_version_before": server_context.version,
            "version_staleness": server_context.version - version,
            "selected_indices": copy.deepcopy(update.metadata["hc_selection"]),
            "selected_rank": {
                name: len(indices) for name, indices in update.metadata["hc_selection"].items()
            },
            "module_sensitivity": dict(sensitivity),
            "sensitivity_valid": bool(update.metadata.get("sensitivity_valid", False)),
            "sensitivity_observations": dict(update.metadata.get("sensitivity_observations", {})),
            "sensitivity_transform": copy.deepcopy(update.metadata["sensitivity_transform"]),
            "local": copy.deepcopy(update.metadata["hc_local"]),
            "communication": copy.deepcopy(update.metadata["hc_communication"]),
            "functional_drift": drift,
            "functional_local_energy": energy,
            "relative_staleness": relative,
            "reliability": reliability,
            "frequency_weight": weight,
            "frequency_counts_before": dict(
                (task, self.window.count(task)) for task in self.task_weights
            ),
            "history_before": copy.deepcopy(self.history),
            "omega_before": omega,
            "module_alpha_proposed": dict(alphas),
            "module_alpha": dict(alphas) if accepted else dict.fromkeys(alphas, 0.0),
            "accepted": accepted,
        }
        new_state = (
            payload.fuse(server_context.global_state, update, alphas)
            if accepted
            else (clone_state(server_context.global_state))
        )
        if not payload.finite_state(new_state):
            raise ValueError("Non-finite fusion result; history/frequency were not changed")
        if accepted:
            for module in self.modules:
                c = float(sensitivity[module])
                self.history[update.task][module] = (
                    c
                    if not self.history_counts[update.task]
                    else (
                        self.value("history_beta", 0.9) * self.history[update.task][module]
                        + (1 - self.value("history_beta", 0.9)) * c
                    )
                )
            self.history_counts[update.task] += 1
            self.window.append(update.task)
            self.window = self.window[-int(self.params.get("frequency_window_size", 18)) :]
        diagnostics["history_after"] = copy.deepcopy(self.history)
        diagnostics["frequency_counts_after"] = {
            task: self.window.count(task) for task in self.task_weights
        }
        self.seen.add(update.update_id)
        del self.downloads[key]
        return [
            ServerMutation(
                new_state,
                sum(alphas.values()) / len(alphas) if accepted else 0,
                [update.update_id],
                increment_version=accepted,
                metadata={
                    "hc_diagnostics": diagnostics,
                    **({} if accepted else {"rejected": "empty_or_no_effective_update"}),
                },
            )
        ]

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "params": self.params,
                "modules": self.modules,
                "client_tasks": self.client_tasks,
                "history": self.history,
                "history_counts": self.history_counts,
                "task_weights": self.task_weights,
                "window": self.window,
                "seen": self.seen,
                "downloads": self.downloads,
            }
        )

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        self.validate_runtime()
        for key in (
            "modules",
            "client_tasks",
            "history",
            "history_counts",
            "task_weights",
            "window",
            "seen",
            "downloads",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
