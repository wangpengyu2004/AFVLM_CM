"""Latest user design: one spectral probe, fixed-budget private top-k, async fusion.

Design source: thread 01a163c2-00b0-74b3-ab3c-6faef58610a9, 27 Oct 2026.
28 Oct: selection overlap switches the original sensitivity rule; freeze by task.
02 Nov: conflict_router uses probe-warmed selection and uniform eta*rho fusion.
Private router regression never enters the VLM loss or the server aggregation.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerMutation
from afl_vlm.methods.ours_hc import payload
from afl_vlm.methods.ours_hc.method import OursHC
from afl_vlm.methods.ours_layer_protection import select_layers, sum_extra
from afl_vlm.methods.ours_router.recent import RecentSelectionRegistry
from afl_vlm.methods.ours_router.wire import pack_indexed
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import clone_state


@register_method("ours_router")
class OursRouter(OursHC):
    name = "ours_router"
    capabilities = MethodCapabilities(
        mode="asynchronous",
        requires_task_id=True,
        requires_staleness=True,
        requires_local_state=True,
        requires_custom_adapter=True,
        requires_group_sensitivity=True,
        requires_client_calibration=True,
    )
    allowed_params = {
        "probe_local_epochs",
        "capacity_energy_threshold",
        "capacity_min_rank",
        "capacity_max_rank",
        "router_hidden_size",
        "router_sharing",
        "router_learning_rate",
        "router_label_beta",
        "router_huber_delta",
        "router_label_min",
        "router_max_grad_norm",
        "sensitivity_warmup_steps",
        "sensitivity_uniform_mix",
        "gamma",
        "frequency_window_size",
        "frequency_power",
        "frequency_min",
        "frequency_max",
        "history_beta",
        "history_lambda",
        "history_age_enabled",
        "history_age_boost",
        "history_age_time_scale",
        "eps",
        "min_update_energy",
        "rank_aggregation_mode",
        "selection_window_size",
        "server_step_size",
        "history_strength",
        "base_precision",
        "alpha_max",
        "conflict_per_task_window",
        "conflict_max_age",
        "conflict_penalty",
        "conflict_min_norm",
        "local_frozen_background",
        "staleness_mode",
        "router_supervision",
        "router_rank_lambda",
        "router_rank_min_gap",
    }

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.choices, self.planned, self.resolved = {}, {}, set()
        self.finished, self.accepted_count = {}, 0
        self.selection_registry = None
        self.conflict_window = None
        self.capabilities = replace(
            type(self).capabilities, requires_job_resolution=self.scoped_mode
        )

    @property
    def scoped_mode(self) -> bool:
        return self.params.get("rank_aggregation_mode", "legacy") == "overlap_switch"

    @property
    def conflict_mode(self) -> bool:
        return self.params.get("rank_aggregation_mode", "legacy") == "conflict_router"

    def validate_runtime(self) -> None:
        for key, default in (("router_rank_lambda", 0.0), ("router_rank_min_gap", 0.1)):
            if not math.isfinite(self.value(key, default)) or self.value(key, default) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        supervision = self.params.get("router_supervision", "selected")
        if supervision not in {"selected", "full_candidates"}:
            raise ValueError("Unknown router_supervision")
        if supervision == "full_candidates" and (
            not self.conflict_mode or not self.params.get("local_frozen_background", False)
        ):
            raise ValueError(
                "Full-candidate supervision requires conflict_router frozen background"
            )
        if not isinstance(self.params.get("local_frozen_background", False), bool):
            raise ValueError("local_frozen_background must be boolean")
        if self.params.get("staleness_mode", "selected_relative") not in {
            "selected_relative", "full_functional_absolute"
        }:
            raise ValueError("Unknown staleness_mode")
        if not self.conflict_mode and (
            self.params.get("local_frozen_background", False)
            or self.params.get("staleness_mode", "selected_relative") != "selected_relative"
        ):
            raise ValueError("Full-background/absolute drift options require conflict_router")
        if self.params.get("router_sharing", "dimension") not in {"dimension", "layer"}:
            raise ValueError("router_sharing must be dimension or layer")
        if self.params.get("rank_aggregation_mode", "legacy") not in {
            "legacy",
            "overlap_switch",
            "conflict_router",
        }:
            raise ValueError("Unknown rank_aggregation_mode")
        window = self.params.get("selection_window_size", 3)
        if type(window) is not int or window < 1:
            raise ValueError("selection_window_size must be a positive integer")
        if not 0 < self.value("server_step_size", 1) <= 1:
            raise ValueError("server_step_size must lie in (0,1]")
        strength = self.value("history_strength", 1)
        if not math.isfinite(strength) or strength < 0:
            raise ValueError("history_strength must be finite and nonnegative")
        if self.value("base_precision", 0) != 0:
            raise ValueError("Scoped router requires base_precision=0; no unconditional protection")
        maximum = self.params.get("alpha_max")
        if maximum is not None and not 0 < float(maximum) <= 1:
            raise ValueError("alpha_max must be null or in (0,1]")
        if self.conflict_mode:
            if maximum is not None:
                raise ValueError("conflict_router uses alpha=eta*rho without an extra hard cap")
            for key, default, minimum in (("conflict_per_task_window", 3, 1),
                                          ("conflict_max_age", 24, 0)):
                value = self.params.get(key, default)
                if type(value) is not int or value < minimum:
                    raise ValueError(f"{key} must be an integer >= {minimum}")
            if not math.isfinite(self.value("conflict_penalty", 0.3)) or self.value(
                "conflict_penalty", 0.3
            ) < 0:
                raise ValueError("conflict_penalty must be finite and nonnegative")
            if not math.isfinite(self.value("conflict_min_norm", 1e-12)) or self.value(
                "conflict_min_norm", 1e-12
            ) <= 0:
                raise ValueError("conflict_min_norm must be finite and positive")
        for key, default in (
            ("probe_local_epochs", 1),
            ("capacity_min_rank", 1),
            ("router_hidden_size", 32),
            ("frequency_window_size", 18),
            ("sensitivity_warmup_steps", 1),
        ):
            value = self.params.get(key, default)
            minimum = 0 if key == "sensitivity_warmup_steps" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{key} must be an integer >= {minimum}")
        maximum = self.params.get("capacity_max_rank")
        if maximum is not None and (
            isinstance(maximum, bool)
            or not isinstance(maximum, int)
            or maximum < int(self.params.get("capacity_min_rank", 1))
        ):
            raise ValueError("Invalid capacity_max_rank")
        for key, default in (
            ("router_learning_rate", 0.001),
            ("router_huber_delta", 1.0),
            ("router_max_grad_norm", 1.0),
            ("gamma", 1.0),
            ("history_lambda", 1.0),
            ("history_age_time_scale", 18.0),
            ("eps", 1e-12),
            ("min_update_energy", 1e-16),
        ):
            if not math.isfinite(self.value(key, default)) or self.value(key, default) <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key, default in (
            ("history_age_boost", 3.0),
            ("frequency_power", 0.5),
            ("router_label_min", 1e-12),
        ):
            if not math.isfinite(self.value(key, default)) or self.value(key, default) < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key, default in (("history_beta", 0.9), ("router_label_beta", 0.9)):
            if not 0 <= self.value(key, default) < 1:
                raise ValueError(f"{key} must lie in [0,1)")
        if not 0 < self.value("capacity_energy_threshold", 0.95) <= 1:
            raise ValueError("capacity_energy_threshold must lie in (0,1]")
        if not 0 <= self.value("sensitivity_uniform_mix", 0.75) <= 1:
            raise ValueError("sensitivity_uniform_mix must lie in [0,1]")
        if not (
            0 < self.value("frequency_min", 0.5) <= self.value("frequency_max", 1.5)
            and math.isfinite(self.value("frequency_max", 1.5))
        ):
            raise ValueError("Invalid frequency bounds")
        if not isinstance(self.params.get("history_age_enabled", True), bool):
            raise ValueError("history_age_enabled must be boolean")

    def configure_model(self, model: Any, clients: list) -> None:
        from afl_vlm.methods.ours_router.local import install

        self.controller = install(model, self.params)
        if any(
            int(self.params.get("capacity_min_rank", 1)) > m["rank"]
            for m in self.controller.full_metadata.values()
        ):
            raise ValueError("Minimum client capacity exceeds the global candidate rank")

    def calibrate_clients(
        self, model: Any, clients: Mapping, initial_state: dict, train_config: Any, output: Any
    ) -> dict:
        if train_config.optimizer != "adamw":
            raise ValueError("ours_router capacity probing and formal training require AdamW")
        return self.controller.calibrate(dict(clients), initial_state, train_config, output)

    def configure_schedule(self, events: list) -> None:
        if self.conflict_mode and self.planned and self.planned != {
            (e.client_id, e.local_round): e.task for e in events
        }:
            raise ValueError("Cannot replace a conflict-router Plan during a run")
        self.planned = {(e.client_id, e.local_round): e.task for e in events}
        if len(self.planned) != len(events):
            raise ValueError("Duplicate client participation in Plan")
        if self.scoped_mode:
            if self.selection_registry is None:
                raise ValueError("Configure the server before the selection Plan")
            self.selection_registry.configure(events)

    def configure_server(self, initial_state: dict, clients: list) -> None:
        saved_modules = (
            copy.deepcopy(self.modules) if self.scoped_mode or self.conflict_mode else {}
        )
        super().configure_server(initial_state, clients)
        if self.conflict_mode:
            from afl_vlm.methods.ours_router.conflict import TaskConflictWindow

            for name, saved in saved_modules.items():
                if name not in self.modules or any(
                    saved[k] != self.modules[name][k] for k in saved if k != "scaling"
                ):
                    raise ValueError("Restored conflict LoRA schema differs from the server")
                if self.modules[name]["scaling"] is None:
                    self.modules[name]["scaling"] = saved["scaling"]
            # Histories/frequencies belong only to previous explicit modes.
            self.history, self.history_counts, self.window = {}, {}, []
            if self.conflict_window is None:
                self.conflict_window = TaskConflictWindow(
                    self.modules, list(self.task_weights),
                    int(self.params.get("conflict_per_task_window", 3)),
                    int(self.params.get("conflict_max_age", 24)),
                    self.value("eps", 1e-12), self.value("conflict_min_norm", 1e-12),
                )
            elif self.conflict_window.modules != self.modules:
                raise ValueError("Restored conflict module schema differs from the server")
        if self.scoped_mode:
            for name, saved in saved_modules.items():
                if name not in self.modules or any(
                    saved[k] != self.modules[name][k] for k in saved if k != "scaling"
                ):
                    raise ValueError("Restored LoRA schema differs from the server")
                if self.modules[name]["scaling"] is None:
                    self.modules[name]["scaling"] = saved["scaling"]
            ranks = {name: row["rank"] for name, row in self.modules.items()}
            if self.selection_registry is None:
                self.selection_registry = RecentSelectionRegistry(
                    ranks, self.client_tasks, int(self.params.get("selection_window_size", 3))
                )
            elif (
                self.selection_registry.ranks != ranks
                or self.selection_registry.clients != self.client_tasks
            ):
                raise ValueError(
                    "Restored selection schema differs from server/client configuration"
                )

    def calibration_initial_state(self) -> dict | None:
        if not self.conflict_mode:
            return None
        if self.controller is None or self.controller.initialized_state is None:
            raise RuntimeError("Missing common LoRA initialization from the full-rank probe")
        state = self.controller.initialized_state
        self.controller.initialized_state = None
        return state

    def on_job_resolved(
        self, client_id: str, local_round: int, outcome: str, virtual_time: float | None = None
    ) -> dict:
        if not self.scoped_mode:
            return {}
        from afl_vlm.methods.ours_router.scoped_server import resolve_terminal

        return resolve_terminal(self, client_id, local_round, outcome, virtual_time)

    def prepare_download(self, global_state: dict, context: Any) -> dict:
        if self.conflict_mode and (
            (self.planned and self.planned.get((context.client_id, context.local_round))
             != context.task) or (context.client_id, context.local_round) in self.resolved
            or self.client_tasks.get(context.client_id) != context.task
        ):
            raise ValueError("Unknown or already incorporated conflict-router download")
        if self.scoped_mode and (
            self.selection_registry.planned.get((context.client_id, context.local_round))
            != context.task
            or (context.client_id, context.local_round) in self.selection_registry.resolved
            or context.client_id in self.selection_registry.exits
        ):
            raise ValueError("Unknown or terminal scoped router download")
        # Selection is made at the VIRTUAL download event, not delayed physical training.
        if any(client == context.client_id for client, _ in self.downloads):
            raise ValueError("A private client cannot overlap two local jobs in the Plan")
        result = super().prepare_download(global_state, context)
        if self.conflict_mode:
            if self.conflict_window is None:
                raise RuntimeError("Configure the conflict server before downloading")
            self.conflict_window.to(next(iter(result.values())).device)
            self.choices[(context.client_id, context.local_round)] = self.controller.select(
                result, context, self.conflict_window.scores(context.task)
            )
        else:
            self.choices[(context.client_id, context.local_round)] = self.controller.select(
                result, context
            )
        return result

    def client_runtime_state(self, context: Any) -> dict:
        return {"selection": copy.deepcopy(self.choices[(context.client_id, context.local_round)])}

    def load_client_runtime_state(self, state: Mapping, context: Any) -> None:
        self.local_selection = copy.deepcopy(state["selection"])

    def prepare_local_model(self, model: Any, context: Any) -> None:
        self.controller = model._ours_router
        self.controller.prepare(context, self.local_selection)

    def configure_local_training(self, train_config: Any) -> Any:
        if train_config.optimizer != "adamw" or train_config.gradient_estimation_only:
            raise ValueError("ours_router formal training requires ordinary AdamW")
        return train_config

    def local_loss(self, base_loss: Any, model: Any, batch: Any, context: Mapping) -> Any:
        return self.controller.task_loss(base_loss)

    def transform_gradients(self, model: Any, context: Mapping) -> None:
        self.controller.observe(int(context["optimizer_step"]))

    def local_step(self, model: Any, step: int, total_steps: int, context: Mapping) -> dict:
        self.controller.step(step)
        return {}

    def export_local_metadata(self, model: Any, context: Any) -> dict:
        return self.controller.finish()

    def pack_local_update(self, update: Any, context: Any) -> Any:
        return pack_indexed(update)

    def prepare_upload(self, update: Any, context: Any) -> Any:
        if not self.scoped_mode:
            return super().prepare_upload(update, context)
        from afl_vlm.methods.ours_router.scoped import validate

        if (update.client_id, update.local_round, update.task) != (
            context.client_id,
            context.local_round,
            context.task,
        ):
            raise ValueError("Scoped upload differs from its dispatched identity")
        validate(update, self.modules)
        return update

    def on_arrival(self, update: Any, server_context: Any) -> list[ServerMutation]:
        if self.conflict_mode:
            from afl_vlm.methods.ours_router.conflict_server import arrive

            return arrive(self, update, server_context)
        if self.scoped_mode:
            from afl_vlm.methods.ours_router.scoped_server import arrive

            return arrive(self, update, server_context)
        payload.validate(update, self.modules)
        key = (update.client_id, update.local_round)
        if key not in self.downloads or update.update_id in self.seen:
            raise ValueError("Duplicate/missing router download")
        version, base = self.downloads[key]
        if (
            version != update.base_version
            or version > server_context.version
            or self.client_tasks.get(update.client_id) != update.task
        ):
            raise ValueError("Router update version/task mismatch")
        if self.params.get("history_age_enabled", True) and self.planned.get(key) != update.task:
            raise ValueError("Router upload is absent from the configured Plan")
        for name in self.modules:
            if self.modules[name]["scaling"] is None:
                self.modules[name]["scaling"] = update.metadata["lora_metadata"][name]["scaling"]
        drift, energy = payload.functional_energies(base, server_context.global_state, update)
        if not all(math.isfinite(x) for x in (drift, energy)):
            raise ValueError("Non-finite router functional energy")
        eps = self.value("eps", 1e-12)
        relative = math.sqrt(drift / (energy + eps))
        if not math.isfinite(relative):
            raise ValueError("Non-finite router functional staleness")
        rho = math.exp(-self.value("gamma", 1) * relative)
        frequency_counts_before = {q: self.window.count(q) for q in self.task_weights}
        frequency = (
            (len(self.window) / len(self.task_weights) + 1) / (self.window.count(update.task) + 1)
        ) ** self.value("frequency_power", 0.5)
        frequency = min(
            self.value("frequency_max", 1.5), max(self.value("frequency_min", 0.5), frequency)
        )
        base_omega = {
            m: sum(w * self.history[q][m] for q, w in self.task_weights.items())
            for m in self.modules
        }
        boosts = {
            q: self.value("history_age_boost", 3)
            * -math.expm1(
                -(self.accepted_count - row["finished_at"])
                / self.value("history_age_time_scale", 18)
            )
            for q, row in self.finished.items()
        }
        extra, demands = (
            sum_extra(self.finished, self.task_weights, boosts, tuple(self.modules))
            if self.finished and self.params.get("history_age_enabled", True)
            else (dict.fromkeys(self.modules, 0.0), {})
        )
        omega = {m: base_omega[m] + extra[m] for m in self.modules}
        c = update.metadata["module_sensitivity"]
        precision = {m: rho * frequency * float(c[m]) for m in self.modules}
        alpha = {
            m: precision[m] / (precision[m] + self.value("history_lambda", 1) * omega[m] + eps)
            for m in self.modules
        }
        accepted = energy >= self.value("min_update_energy", 1e-16) and sum(alpha.values()) > 0
        history_before = copy.deepcopy(self.history)
        state = (
            payload.fuse(server_context.global_state, update, alpha)
            if accepted
            else (clone_state(server_context.global_state))
        )
        if not payload.finite_state(state):
            raise ValueError("Non-finite router fusion result")
        end_snapshot = None
        if accepted:
            for m in self.modules:
                self.history[update.task][m] = (
                    float(c[m])
                    if not self.history_counts[update.task]
                    else (
                        self.value("history_beta", 0.9) * self.history[update.task][m]
                        + (1 - self.value("history_beta", 0.9)) * float(c[m])
                    )
                )
            self.history_counts[update.task] += 1
            self.window = (self.window + [update.task])[
                -int(self.params.get("frequency_window_size", 18)) :
            ]
            self.accepted_count += 1
            self.resolved.add(key)
            remaining = any(
                task == update.task and job not in self.resolved
                for job, task in self.planned.items()
            )
            if self.params.get("history_age_enabled", True) and not remaining:
                end_snapshot = select_layers(
                    self.history[update.task], 1 - self.value("sensitivity_uniform_mix", 0.75), eps
                )
                end_snapshot["finished_at"] = self.accepted_count
                self.finished[update.task] = end_snapshot
        diagnostic = {
            "method": self.name,
            "update_id": update.update_id,
            "client_id": update.client_id,
            "task": update.task,
            "base_version": version,
            "server_version_before": server_context.version,
            "arrival_time": update.arrival_time,
            "optimizer_steps": update.optimizer_steps,
            "accepted": accepted,
            "version_staleness": server_context.version - version,
            "functional_drift": drift,
            "functional_local_energy": energy,
            "relative_staleness": relative,
            "reliability": rho,
            "frequency_weight": frequency,
            "frequency_counts_before": frequency_counts_before,
            "frequency_counts_after": {q: self.window.count(q) for q in self.task_weights},
            "history_accepted_updates_before": self.accepted_count - int(accepted),
            "history_accepted_updates_after": self.accepted_count,
            "history_age_boosts_before": boosts,
            "selected_indices": update.metadata["hc_selection"],
            "selected_rank": {m: len(v) for m, v in update.metadata["hc_selection"].items()},
            "module_sensitivity": c,
            "sensitivity_transform": update.metadata["sensitivity_transform"],
            "sensitivity_valid": update.metadata["sensitivity_valid"],
            "sensitivity_observations": update.metadata["sensitivity_observations"],
            "history_before": history_before,
            "history_after": copy.deepcopy(self.history),
            "omega_base": base_omega,
            "omega_extra": extra,
            "omega": omega,
            "ended_task_demands": demands,
            "finished_task_snapshot": end_snapshot,
            "module_alpha": alpha if accepted else dict.fromkeys(alpha, 0.0),
            "local": update.metadata["hc_local"],
            "communication": update.metadata["hc_communication"],
        }
        self.seen.add(update.update_id)
        del self.downloads[key]
        self.choices.pop(key, None)
        return [
            ServerMutation(
                state,
                sum(alpha.values()) / len(alpha) if accepted else 0,
                [update.update_id],
                increment_version=accepted,
                metadata={
                    "router_diagnostics": diagnostic,
                    **({} if accepted else {"rejected": "no_effective_update"}),
                },
            )
        ]

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["router_protocol"] = copy.deepcopy(
            {
                "schema_version": 3,
                "choices": self.choices,
                "planned": self.planned,
                "resolved": self.resolved,
                "finished": self.finished,
                "accepted_count": self.accepted_count,
            }
        )
        if self.scoped_mode:
            state["router_protocol"]["selection_registry"] = self.selection_registry.state_dict()
        if self.conflict_mode:
            state["router_protocol"]["schema_version"] = 4
            state["router_protocol"]["conflict_window"] = self.conflict_window.state_dict()
        # Router parameters/optimizers live in client_private, never global checkpoints.
        return state

    def load_state_dict(self, state: Mapping) -> None:
        source_mode = state["params"].get("rank_aggregation_mode", "legacy")
        if (self.conflict_mode or source_mode == "conflict_router") and (
            source_mode != self.params.get("rank_aggregation_mode", "legacy")
            or dict(state["params"]) != self.params
        ):
            raise ValueError("Conflict state cannot silently migrate methods or configurations")
        source_scoped = state["params"].get("rank_aggregation_mode", "legacy") == "overlap_switch"
        protocol = state["router_protocol"]
        if source_mode == "conflict_router" and (
            protocol.get("schema_version") != 4 or "conflict_window" not in protocol
        ):
            raise ValueError("Conflict state is missing its exact functional window")
        if (self.scoped_mode and not source_scoped) or (
            source_scoped
            and (protocol.get("schema_version") != 3 or "selection_registry" not in protocol)
        ):
            raise ValueError(
                "Scoped state requires selection history; "
                "restart rather than migrating legacy state"
            )
        super().load_state_dict(state)
        for name, value in protocol.items():
            if name in {"schema_version", "selection_registry", "conflict_window"}:
                continue
            setattr(self, name, copy.deepcopy(value))
        self.capabilities = replace(
            type(self).capabilities, requires_job_resolution=self.scoped_mode
        )
        if self.conflict_mode:
            from afl_vlm.methods.ours_router.conflict import TaskConflictWindow

            self.conflict_window = TaskConflictWindow(
                self.modules, list(self.task_weights),
                int(self.params.get("conflict_per_task_window", 3)),
                int(self.params.get("conflict_max_age", 24)),
                self.value("eps", 1e-12), self.value("conflict_min_norm", 1e-12),
            )
            self.conflict_window.load_state_dict(protocol["conflict_window"])
            if self.conflict_window.clock != self.accepted_count:
                raise ValueError("Conflict window and accepted update clocks differ")
        if source_scoped:
            ranks = {name: row["rank"] for name, row in self.modules.items()}
            self.selection_registry = RecentSelectionRegistry(
                ranks, self.client_tasks, int(self.params.get("selection_window_size", 3))
            )
            self.selection_registry.load_state_dict(protocol["selection_registry"])
            if (
                self.selection_registry.planned != self.planned
                or set(self.selection_registry.resolved) != self.resolved
            ):
                raise ValueError("Router and selection scheduler snapshots differ")
            if set(self.finished) != set(self.selection_registry.finished_tasks):
                raise ValueError("Frozen task histories and frozen task sets differ")
            for task, snapshot in self.finished.items():
                frozen = self.selection_registry.finished_tasks[task]
                if (
                    snapshot.get("selected_indices") != frozen["selected_indices"]
                    or snapshot["finished_at"] != frozen["accepted_updates_at_finish"]
                ):
                    raise ValueError("Frozen task history must match its frozen selection/time")
