"""Logical clients share one physical LLaVA backbone and swap adapter states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.data.partition import sample_ids_hash
from afl_vlm.federation.types import ClientContext, Update
from afl_vlm.models.base import TrainConfig, add_scaled, clone_state, subtract


class FederatedClient:
    def __init__(self, client_id: str, task: str, dataset: str, samples: list[Any]) -> None:
        self.id, self.task, self.dataset = client_id, task, dataset
        self.samples = list(samples)

    def train(
        self,
        model: Any,
        method: Any,
        global_state: Mapping[str, Any],
        base_version: int,
        local_round: int,
        arrival_time: float,
        train_config: TrainConfig,
        fresh_global_state: Mapping[str, Any] | None = None,
        fresh_global_version: int | None = None,
        group_id: int | None = None,
        group_size: int | None = None,
    ) -> Update:
        context = ClientContext(
            client_id=self.id,
            task=self.task,
            dataset=self.dataset,
            num_samples=len(self.samples),
            local_round=local_round,
            base_version=base_version,
            arrival_time=arrival_time,
            seed=train_config.seed,
            fresh_global_state=clone_state(fresh_global_state)
            if fresh_global_state is not None
            else None,
            fresh_global_version=fresh_global_version,
        )
        start_state = method.prepare_download(clone_state(global_state), context)
        update = self.train_prepared(
            model=model,
            method=method,
            global_state=global_state,
            start_state=start_state,
            context=context,
            train_config=train_config,
            group_id=group_id,
            group_size=group_size,
        )
        return method.prepare_upload(update, context)

    def train_prepared(
        self,
        model: Any,
        method: Any,
        global_state: Mapping[str, Any],
        start_state: Mapping[str, Any],
        context: ClientContext,
        train_config: TrainConfig,
        group_id: int | None = None,
        group_size: int | None = None,
        fresh_state_provider: Any | None = None,
    ) -> Update:
        """Train one already-dispatched package without touching server-owned state.

        The parent process must call ``prepare_download`` before dispatch and
        ``prepare_upload`` after this raw update returns.  This mirrors the
        pack/train/unpack boundary used by established asynchronous FL runtimes.
        """
        model.load_trainable(start_state)
        model._set_context(self.task, self.id)
        configured = TrainConfig(
            local_epochs=train_config.local_epochs,
            batch_size=train_config.batch_size,
            gradient_accumulation=train_config.gradient_accumulation,
            learning_rate=train_config.learning_rate,
            max_text_length=train_config.max_text_length,
            seed=train_config.seed,
            loss_hook=method.local_loss,
            gradient_hook=method.transform_gradients,
            step_hook=method.local_step,
            progress_hook=train_config.progress_hook,
            collect_mean_gradient=train_config.collect_mean_gradient,
            context={
                "client": context,
                "method": method.name,
                "base_state": clone_state(global_state),
                "fresh_state_provider": fresh_state_provider,
            },
            planned_optimizer_steps=train_config.planned_optimizer_steps,
            max_local_steps=train_config.max_local_steps,
            distributed_rank=train_config.distributed_rank,
            distributed_world_size=train_config.distributed_world_size,
        )
        loaded_start_state = model.snapshot_trainable()
        result = model.local_train(self.samples, configured, self.task)
        local_state = add_scaled(loaded_start_state, result.delta, 1.0)
        update = Update(
            update_id=f"{self.id}-r{context.local_round}",
            client_id=self.id,
            task=self.task,
            dataset=self.dataset,
            num_samples=len(self.samples),
            local_round=context.local_round,
            base_version=context.base_version,
            arrival_time=context.arrival_time,
            base_state=clone_state(global_state),
            local_state=clone_state(local_state),
            delta=subtract(local_state, global_state),
            seed=train_config.seed,
            sample_ids_hash=sample_ids_hash(self.samples),
            losses=list(result.losses),
            optimizer_steps=result.optimizer_steps,
            mean_gradient=clone_state(result.mean_gradient)
            if result.mean_gradient is not None
            else None,
            group_id=group_id,
            metadata={
                **dict(result.extra),
                "group_size": group_size,
                "federated_parameter_scope": "lora_and_configured_modules",
            },
        )
        return update
