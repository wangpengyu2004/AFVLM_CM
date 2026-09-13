"""Sequential replay client; virtual concurrency lives in the event schedule."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from afl_vlm.data.partition import sample_ids_hash
from afl_vlm.federation.types import ClientContext, Update
from afl_vlm.models.base import TrainConfig, clone_state


class FederatedClient:
    def __init__(self, client_id: str, task: str, samples: list[Any]) -> None:
        self.id = client_id
        self.task = task
        self.samples = list(samples)

    def train(
        self,
        model: Any,
        method: Any,
        global_state: Mapping[str, Any],
        download_version: int,
        local_round: int,
        samples: list[Any],
        train_config: TrainConfig,
        update_kind: str = "normal",
        pair_id: str | None = None,
    ) -> Update:
        server_state = model.snapshot_trainable()
        context = ClientContext(
            client_id=self.id,
            task=self.task,
            local_round=local_round,
            download_version=download_version,
            seed=train_config.seed,
        )
        start_state = method.prepare_download(clone_state(global_state), context)
        model.load_trainable(start_state)
        try:
            configured = TrainConfig(
                local_steps=train_config.local_steps,
                batch_size=train_config.batch_size,
                grad_accumulation=train_config.grad_accumulation,
                client_lr=train_config.client_lr,
                max_text_length=train_config.max_text_length,
                seed=train_config.seed,
                loss_hook=method.local_loss,
                context={"client": context, "method": method.name},
            )
            result = model.local_train(samples, configured, self.task)
        finally:
            model.load_trainable(server_state)
        update = Update(
            update_id=f"{self.id}-r{local_round}-{update_kind}",
            client_id=self.id,
            task=self.task,
            local_round=local_round,
            download_version=download_version,
            # The base is the server state at download, not the method-specific
            # client initialization. This keeps staleness drift well-defined;
            # a download correction influences local optimization without being
            # re-applied as a raw server update.
            base_state=clone_state(global_state),
            delta=clone_state(result.delta),
            seed=train_config.seed,
            sample_ids_hash=sample_ids_hash(samples),
            losses=list(result.losses),
            optimizer_steps=result.optimizer_steps,
            update_kind=update_kind,
            pair_id=pair_id,
            metadata={
                **dict(result.extra),
                "num_examples": len(self.samples),
                "batch_examples": len(samples),
            },
        )
        return method.prepare_upload(update, context)
