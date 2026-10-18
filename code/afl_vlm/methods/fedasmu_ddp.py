"""FedASMU-private DDP oracle/policy bridge; no common model/trainer patches.

Training still uses the historical DDP AdamW body. Extra meta-objectives run on
the unwrapped eval-mode network, so autograd.grad never uses a DDP forward or a
re-entrant training checkpoint. Three scalar observations are explicitly reduced;
no parameter .grad, optimizer or reducer state is modified by this oracle.
Eval disables Hugging Face's training-only activation checkpoints, so the extra
gradient objective uses temporary non-reentrant checkpoints per decoder layer.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import wraps
from types import MethodType
from typing import Any


def distributed_active() -> bool:
    import torch

    return torch.distributed.is_available() and torch.distributed.is_initialized()


def controller_owner() -> bool:
    import torch

    return not distributed_active() or torch.distributed.get_rank() == 0


def select_decision(controller: Any, context: Any, total_steps: int) -> dict[str, Any]:
    """One authoritative RL decision; broadcast only small control metadata."""
    import torch

    decision = (
        controller.select(
            context.client_id, context.local_round, context.seed, total_steps, context.base_version
        )
        if controller_owner()
        else None
    )
    message = [decision]
    torch.distributed.broadcast_object_list(message, src=0)
    return message[0]


def reduce_observations(values: Any, records: int) -> tuple[Any, int, int]:
    """Equal-rank loss objective, matching ordinary DDP gradient averaging."""
    import torch

    world = torch.distributed.get_world_size()
    packed = torch.cat([values.detach().float(), values.new_tensor([float(records)]).float()])
    torch.distributed.all_reduce(packed)
    if not bool(torch.isfinite(packed).all()):
        raise ValueError("Non-finite FedASMU distributed objective feedback")
    return packed[:3] / world, int(packed[3].item()), world


def _checkpointed_forward(original: Any) -> Any:
    """Checkpoint one decoder, not the whole model (which has a large replay)."""
    import torch
    from torch.utils.checkpoint import checkpoint

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if not torch.is_grad_enabled():
            return original(*args, **kwargs)
        # Supported by the project's torch 2.0.1: unlike reentrant checkpoint,
        # this preserves keyword inputs and works with torch.autograd.grad.
        return checkpoint(original, *args, use_reentrant=False, **kwargs)

    return wrapped


@contextmanager
def objective_checkpoints(network: Any) -> Iterator[int]:
    """Temporary instance-only wrappers, active through gradient recomputation.

    Leave all modules in eval mode, preserve class/instance forward lookup,
    and restore even after a forward/autograd/OOM exception. Do not change
    common LLaVA checkpoint flags or install wrappers on its DDP shell.
    """
    import torch

    backbone = network.get_base_model()
    layers = backbone.get_model().layers
    if not isinstance(layers, torch.nn.ModuleList) or not len(layers):
        raise ValueError("FedASMU feedback requires LLaVA Llama decoder layers")
    saved = []
    try:
        for layer in layers:
            original = layer.forward
            had_override = "forward" in layer.__dict__
            override = layer.__dict__.get("forward")
            saved.append((layer, had_override, override))
            layer.forward = _checkpointed_forward(original)
        yield len(layers)
    finally:
        for layer, had_override, override in reversed(saved):
            if had_override:
                layer.forward = override
            else:
                delattr(layer, "forward")


def feedback(
    model: Any,
    left: dict[str, Any],
    right: dict[str, Any],
    weight: float,
    *,
    samples: list[Any] | None = None,
    encoded: Mapping[str, Any] | None = None,
    max_text_length: int | None = None,
) -> dict[str, Any]:
    """Read-only meta-gradient of mean rank losses; preserve LoRA/modes/.grad."""
    import torch

    from afl_vlm.models.base import add_scaled, subtract

    if not distributed_active():
        raise RuntimeError("FedASMU DDP feedback requires an initialized process group")
    if not 0 <= weight <= 1:
        raise ValueError("Feedback mixing weight must lie in [0, 1]")
    replicated_probe = encoded is None
    if encoded is None:
        if not samples:
            raise ValueError("Client objective feedback requires training samples")
        # Arrival probes use the exact same seeded training records on all
        # ranks, preserving the single-client server-objective definition.
        encoded = model._encode_batch(
            samples, max_text_length or int(model._config["max_text_length"])
        )
    saved = model.snapshot_trainable()
    modes = [(module, module.training) for module in model.model.modules()]
    direction = subtract(right, left)
    network = model._network()
    try:
        # Bypass the DDP forward/reducer ONLY for the extra objective. In eval
        # mode HF training checkpoints/dropout are off; main training is intact.
        # Disable KV/attention/hidden-state outputs for this loss-only oracle.
        # Copy the mapping: the caller's cached training batch stays unchanged.
        model.model.eval()
        objective_inputs = dict(encoded)
        objective_inputs.update(
            use_cache=False, output_attentions=False, output_hidden_states=False
        )
        model.load_trainable(left)
        with torch.no_grad():
            before = network(**objective_inputs).loss.detach().float()
        model.load_trainable(add_scaled(left, direction, weight))
        with torch.enable_grad(), objective_checkpoints(network) as checkpointed_layers:
            loss = network(**objective_inputs).loss
            parameters = dict(model.named_federated_parameters())
            gradients = torch.autograd.grad(loss, tuple(parameters.values()), allow_unused=True)
        derivative = loss.detach().new_zeros((), dtype=torch.float32)
        for (name, parameter), gradient in zip(parameters.items(), gradients, strict=True):
            if gradient is not None:
                derivative = (
                    derivative
                    + (gradient.detach().float() * direction[name].to(parameter).float()).sum()
                )
        values, total_records, world = reduce_observations(
            torch.stack([before, loss.detach().float(), derivative]),
            int(encoded["input_ids"].shape[0]),
        )
        return {
            "loss_before": float(values[0].item()),
            "loss_mixed": float(values[1].item()),
            "directional_derivative": float(values[2].item()),
            "objective_forwards": 2,
            "objective_backwards": 1,
            "optimizer_steps": 0,
            "batch_records": int(encoded["input_ids"].shape[0])
            if replicated_probe
            else total_records,
            "dropout_disabled": True,
            "objective_reduction": "mean_of_rank_losses",
            "objective_batch_scope": "replicated_arrival_training_probe"
            if replicated_probe
            else "ddp_last_local_microbatches",
            "objective_world_size": world,
            "objective_per_rank_batch_records": int(encoded["input_ids"].shape[0]),
            "objective_activation_checkpointing": "non_reentrant_decoder_layers",
            "objective_checkpointed_layers": checkpointed_layers,
            "physical_objective_forwards": 2 * world,
            "physical_objective_backwards": world,
            "physical_batch_records": total_records,
        }
    finally:
        model.load_trainable(saved)
        for module, training in modes:
            module.training = training


def install_feedback(model: Any) -> None:
    """Instance-local bridge, installed exclusively by FedASMU.configure_model."""
    if getattr(model, "_fedasmu_feedback_installed", False):
        return
    original = model.mixing_objective_feedback

    def dispatch(adapter: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if adapter._ddp_enabled:
            return feedback(adapter, *args, **kwargs)
        return original(*args, **kwargs)

    model.mixing_objective_feedback = MethodType(dispatch, model)
    model._fedasmu_feedback_installed = True
