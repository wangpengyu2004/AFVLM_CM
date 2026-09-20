"""Offline original-LLaVA-1.5-7B LoRA adapter for AFVLM-CM."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from afl_vlm.models.base import (
    LoRAState,
    ModelAdapter,
    TrainConfig,
    TrainResult,
    scale_state,
    subtract,
    zeros_like,
)
from afl_vlm.models.registry import register_model


@register_model("llava15")
class Llava15Adapter(ModelAdapter):
    """Keep one frozen 7B backbone in memory and swap small adapter states."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.image_processor: Any = None
        self.device: Any = None
        self._config: dict[str, Any] = {}
        self._pilot_connector: Any = None

    @staticmethod
    def _imports() -> tuple[Any, ...]:
        try:
            import torch
            from llava import conversation as conversation_lib
            from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from llava.mm_utils import process_images, tokenizer_image_token
            from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from PIL import Image
            from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA dependencies are missing. Install with: pip install -e '.[llava]'"
            ) from exc
        return (
            torch,
            conversation_lib,
            DEFAULT_IMAGE_TOKEN,
            IMAGE_TOKEN_INDEX,
            process_images,
            tokenizer_image_token,
            LlavaLlamaForCausalLM,
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            Image,
            AutoConfig,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

    def load(self, config: Mapping[str, Any]) -> None:
        (
            torch,
            _,
            _,
            _,
            _,
            _,
            ModelClass,
            LoraConfig,
            get_peft_model,
            prepare_kbit,
            _,
            AutoConfig,
            AutoTokenizer,
            BitsAndBytesConfig,
        ) = self._imports()
        self._config = dict(config)
        model_path = Path(str(config["model_path"])).resolve()
        vision_path = Path(str(config["vision_tower_path"])).resolve()
        if not model_path.is_dir() or not vision_path.is_dir():
            raise FileNotFoundError(
                f"Local LLaVA/CLIP weights are required: {model_path}, {vision_path}. "
                "Run tools/download_models.py first."
            )
        if not bool(config.get("local_files_only", True)):
            raise ValueError("LLaVA training is intentionally local-files-only")
        dtype = torch.bfloat16 if config.get("dtype", "bf16") == "bf16" else torch.float16
        quantization = str(config.get("quantization", "none"))
        quant_config = None
        if quantization == "nf4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        base_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        base_config.mm_vision_tower = str(vision_path)
        kwargs: dict[str, Any] = {
            "config": base_config,
            "torch_dtype": dtype,
            "device_map": config.get("device_map", "auto"),
            "local_files_only": True,
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        self.model = ModelClass.from_pretrained(model_path, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, use_fast=False
        )
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            # Official LLaVA v1.1.3 exposes load_model(self) without device_map.
            # Load from the configured local path first, then colocate the tower
            # with the token embeddings used by multimodal input preparation.
            vision_tower.load_model()
        vision_device = self.model.get_input_embeddings().weight.device
        if vision_device.type == "meta":
            raise RuntimeError("LLaVA input embeddings remain on the meta device after loading")
        vision_tower.to(device=vision_device, dtype=dtype)
        self.image_processor = vision_tower.image_processor
        if quant_config is not None:
            self.model = prepare_kbit(
                self.model,
                use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
            )
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
        lora = dict(config["lora"])
        peft = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(lora["target_modules"]),
            bias=str(lora.get("bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft)
        for name, parameter in self.model.named_parameters():
            if "mm_projector" in name:
                parameter.requires_grad = bool(lora.get("train_mm_projector", False))
        self.device = self.model.get_input_embeddings().weight.device

    def named_federated_parameters(self) -> list[tuple[str, Any]]:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

    def snapshot_trainable(self) -> LoRAState:
        """Return FP32 trainable state on the model's current device."""
        return {
            name: parameter.detach().float().clone()
            for name, parameter in self.named_federated_parameters()
        }

    def load_trainable(self, state: Mapping[str, Any]) -> None:
        trainable = dict(self.named_federated_parameters())
        if set(trainable) != set(state):
            missing, extra = set(trainable) - set(state), set(state) - set(trainable)
            raise ValueError(
                "Federated state keys differ: "
                f"missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}"
            )
        for name, parameter in trainable.items():
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def install_pilot_extension(
        self, tasks: list[str], clients: list[str], bottleneck: int
    ) -> None:
        from afl_vlm.methods.pilot.adapters import build_pilot_connector

        base = self.model.base_model.model.model.mm_projector
        hidden = int(self.model.config.hidden_size)
        wrapper = build_pilot_connector(base, hidden, tasks, clients, bottleneck)
        self.model.base_model.model.model.mm_projector = wrapper
        self._pilot_connector = wrapper

    def pilot_auxiliary_losses(self) -> dict[str, Any]:
        return dict(self._pilot_connector.auxiliary_losses) if self._pilot_connector else {}

    def _set_context(self, task: str, client: str | None) -> None:
        if self._pilot_connector is not None:
            self._pilot_connector.set_context(task, client)

    def set_evaluation_context(self, task: str, client_id: str | None = None) -> None:
        self._set_context(task, client_id)

    def _image(self, sample: Any) -> Any:
        if not sample.image:
            raise ValueError(f"LLaVA sample has no image: {sample.id}")
        Image = self._imports()[10]
        with Image.open(sample.image) as handle:
            return handle.convert("RGB")

    def _encode(self, sample: Any, max_length: int, include_answer: bool = True) -> dict[str, Any]:
        (
            torch,
            conversation_lib,
            image_token,
            image_index,
            process_images,
            tokenizer_image_token,
            *_,
        ) = self._imports()
        template = str(self._config.get("conversation_template", "v1"))
        conv = conversation_lib.conv_templates[template].copy()
        question = f"{image_token}\n{sample.instruction}"
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample.answer if include_answer else None)
        prompt = conv.get_prompt()
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, image_index, return_tensors="pt")[
                :max_length
            ]
            .unsqueeze(0)
            .to(self.device)
        )
        image = process_images([self._image(sample)], self.image_processor, self.model.config)
        image = image.to(device=self.device, dtype=next(self.model.parameters()).dtype)
        encoded = {
            "input_ids": input_ids,
            "images": image,
        }
        if include_answer:
            prompt_conv = conversation_lib.conv_templates[template].copy()
            prompt_conv.append_message(prompt_conv.roles[0], question)
            prompt_conv.append_message(prompt_conv.roles[1], None)
            prompt_ids = tokenizer_image_token(
                prompt_conv.get_prompt(), self.tokenizer, image_index, return_tensors="pt"
            )
            labels = input_ids.clone()
            labels[:, : min(labels.shape[1], prompt_ids.shape[0])] = -100
            encoded["labels"] = labels
        return encoded

    def _encode_batch(
        self, samples: list[Any], max_length: int, include_answer: bool = True
    ) -> dict[str, Any]:
        """Collate samples into one dynamically padded multimodal model batch."""
        if not samples:
            raise ValueError("Cannot encode an empty LLaVA batch")
        (
            torch,
            conversation_lib,
            image_token,
            image_index,
            process_images,
            tokenizer_image_token,
            _,
            _,
            _,
            _,
            Image,
            _,
            _,
            _,
        ) = self._imports()
        template = str(self._config.get("conversation_template", "v1"))
        input_sequences: list[Any] = []
        label_sequences: list[Any] = []

        for sample in samples:
            if not sample.image:
                raise ValueError(f"LLaVA sample has no image: {sample.id}")
            question = f"{image_token}\n{sample.instruction}"
            conv = conversation_lib.conv_templates[template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], sample.answer if include_answer else None)
            input_ids = tokenizer_image_token(
                conv.get_prompt(), self.tokenizer, image_index, return_tensors="pt"
            )[:max_length]
            input_sequences.append(input_ids)

            if include_answer:
                prompt_conv = conversation_lib.conv_templates[template].copy()
                prompt_conv.append_message(prompt_conv.roles[0], question)
                prompt_conv.append_message(prompt_conv.roles[1], None)
                prompt_ids = tokenizer_image_token(
                    prompt_conv.get_prompt(),
                    self.tokenizer,
                    image_index,
                    return_tensors="pt",
                )
                labels = input_ids.clone()
                labels[: min(labels.shape[0], prompt_ids.shape[0])] = -100
                label_sequences.append(labels)

        sequence_length = max(sequence.shape[0] for sequence in input_sequences)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.unk_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("The LLaVA tokenizer has no usable padding token")

        batch_size = len(input_sequences)
        input_ids = torch.full(
            (batch_size, sequence_length),
            int(pad_token_id),
            dtype=input_sequences[0].dtype,
        )
        attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool)
        labels = (
            torch.full((batch_size, sequence_length), -100, dtype=input_sequences[0].dtype)
            if include_answer
            else None
        )
        padding_side = str(getattr(self.tokenizer, "padding_side", "right"))
        if padding_side not in {"left", "right"}:
            raise ValueError(f"Unsupported tokenizer padding side: {padding_side}")
        for row, sequence in enumerate(input_sequences):
            length = sequence.shape[0]
            target = (
                slice(sequence_length - length, sequence_length)
                if padding_side == "left"
                else slice(0, length)
            )
            input_ids[row, target] = sequence
            attention_mask[row, target] = True
            if labels is not None:
                labels[row, target] = label_sequences[row]

        images: list[Any] = []
        try:
            for sample in samples:
                with Image.open(sample.image) as handle:
                    images.append(handle.convert("RGB"))
            image_batch = process_images(images, self.image_processor, self.model.config)
        finally:
            for image in images:
                image.close()
        if isinstance(image_batch, list):
            try:
                image_batch = torch.stack(image_batch, dim=0)
            except RuntimeError as exc:
                shapes = [tuple(image.shape) for image in image_batch]
                raise ValueError(
                    f"Processed images cannot be stacked into one batch: {shapes}"
                ) from exc
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        if image_batch.shape[0] != batch_size:
            raise ValueError(
                "Image processor returned a mismatched batch: "
                f"expected {batch_size}, got {image_batch.shape[0]}"
            )

        encoded = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "images": image_batch.to(device=self.device, dtype=next(self.model.parameters()).dtype),
        }
        if labels is not None:
            encoded["labels"] = labels.to(self.device)
        return encoded

    def local_train(
        self, task_batch_stream: Iterable[Any], train_config: TrainConfig, task_name: str
    ) -> TrainResult:
        torch = self._imports()[0]
        samples = list(task_batch_stream)
        if not samples:
            raise ValueError("Local training received no samples")
        rng = random.Random(train_config.seed)
        torch.random.default_generator.manual_seed(train_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(train_config.seed)
        start = self.snapshot_trainable()
        gradient_sum = zeros_like(start) if train_config.collect_mean_gradient else None
        parameter_groups = [
            {
                "params": [
                    p for n, p in self.named_federated_parameters() if "mm_projector" not in n
                ],
                "lr": train_config.learning_rate,
            }
        ]
        projector = [p for n, p in self.named_federated_parameters() if "mm_projector" in n]
        if projector:
            parameter_groups.append(
                {
                    "params": projector,
                    "lr": float(self._config.get("mm_projector_lr", train_config.learning_rate)),
                }
            )
        optimizer = torch.optim.AdamW(parameter_groups)
        losses, steps, hook_metadata = [], 0, {}
        client_context = train_config.context.get("client")
        client_id = getattr(client_context, "client_id", task_name)
        local_round = getattr(client_context, "local_round", "?")
        progress = tqdm(
            total=train_config.planned_optimizer_steps,
            desc=f"local {client_id} r{local_round}",
            unit="step",
            position=1,
            leave=False,
            dynamic_ncols=True,
            disable=not bool(self._config.get("progress_bar", True)),
        )
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        epoch = 0
        while train_config.max_local_steps is not None or epoch < train_config.local_epochs:
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            micro_batches = [
                indices[offset : offset + train_config.batch_size]
                for offset in range(0, len(indices), train_config.batch_size)
            ]
            for window_start in range(0, len(micro_batches), train_config.gradient_accumulation):
                window = micro_batches[
                    window_start : window_start + train_config.gradient_accumulation
                ]
                accumulated = None
                for batch_indices in window:
                    encoded = self._encode_batch(
                        [samples[index] for index in batch_indices],
                        train_config.max_text_length,
                    )
                    base_loss = self.model(**encoded).loss
                    loss = (
                        train_config.loss_hook(base_loss, self, encoded, train_config.context)
                        if train_config.loss_hook
                        else base_loss
                    )
                    (loss / len(window)).backward()
                    detached_loss = loss.detach()
                    accumulated = (
                        detached_loss if accumulated is None else accumulated + detached_loss
                    )
                if gradient_sum is not None:
                    for name, parameter in self.named_federated_parameters():
                        if parameter.grad is not None:
                            gradient_sum[name] = (
                                gradient_sum[name] + parameter.grad.detach().float()
                            )
                if train_config.gradient_hook:
                    train_config.gradient_hook(self, train_config.context)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if accumulated is None:
                    raise RuntimeError("Gradient accumulation window produced no loss")
                losses.append(float((accumulated / len(window)).item()))
                progress.set_postfix(loss=f"{losses[-1]:.4f}", refresh=False)
                progress.update(1)
                if train_config.progress_hook:
                    train_config.progress_hook(
                        steps, train_config.planned_optimizer_steps, losses[-1]
                    )
                if train_config.step_hook:
                    hook_metadata.update(
                        train_config.step_hook(
                            self,
                            steps,
                            train_config.planned_optimizer_steps,
                            train_config.context,
                        )
                    )
                if (
                    train_config.max_local_steps is not None
                    and steps >= train_config.max_local_steps
                ):
                    break
            epoch += 1
            if train_config.max_local_steps is not None and steps >= train_config.max_local_steps:
                break
        progress.close()
        if steps == 0:
            raise RuntimeError("No optimizer step was executed; lower gradient_accumulation")
        end = self.snapshot_trainable()
        return TrainResult(
            delta=subtract(end, start),
            losses=losses,
            optimizer_steps=steps,
            mean_gradient=scale_state(gradient_sum, 1.0 / steps)
            if gradient_sum is not None
            else None,
            extra=hook_metadata,
        )

    def evaluate(self, task_adapter: Any, sample_ids: list[str], mode: str) -> dict[str, float]:
        torch = self._imports()[0]
        samples = task_adapter.samples_by_id(sample_ids)
        self.model.eval()
        losses, predictions, references = [], [], []
        with torch.no_grad():
            for sample in tqdm(
                samples,
                desc=f"evaluate {task_adapter.task_key}",
                unit="sample",
                position=1,
                leave=False,
                dynamic_ncols=True,
                disable=not bool(self._config.get("progress_bar", True)),
            ):
                encoded = self._encode(
                    sample,
                    int(self._config.get("max_text_length", 512)),
                    include_answer=mode == "probe",
                )
                if mode == "probe":
                    losses.append(float(self.model(**encoded).loss.detach().cpu()))
                else:
                    generated = self.model.generate(
                        **encoded, max_new_tokens=int(self._config.get("max_new_tokens", 64))
                    )
                    predictions.append(
                        self.tokenizer.decode(
                            generated[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
                        ).strip()
                    )
                    references.append(sample.answer)
        return (
            {"loss": sum(losses) / len(losses)}
            if mode == "probe"
            else task_adapter.metric(predictions, references)
        )
