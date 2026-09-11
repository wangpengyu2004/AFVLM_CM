"""Qwen2.5-VL LoRA/QLoRA adapter.

Heavy dependencies are imported lazily so dry-runs and ``tiny_mock`` need only
PyYAML. The adapter trains language projections by default and never serializes
the frozen base model as federated state.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from afl_vlm.models.base import (
    LoRAState,
    ModelAdapter,
    TrainConfig,
    TrainResult,
    subtract,
)
from afl_vlm.models.registry import register_model


@register_model("qwen25_vl")
class Qwen25VLAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: Any = None
        self._config: dict[str, Any] = {}

    @staticmethod
    def _imports() -> tuple[Any, ...]:
        try:
            import torch
            from peft import LoraConfig, get_peft_model
            from qwen_vl_utils import process_vision_info
            from transformers import (
                AutoProcessor,
                BitsAndBytesConfig,
                Qwen2_5_VLForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are missing. Install with: pip install -e '.[qwen]'"
            ) from exc
        return (
            torch,
            LoraConfig,
            get_peft_model,
            process_vision_info,
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

    def load(self, config: Mapping[str, Any]) -> None:
        (
            torch,
            LoraConfig,
            get_peft_model,
            _,
            AutoProcessor,
            BitsAndBytesConfig,
            ModelClass,
        ) = self._imports()
        self._config = dict(config)
        dtype_name = str(config.get("dtype", "bf16"))
        dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
        quantization = config.get("quantization", "none")
        quant_config = None
        if quantization == "nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("NF4 QLoRA requires CUDA; set model.quantization to 'none'")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.get("device_map", "auto"),
        }
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        hf_id = str(config["hf_id"])
        self.model = ModelClass.from_pretrained(hf_id, **model_kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        lora = dict(config["lora"])
        target_modules = list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]))
        if lora.get("include_visual", False) and "target_modules" not in lora:
            raise ValueError(
                "Visual-tower LoRA needs model-specific target modules; list them explicitly "
                "in model.lora.target_modules"
            )
        peft_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, peft_config)
        self.device = next(self.model.parameters()).device

    def snapshot_trainable(self) -> LoRAState:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def load_trainable(self, state: Mapping[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        trainable = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if set(trainable) != set(state):
            raise ValueError("Qwen trainable state keys do not match the loaded LoRA adapter")
        for name, parameter in trainable.items():
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    def _messages(self, sample: Any, include_answer: bool) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if sample.image:
            content.append({"type": "image", "image": sample.image})
        content.append({"type": "text", "text": sample.instruction})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if include_answer:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}
            )
        return messages

    def _encode(self, sample: Any, max_length: int) -> dict[str, Any]:
        torch, _, _, process_vision_info, *_ = self._imports()
        full_messages = self._messages(sample, include_answer=True)
        prompt_messages = self._messages(sample, include_answer=False)
        text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(full_messages)
        encoded = self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt = self.processor(
            text=[prompt_text],
            images=images,
            videos=videos,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = encoded["input_ids"].clone()
        prompt_length = min(prompt["input_ids"].shape[1], labels.shape[1])
        labels[:, :prompt_length] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        encoded["labels"] = labels
        return {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in encoded.items()
        }

    def local_train(
        self, task_batch_stream: Iterable[Any], train_config: TrainConfig, task_name: str
    ) -> TrainResult:
        torch, *_ = self._imports()
        samples = list(task_batch_stream)
        if not samples:
            raise ValueError("Local training received no samples")
        random.seed(train_config.seed)
        torch.manual_seed(train_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(train_config.seed)
        start = self.snapshot_trainable()
        optimizer = torch.optim.AdamW(
            (parameter for parameter in self.model.parameters() if parameter.requires_grad),
            lr=train_config.client_lr,
        )
        self.model.train()
        losses: list[float] = []
        optimizer.zero_grad(set_to_none=True)
        for step in range(train_config.local_steps):
            accumulated = 0.0
            for micro_step in range(train_config.grad_accumulation):
                sample = samples[
                    (step * train_config.grad_accumulation + micro_step) % len(samples)
                ]
                encoded = self._encode(sample, train_config.max_text_length)
                base_loss = self.model(**encoded).loss
                loss = base_loss
                if train_config.loss_hook is not None:
                    loss = train_config.loss_hook(
                        base_loss, self.model, encoded, train_config.context
                    )
                (loss / train_config.grad_accumulation).backward()
                accumulated += float(loss.detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(accumulated / train_config.grad_accumulation)
        end = self.snapshot_trainable()
        return TrainResult(
            delta=subtract(end, start),
            losses=losses,
            optimizer_steps=train_config.local_steps,
        )

    def evaluate(self, task_adapter: Any, sample_ids: list[str], mode: str) -> dict[str, float]:
        torch, *_ = self._imports()
        samples = task_adapter.samples_by_id(sample_ids)
        self.model.eval()
        losses: list[float] = []
        predictions: list[str] = []
        references: list[str] = []
        with torch.no_grad():
            for sample in samples:
                if mode == "probe":
                    encoded = self._encode(sample, int(self._config.get("max_text_length", 512)))
                    losses.append(float(self.model(**encoded).loss.detach().cpu()))
                else:
                    prompt_messages = self._messages(sample, include_answer=False)
                    text = self.processor.apply_chat_template(
                        prompt_messages, tokenize=False, add_generation_prompt=True
                    )
                    _, _, _, process_vision_info, *_ = self._imports()
                    images, videos = process_vision_info(prompt_messages)
                    encoded = self.processor(
                        text=[text], images=images, videos=videos, return_tensors="pt"
                    )
                    encoded = {
                        key: value.to(self.device) if torch.is_tensor(value) else value
                        for key, value in encoded.items()
                    }
                    generated = self.model.generate(**encoded, max_new_tokens=32)
                    new_tokens = generated[:, encoded["input_ids"].shape[1] :]
                    predictions.append(
                        self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
                    )
                    references.append(sample.answer)
        if mode == "probe":
            return {"loss": sum(losses) / len(losses)}
        return task_adapter.metric(predictions, references)
