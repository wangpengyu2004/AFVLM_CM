"""LLaVA-1.5 7B LoRA adapter for AFVLM-CM experiments."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from afl_vlm.models.base import LoRAState, ModelAdapter, TrainConfig, TrainResult, subtract
from afl_vlm.models.registry import register_model


@register_model("llava15")
class Llava15Adapter(ModelAdapter):
    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.device: Any = None
        self._config: dict[str, Any] = {}

    @staticmethod
    def _imports() -> tuple[Any, ...]:
        try:
            import torch
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from PIL import Image
            from transformers import AutoProcessor, BitsAndBytesConfig
            from transformers import LlavaForConditionalGeneration as ModelClass
        except ImportError as exc:
            raise RuntimeError(
                "LLaVA dependencies are missing. Install with: pip install -e '.[llava]'"
            ) from exc
        return (
            torch,
            LoraConfig,
            get_peft_model,
            prepare_model_for_kbit_training,
            Image,
            AutoProcessor,
            BitsAndBytesConfig,
            ModelClass,
        )

    def load(self, config: Mapping[str, Any]) -> None:
        (
            torch,
            LoraConfig,
            get_peft_model,
            _,
            _,
            AutoProcessor,
            BitsAndBytesConfig,
            ModelClass,
        ) = self._imports()
        self._config = dict(config)
        dtype = torch.bfloat16 if config.get("dtype", "bf16") == "bf16" else torch.float16
        quantization = str(config.get("quantization", "none"))
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
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": config.get("device_map", "auto"),
        }
        if quant_config is not None:
            kwargs["quantization_config"] = quant_config
        hf_id = str(config["hf_id"])
        self.model = ModelClass.from_pretrained(hf_id, **kwargs)
        self.processor = AutoProcessor.from_pretrained(hf_id)
        if quant_config is not None:
            prepare_model_for_kbit_training = self._imports()[3]
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
            )
        if config.get("gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            self.model.config.use_cache = False
        lora = dict(config["lora"])
        if lora.get("include_visual", False):
            raise ValueError(
                "llava15 currently federates language-model LoRA only; "
                "set model.lora.include_visual to false"
            )
        peft_config = LoraConfig(
            r=int(lora["r"]),
            lora_alpha=int(lora["alpha"]),
            lora_dropout=float(lora["dropout"]),
            target_modules=list(
                lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            ),
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["multi_modal_projector"]
            if lora.get("train_mm_projector", True)
            else None,
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
            raise ValueError("LLaVA trainable state keys do not match the loaded LoRA adapter")
        for name, parameter in trainable.items():
            parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))

    @staticmethod
    def _messages(sample: Any, include_answer: bool) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "image"}]
        content.append({"type": "text", "text": sample.instruction})
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if include_answer:
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": sample.answer}]}
            )
        return messages

    def _image(self, sample: Any) -> Any:
        if not sample.image:
            raise ValueError(f"LLaVA sample has no image: {sample.id}")
        _, _, _, _, Image, *_ = self._imports()
        with Image.open(sample.image) as handle:
            return handle.convert("RGB")

    def _encode(self, sample: Any, max_length: int) -> dict[str, Any]:
        torch, *_ = self._imports()
        full_messages = self._messages(sample, include_answer=True)
        prompt_messages = self._messages(sample, include_answer=False)
        full_text = self.processor.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        image = self._image(sample)
        encoded = self.processor(
            text=[full_text],
            images=[image],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        prompt = self.processor(
            text=[prompt_text],
            images=[image],
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
        projector = []
        adapter = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (projector if "multi_modal_projector" in name else adapter).append(parameter)
        parameter_groups: list[dict[str, Any]] = [{"params": adapter, "lr": train_config.client_lr}]
        if projector:
            parameter_groups.append(
                {
                    "params": projector,
                    "lr": float(self._config.get("mm_projector_lr", train_config.client_lr)),
                }
            )
        optimizer = torch.optim.AdamW(parameter_groups)
        self.model.train()
        self.model.zero_grad(set_to_none=True)
        losses: list[float] = []
        for step in range(train_config.local_steps):
            accumulated = 0.0
            for micro_step in range(train_config.grad_accumulation):
                index = step * train_config.grad_accumulation + micro_step
                encoded = self._encode(samples[index % len(samples)], train_config.max_text_length)
                base_loss = self.model(**encoded).loss
                loss = (
                    train_config.loss_hook(base_loss, self.model, encoded, train_config.context)
                    if train_config.loss_hook is not None
                    else base_loss
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
                    continue
                messages = self._messages(sample, include_answer=False)
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                encoded = self.processor(
                    text=[text], images=[self._image(sample)], return_tensors="pt"
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
