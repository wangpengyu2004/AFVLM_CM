from __future__ import annotations

import unittest
from types import SimpleNamespace

from afl_vlm.models.llava15 import Llava15Adapter, _is_frozen_multimodal_parameter


class _FakeTensor(list[int]):
    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def new_full(self, shape: tuple[int], value: int) -> _FakeTensor:
        return _FakeTensor([value] * shape[0])

    def __getitem__(self, item: object) -> object:
        value = super().__getitem__(item)
        return _FakeTensor(value) if isinstance(item, slice) else value


class _FakeConversation:
    roles = ("USER", "ASSISTANT")

    def __init__(self) -> None:
        self.messages: list[tuple[str, str | None]] = []

    def copy(self) -> _FakeConversation:
        return _FakeConversation()

    def append_message(self, role: str, message: str | None) -> None:
        self.messages.append((role, message))

    def get_prompt(self) -> str:
        prompt = "SYSTEM;"
        for role, message in self.messages:
            prompt += f"{role}:"
            if message is None:
                return prompt
            prompt += f"{message};"
        return prompt


class Llava15LoRAScopeTests(unittest.TestCase):
    def test_vision_tower_lora_is_recognized_as_frozen(self) -> None:
        name = (
            "base_model.model.model.vision_tower.vision_tower.vision_model.encoder."
            "layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        self.assertTrue(_is_frozen_multimodal_parameter(name))

    def test_language_lora_remains_federated(self) -> None:
        name = (
            "base_model.model.model.layers.0.self_attn.q_proj."
            "lora_A.default.weight"
        )
        self.assertFalse(_is_frozen_multimodal_parameter(name))

    def test_similarly_named_component_does_not_match_by_substring(self) -> None:
        name = "base_model.model.model.layers.0.vision_tower_gate.weight"
        self.assertFalse(_is_frozen_multimodal_parameter(name))

    def test_multiturn_encoding_supervises_every_assistant_response(self) -> None:
        adapter = Llava15Adapter()
        adapter._config = {"conversation_template": "v1"}
        adapter.tokenizer = object()
        conversation_lib = SimpleNamespace(conv_templates={"v1": _FakeConversation()})

        def tokenize(prompt: str, *_args: object, **_kwargs: object) -> _FakeTensor:
            return _FakeTensor(ord(character) for character in prompt)

        sample = SimpleNamespace(
            instruction="Q1",
            answer="A1",
            turns=(("Q1", "A1"), ("Q2", "A2")),
        )
        input_ids, labels = adapter._conversation_ids(
            sample,
            10_000,
            True,
            conversation_lib,
            "<image>",
            -200,
            tokenize,
        )

        self.assertIsNotNone(labels)
        assert labels is not None
        supervised = "".join(chr(token) for token in labels if token != -100)
        self.assertIn("A1", supervised)
        self.assertIn("A2", supervised)
        self.assertNotIn("Q2", supervised)
        self.assertEqual(len(input_ids), len(labels))


if __name__ == "__main__":
    unittest.main()
