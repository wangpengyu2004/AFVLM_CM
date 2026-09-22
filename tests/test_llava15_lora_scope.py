from __future__ import annotations

import unittest

from afl_vlm.models.llava15 import _is_frozen_multimodal_parameter


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


if __name__ == "__main__":
    unittest.main()
