"""Storage/optimizer invariants only; no VLM, CUDA context or training job."""

from __future__ import annotations

import copy
import unittest
from unittest.mock import patch

import torch
from afl_vlm.methods.ours_router.conflict import TaskConflictWindow
from afl_vlm.methods.ours_router.router import PrivateRouter


class RouterMemoryChecks(unittest.TestCase):
    def test_releasing_consumed_gradients_preserves_weights_moments_and_rng(self):
        router = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        reference = PrivateRouter([(2, 2)], torch.device("cpu"), 42, {})
        features = torch.arange(18.).reshape(3, 6) / 18
        labels = torch.tensor([-2., 1., 3.])
        observations = {"m": ("d2_2", features, labels)}
        original_zero = reference.optimizer.zero_grad
        calls = 0

        def historical_zero(*args, **kwargs):
            nonlocal calls
            calls += 1
            # The old implementation only cleared at the START of learn().
            if calls % 2:
                original_zero(*args, **kwargs)

        for _ in range(3):
            rng = torch.get_rng_state().clone()
            report = router.learn(observations)
            with patch.object(reference.optimizer, "zero_grad", side_effect=historical_zero):
                expected = reference.learn(observations)
            self.assertEqual(report, expected)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertTrue(all(p.grad is None for p in router.heads.parameters()))
            self.assertTrue(any(p.grad is not None for p in reference.heads.parameters()))
            for name, value in router.heads.state_dict().items():
                self.assertTrue(torch.equal(value, reference.heads.state_dict()[name]))
            for key, state in router.optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(
                        value, reference.optimizer.state_dict()["state"][key][name]
                    ))
        router.learn({"m": ("d2_2", features, torch.zeros_like(labels))})
        self.assertTrue(all(p.grad is None for p in router.heads.parameters()))

    def test_exact_factor_storage_and_checkpoint_roundtrip_keep_values_and_scores(self):
        from test_ours_router_conflict import module_map

        generator = torch.Generator().manual_seed(19)
        window = TaskConflictWindow(module_map(), ["a", "b"])
        for n, task in enumerate(("a", "b", "a", "b"), 1):
            before = {"a": torch.randn(3, 2, generator=generator),
                      "b": torch.randn(2, 3, generator=generator)}
            after = {name: value + .1 for name, value in before.items()}
            old_state = copy.deepcopy(window.state_dict())
            old = window
            window = window.append(before, after, {"m": [0, 2]}, task, str(n), n)
            self.assertEqual(old.clock, n - 1)
            self.assertEqual(len(old.rows), len(old_state["rows"]))
            for previous, unchanged in zip(old_state["rows"], old.rows, strict=True):
                for side in ("left", "right"):
                    self.assertTrue(torch.equal(previous["modules"]["m"][side],
                                                unchanged["modules"]["m"][side]))
            for row in window.rows:
                for side in ("left", "right"):
                    factor = row["modules"]["m"][side]
                    self.assertEqual(factor.device.type, "cpu")
                    self.assertEqual(factor.dtype, torch.float32)
                    self.assertFalse(factor.requires_grad)
            window.to(torch.device("cpu"))
            self.assertEqual(window.summary()["factor_storage_device"], "cpu")
            self.assertEqual(window.summary()["gram_storage_device"], "cpu")
        restored = TaskConflictWindow(module_map(), ["a", "b"])
        restored.load_state_dict(window.state_dict(), device=torch.device("cpu"))
        self.assertEqual(window.scores("a"), restored.scores("a"))
        self.assertTrue(torch.equal(window.grams["m"], restored.grams["m"]))
        for old, new in zip(window.rows, restored.rows, strict=True):
            for side in ("left", "right"):
                self.assertTrue(torch.equal(old["modules"]["m"][side],
                                            new["modules"]["m"][side]))


if __name__ == "__main__":
    unittest.main()
