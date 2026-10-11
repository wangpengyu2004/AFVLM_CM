"""Historical DDP AST/dispatch checks only; never execute a model or training."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from afl_vlm.config import load_config
from afl_vlm.models.base import TrainConfig
from afl_vlm.models.llava15 import Llava15Adapter

ROOT = Path(__file__).resolve().parents[1]
# Generated from git show ced4779:code/afl_vlm/models/llava15.py.
# AST fingerprints ignore comments/formatting, NOT execution statements.
HISTORICAL_AST = {
    "local_train": "ff51effd965a27ed69d9e4e8abe4bd5b049562bfb39818a2098ce2953d30d642",
    "enable_distributed_data_parallel":
        "cf147c1c0623440b1ceb0bc12a3e1e99e7ff06728a0427ff58066954e08b146d",
    "load": "b7402675bfa3da4a317803f9b588782f982624e8a70e559131d4e8eb0a2dc10a",
    "_encode_batch": "0ec28fcb2a09553f60af58407af9aa4c401787fadb2a8d87917280f3964c15a3",
}


class _AlignedPathReached(Exception):
    pass


class HistoricalDDPTests(unittest.TestCase):
    def config(self, **changes) -> TrainConfig:
        return TrainConfig(
            **{
                "local_epochs": 1, "batch_size": 4, "gradient_accumulation": 4,
                "learning_rate": 2e-5, "max_text_length": 2048, "seed": 42,
                "distributed_world_size": 8, **changes,
            }
        )

    def test_original_training_loader_collator_and_wrapper_asts(self) -> None:
        tree = ast.parse((ROOT / "code/afl_vlm/models/llava15.py").read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        for name, expected in HISTORICAL_AST.items():
            current = methods["_local_train_historical_ddp" if name == "local_train" else name]
            current.name = name
            with self.subTest(function=name):
                digest = hashlib.sha256(ast.dump(current, include_attributes=False).encode())
                self.assertEqual(digest.hexdigest(), expected)

    def test_ddp_adamw_old_and_new_profiles_dispatch_historical_body(self) -> None:
        for scope in ("per_device", "reference_ddp"):
            adapter = Llava15Adapter()
            adapter._ddp_enabled = True
            config = self.config(batch_size_scope=scope)
            samples = object()
            result = object()
            with (
                patch.object(adapter, "_local_train_historical_ddp", return_value=result) as old,
                patch.object(adapter, "_imports", side_effect=AssertionError("aligned path")),
                patch("afl_vlm.models.llava15.optimizer_windows",
                      side_effect=AssertionError("shared sampler")),
            ):
                self.assertIs(adapter.local_train(samples, config, "vqa"), result)
                old.assert_called_once_with(samples, config, "vqa")

    def test_ours_opcm_templates_dispatch_historical_body(self) -> None:
        for method in ("ours", "opcm_lora"):
            config = load_config(
                ROOT / f"configs/experiments/llava/afvlm_cm/2clients/{method}.yaml"
            )
            adapter = Llava15Adapter()
            adapter._ddp_enabled = True
            training = config["training"]
            client_config = self.config(optimizer=training["optimizer"])
            with patch.object(adapter, "_local_train_historical_ddp") as old:
                adapter.local_train([], client_config, "vqa")
                old.assert_called_once()

    def test_single_gpu_sgd_probe_and_private_adapter_keep_aligned_path(self) -> None:
        # Stop at the import boundary; no model/optimizer/forward is constructed.
        for ddp, optimizer, probe, private in (
            (False, "adamw", False, False),
            (True, "sgd", False, False),
            (True, "adamw", True, False),
            (True, "adamw", False, True),
        ):
            adapter = Llava15Adapter()
            adapter._ddp_enabled = ddp
            adapter._pilot_connector = object() if private else None
            config = self.config(optimizer=optimizer, gradient_estimation_only=probe)
            with (
                patch.object(adapter, "_imports", side_effect=_AlignedPathReached),
                patch.object(adapter, "_local_train_historical_ddp") as old,
                self.assertRaises(_AlignedPathReached),
            ):
                adapter.local_train([], config, "vqa")
            old.assert_not_called()

    def test_ddp_constructor_restores_original_options(self) -> None:
        adapter = Llava15Adapter()
        adapter._config["gradient_checkpointing"] = True
        constructor = Mock()
        torch = SimpleNamespace(
            distributed=SimpleNamespace(is_initialized=lambda: True),
            nn=SimpleNamespace(parallel=SimpleNamespace(DistributedDataParallel=constructor)),
        )
        with patch.object(adapter, "_imports", return_value=(torch,)):
            adapter.enable_distributed_data_parallel(3)
        self.assertEqual(
            constructor.call_args.kwargs,
            dict(device_ids=[3], output_device=3, broadcast_buffers=False,
                 find_unused_parameters=False),
        )

    def test_incompatible_reference_is_rejected_before_historical_training(self) -> None:
        adapter = Llava15Adapter()
        adapter._ddp_enabled = True
        config = self.config(batch_size_scope="reference_ddp", reference_world_size=4)
        with patch.object(adapter, "_local_train_historical_ddp") as old:
            with self.assertRaises(ValueError):
                adapter.local_train([], config, "vqa")
            old.assert_not_called()


if __name__ == "__main__":
    unittest.main()
