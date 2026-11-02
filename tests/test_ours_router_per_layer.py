"""Per-layer private Router topology/identity checks, without a VLM training run."""

from __future__ import annotations

import copy
import unittest

import torch
from afl_vlm.config import load_config
from afl_vlm.federation.types import ClientContext
from afl_vlm.methods.ours_router.local import LocalRouter
from afl_vlm.methods.ours_router.method import OursRouter
from afl_vlm.methods.ours_router.router import PrivateRouter
from afl_vlm.models.base import clone_state


class PerLayerRouterChecks(unittest.TestCase):
    def setUp(self):
        from test_ours_router import FactorRegistry

        self.fixture = FactorRegistry()
        self.controller = LocalRouter(self.fixture, {"router_sharing": "layer"})
        self.metadata = self.controller.full_metadata

    def name(self, layer, projection="q_proj"):
        return next(m for m in self.metadata if f"layers.{layer}.self_attn.{projection}" in m)

    def test_exact_original_topology_one_head_per_layer_same_layer_qkvo_share(self):
        router = self.controller.router("a/client_0", 42)
        self.assertEqual(len(router.heads), 2)
        for layer in (0, 1):
            heads = {router.module_heads[self.name(layer, p)] for p in
                     ("q_proj", "k_proj", "v_proj", "o_proj")}
            self.assertEqual(len(heads), 1)
            network = router.heads[next(iter(heads))]
            self.assertEqual([type(m).__name__ for m in network], ["Linear", "SiLU", "Linear"])
            self.assertEqual((network[0].in_features, network[0].out_features), (6, 32))
            self.assertEqual((network[2].in_features, network[2].out_features), (32, 1))
        self.assertNotEqual(router.module_heads[self.name(0)], router.module_heads[self.name(1)])
        self.assertTrue(all(p.dtype == torch.float32 for p in router.heads.parameters()))

    def test_scoring_uses_real_layer_identity_even_for_identical_features(self):
        router = self.controller.router("a/client_0", 42)
        with torch.no_grad():
            for parameter in router.heads.parameters():
                parameter.zero_()
            for layer in (0, 1):
                router.heads[router.module_heads[self.name(layer)]][-1].bias.fill_(layer + 1.)
        a, b = torch.ones(3, 2), torch.ones(2, 3)
        for layer in (0, 1):
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                self.assertEqual(router.predict(a, b, self.name(layer, projection)).tolist(),
                                 [layer + 1.] * 3)
        with self.assertRaises(ValueError):
            router.predict(a, b)
        with self.assertRaises(ValueError):
            router.predict(a, b, "unregistered.layers.0.q_proj")

    def test_clients_and_layers_have_no_parameter_aliases_and_preserve_main_rng(self):
        rng = torch.get_rng_state().clone()
        a, b = self.controller.router("a/client_0", 42), self.controller.router("a/client_1", 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue({id(p) for p in a.heads.parameters()}.isdisjoint(
            {id(p) for p in b.heads.parameters()}
        ))
        heads = list(a.heads.values())
        self.assertTrue({id(p) for p in heads[0].parameters()}.isdisjoint(
            {id(p) for p in heads[1].parameters()}
        ))
        self.assertFalse(torch.equal(heads[0][0].weight, heads[1][0].weight))
        self.assertTrue({id(p) for p in a.heads.parameters()}.isdisjoint(
            {id(p) for _, p in self.fixture.named_federated_parameters()}
        ))

    def test_probe_and_compact_observations_use_same_layer_heads(self):
        router = self.controller.router("a/client_0", 42)
        self.controller.probing = True
        for _, parameter in self.fixture.named_federated_parameters():
            parameter.grad = torch.ones_like(parameter)
        self.controller.observe(0)
        probe_keys = {m: item[0] for m, item in self.controller.pending.items()}
        self.assertEqual(probe_keys, router.module_heads)
        self.controller.probing = False
        self.controller.capacities["a/client_0"] = {m: {"rank": 2} for m in self.metadata}
        context = ClientContext("a/client_0", "a", "a", 128, 0, 0, 1., 42)
        selected = {m: [0, 2] for m in self.metadata}
        self.controller.prepare(context, {"indices": selected, "source": {}})
        for _, parameter in self.fixture.named_federated_parameters():
            parameter.grad = torch.ones_like(parameter)
        self.controller.observe(0)
        self.assertEqual({m: item[0] for m, item in self.controller.pending.items()}, probe_keys)
        self.assertTrue(all(item[1].shape[0] == 2 for item in self.controller.pending.values()))
        self.controller.restore()

    def test_supervision_of_one_layer_cannot_update_another_layer_or_lora(self):
        router = self.controller.router("a/client_0", 42)
        weights = clone_state(router.heads.state_dict())
        lora = self.fixture.snapshot_trainable()
        module = self.name(0)
        key = router.module_heads[module]
        router.learn({module: (key, torch.arange(18.).reshape(3, 6) / 18,
                              torch.tensor([-1., 2., 3.]))})
        other = router.module_heads[self.name(1)] + "."
        self.assertTrue(all(torch.equal(value, router.heads.state_dict()[name])
                            for name, value in weights.items() if name.startswith(other)))
        self.assertTrue(any(not torch.equal(value, router.heads.state_dict()[name])
                            for name, value in weights.items() if name.startswith(key + ".")))
        self.assertTrue(all(torch.equal(value, self.fixture.snapshot_trainable()[name])
                            for name, value in lora.items()))
        with self.assertRaisesRegex(ValueError, "wrong Transformer layer"):
            router.learn({module: (router.module_heads[self.name(1)], torch.ones(3, 6),
                                  torch.ones(3))})

    def test_checkpoint_roundtrip_and_explicit_rejection_of_cross_sharing_loads(self):
        router = self.controller.router("a/client_0", 42)
        saved = copy.deepcopy(router.state_dict())
        restored = PrivateRouter([(2, 2)], torch.device("cpu"), 7,
                                 {"router_sharing": "layer"}, self.metadata)
        restored.load_state_dict(saved)
        for name, value in router.heads.state_dict().items():
            self.assertTrue(torch.equal(value, restored.heads.state_dict()[name]))
        dimension = PrivateRouter([(2, 2)], torch.device("cpu"), 7, {})
        with self.assertRaisesRegex(ValueError, "layer layout differs"):
            restored.load_state_dict(dimension.state_dict())
        with self.assertRaisesRegex(ValueError, "per-layer private Router"):
            dimension.load_state_dict(saved)
        legacy = dimension.state_dict()
        self.assertNotIn("router_layout", legacy)
        dimension.load_state_dict(legacy)

    def test_configuration_defaults_do_not_migrate_old_profiles_or_accept_bad_layouts(self):
        from test_ours_router import ROOT

        for setting in (2, 5, 10):
            config = load_config(ROOT / f"configs/experiments/llava/afvlm_cm/"
                                 f"{setting}clients/ours_router.yaml")
            self.assertEqual(config["method"]["params"]["router_sharing"], "layer")
            self.assertEqual(config["method"]["params"]["router_hidden_size"], 32)
            OursRouter(config["method"]["params"]).validate_runtime()
        old = LocalRouter(self.fixture, {}).router("legacy/client_0", 42)
        self.assertEqual(old.sharing, "dimension")
        self.assertEqual(len(old.heads), 1)
        with self.assertRaises(ValueError):
            OursRouter({"router_sharing": "module"})
        with self.assertRaises(ValueError):
            PrivateRouter([(2, 2)], torch.device("cpu"), 42, {"router_sharing": "layer"})
        bad = copy.deepcopy(self.metadata)
        bad[self.name(0, "k_proj")]["out_features"] = 3
        with self.assertRaisesRegex(ValueError, "equal feature dimensions"):
            PrivateRouter([(2, 2), (2, 3)], torch.device("cpu"), 42,
                          {"router_sharing": "layer"}, bad)


if __name__ == "__main__":
    unittest.main()
