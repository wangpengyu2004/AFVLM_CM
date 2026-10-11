"""FedAsync proximal arithmetic/graph-inspection checks, no training or backward."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from afl_vlm.methods.registry import create_method


class FedAsyncProximalTests(unittest.TestCase):
    def setup_state(self, dtype=torch.float32):
        parameter = torch.nn.Parameter(torch.tensor([3.0, -2.0], dtype=dtype))
        reference = torch.tensor([1.0, -1.0], dtype=dtype)
        model = SimpleNamespace(named_federated_parameters=lambda: [("lora", parameter)])
        context = {"base_state": {"lora": reference}}
        method = create_method("fedasync", {"proximal_mode": "paper_loss", "proximal_mu": 0.1})
        return method, model, parameter, reference, context

    def test_numerical_loss_retains_penalty_without_lora_autograd_path(self) -> None:
        method, model, parameter, _, context = self.setup_state()
        base = torch.tensor(2.0, requires_grad=True)
        loss = method.local_loss(base, model, None, context)
        self.assertAlmostEqual(float(loss.detach()), 2.25)
        # Inspect graph leaves only. No loss.backward/autograd.grad/forward.
        pending, seen, leaves = [loss.grad_fn], set(), set()
        while pending:
            node = pending.pop()
            if node is None or node in seen:
                continue
            seen.add(node)
            if hasattr(node, "variable"):
                leaves.add(id(node.variable))
            pending.extend(child for child, _ in node.next_functions)
        self.assertEqual(leaves, {id(base)})
        self.assertNotIn(id(parameter), leaves)
        self.assertIsNone(parameter.grad)

    def test_exact_derivative_added_once_without_parameter_or_anchor_change(self) -> None:
        method, model, parameter, reference, context = self.setup_state()
        parameter.grad = torch.tensor([0.3, -0.7])
        before, anchor = parameter.detach().clone(), reference.clone()
        method.transform_gradients(model, context)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.5, -0.8]))
        torch.testing.assert_close(parameter.detach(), before)
        torch.testing.assert_close(reference, anchor)
        self.assertFalse(parameter.grad.requires_grad)

    def test_loss_observations_do_not_inject_regularizer_per_microbatch(self) -> None:
        method, model, parameter, _, context = self.setup_state()
        for accumulation in (1, 4, 32):
            parameter.grad = torch.tensor([0.3, -0.7])
            for _ in range(accumulation):
                method.local_loss(torch.tensor(2.0), model, None, context)
            torch.testing.assert_close(parameter.grad, torch.tensor([0.3, -0.7]))
            method.transform_gradients(model, context)
            torch.testing.assert_close(parameter.grad, torch.tensor([0.5, -0.8]))

    def test_no_extra_world_size_division(self) -> None:
        for world in (1, 8):
            method, model, parameter, _, context = self.setup_state()
            context["distributed_world_size"] = world
            parameter.grad = torch.tensor([0.3, -0.7])
            method.transform_gradients(model, context)
            torch.testing.assert_close(parameter.grad, torch.tensor([0.5, -0.8]))

    def test_missing_grad_gets_only_proximal_derivative(self) -> None:
        method, model, parameter, _, context = self.setup_state()
        method.transform_gradients(model, context)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.2, -0.1]))

    def test_fp16_dtype_and_fp32_anchor(self) -> None:
        method, model, parameter, _, context = self.setup_state(torch.float16)
        parameter.grad = torch.tensor([0.3, -0.7], dtype=torch.float16)
        method.local_loss(torch.tensor(2.0), model, None, context)
        method.transform_gradients(model, context)
        self.assertEqual(context["proximal_device_anchor"]["lora"].dtype, torch.float32)
        self.assertEqual(parameter.grad.dtype, torch.float16)
        torch.testing.assert_close(parameter.grad, torch.tensor([0.5, -0.8], dtype=torch.float16))

    def test_mu_zero_and_none_mode_preserve_gradient(self) -> None:
        for params in ({"proximal_mu": 0.0}, {"proximal_mode": "none"}):
            _, model, parameter, _, context = self.setup_state()
            method = create_method("fedasync", params)
            parameter.grad = torch.tensor([0.3, -0.7])
            base = torch.tensor(2.0, requires_grad=True)
            self.assertIs(method.local_loss(base, model, None, context), base)
            method.transform_gradients(model, context)
            torch.testing.assert_close(parameter.grad, torch.tensor([0.3, -0.7]))

    def test_ours_and_opcm_do_not_inject_proximal_gradient(self) -> None:
        for name in ("ours", "opcm_lora"):
            _, model, parameter, _, context = self.setup_state()
            method = create_method(name, {})
            parameter.grad = torch.tensor([0.3, -0.7])
            # Ours' diagnostics require its local context; its inherited loss,
            # like OPCM's, must simply return the original loss.
            base = torch.tensor(2.0, requires_grad=True)
            self.assertIs(method.local_loss(base, model, None, context), base)
            self.assertNotIn("proximal_device_anchor", context)


if __name__ == "__main__":
    unittest.main()
