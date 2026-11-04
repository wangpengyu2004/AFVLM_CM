"""Full downloaded LoRA context with compact trainable factors.

The intermediate activation contains selected and frozen coordinates. PEFT's
existing dropout is applied once before A, and scaling once after B. Frozen
weights are detached; gradients through their input remain enabled.
"""
from __future__ import annotations

from collections.abc import Callable
from types import MethodType
from typing import Any


def attach_background(a: Any, b: Any, frozen_a: Any, frozen_b: Any,
                      gate: Any = None, order: Any = None) -> Callable[[], None]:
    import torch
    import torch.nn.functional as F

    if a.bias is not None or b.bias is not None:
        raise ValueError("Frozen LoRA background requires bias-free factors")
    # Private snapshot, not a view of mutable global parameters or an optimizer parameter.
    fa, fb = frozen_a.detach().clone(), frozen_b.detach().clone()
    old_a, old_b = a.forward, b.forward

    def forward_a(this: Any, x: Any) -> Any:
        hidden = F.linear(x, torch.cat((this.weight, fa), dim=0))
        if gate is not None:
            # Gate is a fixed-one, non-Parameter leaf in original global order.
            # Multiplication sits between A and B and sees PEFT dropout once.
            hidden = hidden * gate.index_select(0, order).to(dtype=hidden.dtype)
        return hidden

    def forward_b(this: Any, x: Any) -> Any:
        return F.linear(x, torch.cat((this.weight, fb), dim=1))

    a.forward = MethodType(forward_a, a)
    b.forward = MethodType(forward_b, b)

    def restore() -> None:
        a.forward, b.forward = old_a, old_b

    return restore
