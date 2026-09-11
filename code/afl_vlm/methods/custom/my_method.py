"""Deliberately disabled custom-method template."""

from __future__ import annotations

from typing import Any

from afl_vlm.federation.types import ServerContext, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method


@register_method("my_method")
class MyMethodTemplate(Method):
    name = "my_method"
    allowed_params = {"download_correction", "upload_correction"}

    def __init__(self, params=None) -> None:
        super().__init__(params)
        raise NotImplementedError(
            "'my_method' is a template, not a research method. Implement its hooks and remove "
            "this guard before enabling it."
        )

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[Any]:
        raise NotImplementedError("Implement my_method.on_arrival before use")
