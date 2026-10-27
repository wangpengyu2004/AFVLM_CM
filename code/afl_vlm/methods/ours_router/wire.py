"""Unweighted selected factor deltas; reuse the existing indexed wire schema.

The hc_* keys are legacy schema identifiers used by the tensor validator, not
Hard Concrete gates. This method never constructs or folds gate amplitudes.
"""

from __future__ import annotations

import json
from typing import Any

from afl_vlm.methods.ours_hc.payload import select, subtract


def pack_indexed(update: Any) -> Any:
    if "hc_gate_values" in update.metadata:
        raise ValueError("Router uploads must not contain Hard Concrete amplitudes")
    if update.metadata.get("hc_payload_format"):
        raise ValueError("Indexed upload was already packed")
    compact, elements, full, indices = {}, 0, 0, 0
    for name, item in update.metadata["lora_metadata"].items():
        selected = update.metadata["hc_selection"][name]
        indices += len(selected)
        width = item["in_features"] + item["out_features"]
        elements += len(selected) * width
        full += item["rank"] * width
        for side, key in (("A", "a_name"), ("B", "b_name")):
            parameter = item[key]
            compact[parameter] = subtract(
                select(update.local_state[parameter], selected, side),
                select(update.base_state[parameter], selected, side),
            )
    update.delta, update.base_state, update.local_state = compact, {}, {}
    update.metadata.update(
        hc_payload_format="indexed_delta_v1",
        federated_parameter_scope="indexed_language_lora_deltas",
        selection_algorithm="private_router_fixed_topk",
    )
    update.metadata["hc_communication"] = {
        "selected_components": indices,
        "index_elements": indices,
        "factor_elements": elements,
        "full_factor_elements": full,
        "factor_element_fraction": elements / full,
        "factor_tensor_bytes": sum(v.numel() * v.element_size() for v in compact.values())
        if all(hasattr(v, "numel") for v in compact.values())
        else 4 * elements,
        "includes_backbone": False,
        "includes_base_or_endpoint": False,
        "includes_router_parameters": False,
        "includes_gate_parameters": False,
        "control_json_bytes": len(json.dumps(update.metadata).encode("utf-8")),
    }
    return update
