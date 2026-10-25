"""Sparse factor payloads keep original rank indices; no dense BA or SVD."""

from __future__ import annotations

import json
import math
from typing import Any

from afl_vlm.methods.ours import lora_functional_distance_sq
from afl_vlm.models.base import clone_state


def select(value: Any, indices: list[int], side: str) -> Any:
    if hasattr(value, "index_select"):
        import torch

        index = torch.tensor(indices, dtype=torch.long, device=value.device)
        return value.index_select(0 if side == "A" else 1, index)
    return (
        [list(value[index]) for index in indices]
        if side == "A"
        else [[row[index] for index in indices] for row in value]
    )


def subtract(left: Any, right: Any) -> Any:
    if hasattr(left, "detach"):
        return left - right
    return [[a - b for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def add(left: Any, right: Any) -> Any:
    if hasattr(left, "detach"):
        return left + right
    return [[a + b for a, b in zip(x, y, strict=True)] for x, y in zip(left, right, strict=True)]


def fold_b(value: Any, gates: list[float]) -> Any:
    if hasattr(value, "detach"):
        return value * value.new_tensor(gates).unsqueeze(0)
    return [[item * gate for item, gate in zip(row, gates, strict=True)] for row in value]


def finite_state(state: dict) -> bool:
    if all(hasattr(value, "isfinite") for value in state.values()):
        import torch

        return bool(torch.stack([value.isfinite().all() for value in state.values()]).all().item())

    def finite(value: Any) -> bool:
        return (
            all(finite(item) for item in value)
            if isinstance(value, list)
            else (math.isfinite(float(value)))
        )

    return all(finite(value) for value in state.values())


def pack(update: Any) -> Any:
    if update.metadata.get("hc_payload_format") == "indexed_delta_v1":
        raise ValueError("Hard Concrete upload was already packed")
    compact, components, dense_elements = {}, 0, 0
    for module, metadata in update.metadata["lora_metadata"].items():
        indices = update.metadata["hc_selection"][module]
        gates = [update.metadata["hc_gate_values"][module][index] for index in indices]
        for side, key in (("A", "a_name"), ("B", "b_name")):
            name = metadata[key]
            local = select(update.local_state[name], indices, side)
            if side == "B":
                local = fold_b(local, gates)
            compact[name] = subtract(local, select(update.base_state[name], indices, side))
        components += len(indices)
        dense_elements += metadata["rank"] * (metadata["in_features"] + metadata["out_features"])
    elements = sum(
        len(update.metadata["hc_selection"][module]) * (item["in_features"] + item["out_features"])
        for module, item in update.metadata["lora_metadata"].items()
    )
    update.delta = compact
    # Immutable download snapshots belong to the server, never the upload.
    update.base_state, update.local_state = {}, {}
    # Gate amplitudes have already been folded into selected B. Keep individual
    # gates/probabilities private; only numerical module summaries travel for logs.
    del update.metadata["hc_gate_values"]
    update.metadata["federated_parameter_scope"] = "indexed_language_lora_deltas"
    update.metadata["hc_payload_format"] = "indexed_delta_v1"
    update.metadata["hc_communication"] = {
        "selected_components": components,
        "factor_elements": elements,
        "full_factor_elements": dense_elements,
        "factor_element_fraction": elements / dense_elements,
        "index_elements": components,
        "includes_backbone": False,
        "includes_gate_parameters": False,
        "includes_base_or_endpoint": False,
        "factor_tensor_bytes": sum(
            value.numel() * value.element_size() for value in compact.values()
        )
        if all(hasattr(value, "numel") for value in compact.values())
        else 4 * elements,
        "control_json_bytes": len(
            json.dumps(
                {key: value for key, value in update.metadata.items() if key != "hc_communication"}
            ).encode("utf-8")
        ),
    }
    return update


def validate(update: Any, metadata: dict[str, dict]) -> None:
    from afl_vlm.methods.ours import _shape

    if (
        update.metadata.get("hc_payload_format") != "indexed_delta_v1"
        or update.base_state
        or update.local_state
        or "hc_gate_values" in update.metadata
    ):
        raise ValueError("Expected a compact indexed delta without base/endpoint tensors")
    client_metadata = update.metadata.get("lora_metadata", {})
    selection = update.metadata.get("hc_selection", {})
    sensitivity = update.metadata.get("module_sensitivity", {})
    if set(client_metadata) != set(metadata) or set(selection) != set(metadata):
        raise ValueError("Hard Concrete client/server module names differ")
    if set(sensitivity) != set(metadata):
        raise ValueError("Hard Concrete sensitivity must be module-level")
    expected_keys = set()
    for module, item in metadata.items():
        actual = client_metadata[module]
        for field in ("module_name", "a_name", "b_name", "rank", "in_features", "out_features"):
            if item[field] != actual[field]:
                raise ValueError(f"LoRA metadata changed: {module}.{field}")
        scaling = float(actual["scaling"])
        if not math.isfinite(scaling) or scaling <= 0:
            raise ValueError("Invalid LoRA scaling")
        if item["scaling"] is not None and float(item["scaling"]) != scaling:
            raise ValueError("LoRA scaling differs between clients")
        indices = selection[module]
        if (
            not isinstance(indices, list)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < item["rank"]
                for index in indices
            )
            or indices != sorted(set(indices))
        ):
            raise ValueError("Selected global rank indices must be unique, ordered and in range")
        value = float(sensitivity[module])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid module sensitivity")
        for side, field in (("A", "a_name"), ("B", "b_name")):
            name = item[field]
            expected_keys.add(name)
            if name not in update.delta:
                raise ValueError("Missing selected factor delta")
            shape = _shape(update.delta[name])
            expected = (
                (len(indices), item["in_features"])
                if side == "A"
                else (item["out_features"], len(indices))
            )
            if shape != expected and not (side == "A" and not indices and shape == (0,)):
                raise ValueError(f"Invalid compact delta shape for {name}: {shape}, {expected}")
    if set(update.delta) != expected_keys:
        raise ValueError("Upload contains parameters outside the indexed LoRA payload")
    if not finite_state(update.delta):
        raise ValueError("Non-finite selected factor delta")


def functional_energies(base: dict, current: dict, update: Any) -> tuple[float, float]:
    drift, local = 0.0, 0.0
    for module, item in update.metadata["lora_metadata"].items():
        indices = update.metadata["hc_selection"][module]
        if not indices:
            continue
        a_name, b_name = item["a_name"], item["b_name"]
        a0, b0 = select(base[a_name], indices, "A"), select(base[b_name], indices, "B")
        at, bt = select(current[a_name], indices, "A"), select(current[b_name], indices, "B")
        ak, bk = add(a0, update.delta[a_name]), add(b0, update.delta[b_name])
        c, scale = float(update.metadata["module_sensitivity"][module]), float(item["scaling"])
        # Same selected support in ALL three functions; no dimensional normalization.
        drift = drift + c * lora_functional_distance_sq(at, bt, a0, b0, scale)
        local = local + c * lora_functional_distance_sq(ak, bk, a0, b0, scale)
    drift, local = float(drift), float(local)
    return drift, local


def fuse(current: dict, update: Any, alphas: dict[str, float]) -> dict:
    result = clone_state(current)
    for module, item in update.metadata["lora_metadata"].items():
        indices = update.metadata["hc_selection"][module]
        if not indices:
            continue
        for side, key in (("A", "a_name"), ("B", "b_name")):
            name, alpha = item[key], alphas[module]
            if hasattr(result[name], "index_add_"):
                import torch

                index = torch.tensor(indices, device=result[name].device, dtype=torch.long)
                result[name].index_add_(
                    0 if side == "A" else 1, index, update.delta[name].to(result[name]) * alpha
                )
            elif side == "A":
                for offset, index in enumerate(indices):
                    result[name][index] = [
                        x + alpha * y
                        for x, y in zip(
                            result[name][index], update.delta[name][offset], strict=True
                        )
                    ]
            else:
                for row in range(item["out_features"]):
                    for offset, index in enumerate(indices):
                        result[name][row][index] += alpha * update.delta[name][row][offset]
    return result
