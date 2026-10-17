"""CA2FL, ICLR 2024: Algorithm 2 / Eq. (4.1), and MF variant Appendix A.1.

Tackling the Data Heterogeneity in Asynchronous Federated Learning with Cached
Update Calibration. Primary source and adaptation details: docs/ca2fl.md.
Operate only on federated trainable A/B coordinates, using the common local
optimizer, virtual Plan and global evaluation. No extra client computation.
The paper leaves the unbiased MF quantizer unspecified; ours is seeded stochastic
uniform affine quantization, with genuinely packed 4-bit/8-bit cache storage.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping
from typing import Any

from afl_vlm.federation.types import MethodCapabilities, ServerContext, ServerMutation, Update
from afl_vlm.methods.base import Method
from afl_vlm.methods.registry import register_method
from afl_vlm.models.base import LoRAState


@register_method("ca2fl")
class CA2FL(Method):
    name = "ca2fl"
    capabilities = MethodCapabilities(mode="asynchronous", requires_buffer=True)
    allowed_params = {"server_lr", "buffer_size", "flush_last_buffer", "cache_bits", "cache_seed"}

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        self.server_lr = float(self.params.get("server_lr", 1.0))
        if not math.isfinite(self.server_lr) or self.server_lr <= 0:
            raise ValueError("CA2FL server_lr must be finite and positive")
        self.buffer_size = self.params.get("buffer_size", 3)
        if (
            isinstance(self.buffer_size, bool)
            or not isinstance(self.buffer_size, int)
            or self.buffer_size <= 0
        ):
            raise ValueError("CA2FL buffer_size must be a positive integer")
        self.cache_bits = self.params.get("cache_bits", 32)
        if type(self.cache_bits) is not int or self.cache_bits not in {4, 8, 32}:
            raise ValueError("CA2FL cache_bits must be 32, 8 or 4")
        self.cache_seed = self.params.get("cache_seed", 42)
        if type(self.cache_seed) is not int or self.cache_seed < 0:
            raise ValueError("CA2FL cache_seed must be a non-negative integer")
        if not isinstance(self.params.get("flush_last_buffer", True), bool):
            raise ValueError("CA2FL flush_last_buffer must be boolean")
        self.layout: tuple[tuple[str, tuple[int, ...]], ...] = ()
        self.client_ids: tuple[str, ...] = ()
        self.client_cache: dict[str, Any] = {}
        self.client_rounds: dict[str, int] = {}
        self.seen_update_ids: set[str] = set()
        self.cache_sum: Any = None
        self.global_cache_mean: Any = None
        self.correction_sum: Any = None
        self.buffer: list[dict[str, Any]] = []
        self.buffer_aggregations = 0
        self.occupancy_sum = 0
        self.arrivals = 0

    def configure_server(self, initial_state: LoRAState, clients: list[Any]) -> None:
        """Algorithm 2 line 1: zero cache for ALL clients, including unseen ones."""
        vector = self._vector(initial_state)
        ids = tuple(sorted(str(client.id) for client in clients))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("CA2FL requires distinct configured clients")
        if self.client_ids and self.client_ids != ids:
            raise ValueError("CA2FL configured client population changed")
        self.client_ids = ids
        if self.cache_sum is None:
            self.cache_sum = vector.new_zeros(vector.shape)
            self.global_cache_mean = vector.new_zeros(vector.shape)
            self.correction_sum = vector.new_zeros(vector.shape)

    def _vector(self, state: Mapping[str, Any], device: Any = None) -> Any:
        import torch

        if not state:
            raise ValueError("CA2FL requires non-empty federated trainable state")
        if device is None:
            device = next(
                (value.device for value in state.values() if isinstance(value, torch.Tensor)),
                torch.device("cpu"),
            )
        values = [
            torch.as_tensor(state[name], device=device).detach().float() for name in sorted(state)
        ]
        layout = tuple(
            (name, tuple(value.shape)) for name, value in zip(sorted(state), values, strict=True)
        )
        if self.layout and layout != self.layout:
            raise ValueError("CA2FL federated parameter names/shapes changed")
        result = torch.cat([value.reshape(-1) for value in values])
        if result.numel() == 0:
            raise ValueError("CA2FL requires non-empty federated trainable tensors")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("CA2FL received non-finite federated state")
        if not self.layout:
            self.layout = layout
        return result

    def _state(self, vector: Any, reference: LoRAState) -> LoRAState:
        import torch

        state, offset = {}, 0
        for name, shape in self.layout:
            size = math.prod(shape)
            value = vector[offset : offset + size].reshape(shape).clone()
            ref = reference[name]
            state[name] = value.to(ref) if isinstance(ref, torch.Tensor) else value
            offset += size
        return state

    def _encode(self, vector: Any, update_id: str) -> Any:
        """Appendix A.1: an unbiased Q(delta), isolated from training RNG."""
        if self.cache_bits == 32:
            return vector.clone()
        import torch

        levels = (1 << self.cache_bits) - 1
        lower, upper = vector.min(), vector.max()
        step = (upper - lower) / levels
        scaled = ((vector - lower) / step.clamp_min(torch.finfo(vector.dtype).tiny)).clamp(
            0, levels
        )
        floor = scaled.floor()
        seed = int.from_bytes(
            hashlib.sha256(f"{self.cache_seed}:{update_id}".encode()).digest()[:8], "big"
        ) % (2**63 - 1)
        generator = torch.Generator(device=vector.device).manual_seed(seed)
        random = torch.rand(vector.shape, device=vector.device, generator=generator)
        codes = (floor + (random < scaled - floor)).to(torch.uint8)
        if self.cache_bits == 4:
            if codes.numel() % 2:
                codes = torch.cat((codes, codes.new_zeros(1)))
            codes = codes[0::2] | (codes[1::2] << 4)
        return {
            "codes": codes,
            "lower": lower.clone(),
            "step": step.clone(),
            "size": vector.numel(),
        }

    def _decode(self, encoded: Any, reference: Any) -> Any:
        if self.cache_bits == 32:
            return encoded.to(reference)
        import torch

        codes = encoded["codes"].to(device=reference.device)
        if self.cache_bits == 4:
            codes = torch.stack((codes & 15, codes >> 4), dim=1).reshape(-1)
        return (
            encoded["lower"].to(reference)
            + encoded["step"].to(reference) * codes[: encoded["size"]].float()
        )

    @staticmethod
    def _norm(vector: Any) -> float:
        return float(vector.norm().item())

    @staticmethod
    def _storage_bytes(value: Any) -> int:
        if isinstance(value, Mapping):
            return sum(CA2FL._storage_bytes(item) for item in value.values())
        return value.numel() * value.element_size() if hasattr(value, "numel") else 0

    def on_arrival(self, update: Update, server_context: ServerContext) -> list[ServerMutation]:
        """Algorithm 2 lines 4--6: subtract old cache BEFORE replacing it."""
        import torch

        if not self.client_ids or self.cache_sum is None:
            raise RuntimeError("CA2FL requires configure_server before arrivals")
        if server_context.expected_clients != len(self.client_ids):
            raise ValueError("CA2FL server population differs from configured clients")
        if update.client_id not in self.client_ids:
            raise ValueError("CA2FL received an unknown client")
        if update.update_id in self.seen_update_ids:
            raise ValueError("CA2FL received a duplicate update")
        if update.local_round < 0 or update.local_round <= self.client_rounds.get(
            update.client_id, -1
        ):
            raise ValueError("CA2FL client local_round must strictly increase")
        if not 0 <= update.base_version <= server_context.version:
            raise ValueError("CA2FL update base_version is invalid")
        current = self._vector(server_context.global_state)
        raw = self._vector(update.delta, current.device)
        old = self.client_cache.get(update.client_id)
        previous = current.new_zeros(current.shape) if old is None else self._decode(old, current)
        correction = raw - previous
        encoded = self._encode(raw, update.update_id)
        cached = self._decode(encoded, current)
        with torch.no_grad():
            self.cache_sum = self.cache_sum.to(current) + cached - previous
            self.correction_sum = self.correction_sum.to(current) + correction
            # global_cache_mean intentionally remains the PRE-buffer snapshot.
            self.client_cache[update.client_id] = encoded
        self.buffer.append(
            {
                "update_id": update.update_id,
                "client_id": update.client_id,
                "arrival_time": update.arrival_time,
                "base_version": update.base_version,
                "receive_version": server_context.version,
                "staleness": server_context.version - update.base_version,
                "cache_hit": old is not None,
                "raw_delta_norm": self._norm(raw),
                "previous_cache_norm": self._norm(previous),
                "correction_norm": self._norm(correction),
                "quantization_error_norm": self._norm(cached - raw),
            }
        )
        self.seen_update_ids.add(update.update_id)
        self.client_rounds[update.client_id] = update.local_round
        self.arrivals += 1
        self.occupancy_sum += len(self.buffer)
        if len(self.buffer) >= self.buffer_size:
            return [self._flush(server_context, update.arrival_time)]
        return []

    def _flush(self, context: ServerContext, flush_time: float) -> ServerMutation:
        """Eq. 4.1; only AFTER forming v_t refresh h_{t+1} (line 14)."""
        import torch

        rows = self.buffer
        distinct = len({row["client_id"] for row in rows})
        current = self._vector(context.global_state)
        with torch.no_grad():
            old_mean = self.global_cache_mean.to(current)
            direction = old_mean + self.correction_sum.to(current) / distinct
            new_vector = current + self.server_lr * direction
            if not bool(torch.isfinite(new_vector).all()):
                raise ValueError("CA2FL calibrated server update is non-finite")
            self.global_cache_mean = self.cache_sum.to(current) / len(self.client_ids)
            self.correction_sum = current.new_zeros(current.shape)
        self.buffer = []
        self.buffer_aggregations += 1
        return ServerMutation(
            self._state(new_vector, context.global_state),
            self.server_lr,
            [row["update_id"] for row in rows],
            metadata={
                "aggregation_rule": "ca2fl_cached_update_calibration",
                "buffer_occupancy": len(rows),
                "buffer_distinct_clients": distinct,
                "buffer_aggregation": self.buffer_aggregations,
                "mean_buffer_waiting_time": sum(flush_time - r["arrival_time"] for r in rows)
                / len(rows),
                "server_lr": self.server_lr,
                "cache_bits": self.cache_bits,
                "cache_population": len(self.client_ids),
                "cached_clients": len(self.client_cache),
                "cache_storage_bytes": sum(
                    self._storage_bytes(c) for c in self.client_cache.values()
                ),
                "previous_global_cache_norm": self._norm(old_mean),
                "global_cache_norm_after": self._norm(self.global_cache_mean),
                "calibrated_delta_norm": self._norm(direction),
                "residual_flush": len(rows) < self.buffer_size,
                "ca2fl_diagnostics": {row["update_id"]: dict(row) for row in rows},
            },
        )

    def on_finish(self, server_context: ServerContext) -> list[ServerMutation]:
        if not self.buffer:
            return []
        if not self.params.get("flush_last_buffer", True):
            raise RuntimeError("CA2FL residual updates remain while flush_last_buffer=false")
        return [self._flush(server_context, max(row["arrival_time"] for row in self.buffer))]

    def state_dict(self) -> dict[str, Any]:
        return copy.deepcopy({"schema_version": 1, **self.__dict__})

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported CA2FL method-state schema")
        restored = type(self)(state["params"])
        keys = set(restored.__dict__)
        if keys != set(state) - {"schema_version"}:
            raise ValueError("CA2FL method-state fields do not match schema")
        payload = copy.deepcopy({key: state[key] for key in keys})
        for key in ("server_lr", "buffer_size", "cache_bits", "cache_seed"):
            if payload[key] != getattr(restored, key):
                raise ValueError(f"CA2FL method-state {key} differs from params")
        if not set(payload["client_cache"]) <= set(payload["client_ids"]):
            raise ValueError("CA2FL cache contains unknown clients")
        self.__dict__.update(payload)
