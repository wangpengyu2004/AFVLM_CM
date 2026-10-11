"""Actual LoRA-factor hypernetwork (UniFed-LoRA-L, CVPRW 2026 specialization).
Four-layer shared MLP, learned semantic/depth/module embeddings, client-specific
shape heads. Local displacement supplies the hypernetwork vector-Jacobian
product; there is no scalar gate or norm-regression target.

Initialization is centered on the common PEFT state for comparable zero-B
initialization. Full output heads can be large; CPU/offloading is NOT silently
enabled. hyper_device must be chosen explicitly when GPU memory is insufficient.
"""

from __future__ import annotations

import math
import re
from typing import Any

from afl_vlm.methods.continual_merge.algebra import lora_pairs


class LoRAHypernetwork:
    def __init__(
        self,
        initial_state: dict[str, Any],
        clients: dict[str, str],
        descriptors: dict[str, list[float]],
        hidden_size: int,
        embedding_dim: int,
        learning_rate: float,
        seed: int,
        device: Any,
    ) -> None:
        import torch
        from torch import nn

        self.torch = torch
        self.device = torch.device(device)
        self.initial = {
            k: v.detach().to(device=self.device, dtype=torch.float32).clone()
            for k, v in initial_state.items()
        }
        self.pairs = lora_pairs(self.initial)
        self.clients = dict(clients)
        self.descriptors = dict(descriptors)
        self.module_info = {}
        projection_shapes = {}
        for module, (a, b) in self.pairs.items():
            layer = re.search(r"(?:^|\.)layers\.(\d+)\.", module)
            projection = next(
                (p for p in ("q_proj", "k_proj", "v_proj", "o_proj") if f".{p}:" in module), None
            )
            if layer is None or projection is None:
                raise ValueError(f"UniFed-LoRA requires layer/module metadata for {module}")
            shapes = (tuple(self.initial[a].shape), tuple(self.initial[b].shape))
            if projection in projection_shapes and projection_shapes[projection] != shapes:
                raise ValueError("Homogeneous UniFed-LoRA requires equal per-type output shapes")
            projection_shapes[projection] = shapes
            self.module_info[module] = (int(layer[1]), projection)
        self.projections = sorted(projection_shapes)
        max_layer = max(layer for layer, _ in self.module_info.values())
        descriptor_size = len(next(iter(descriptors.values())))
        if descriptor_size == 0 or any(len(v) != descriptor_size for v in descriptors.values()):
            raise ValueError("UniFed-LoRA task descriptors must have equal length")
        if any(not math.isfinite(float(x)) for v in descriptors.values() for x in v):
            raise ValueError("UniFed-LoRA task descriptors must be finite")
        if hidden_size <= 0 or embedding_dim <= 0:
            raise ValueError("UniFed-LoRA hidden/embedding dimensions must be positive")
        # Full client-specific output heads are intentionally not a scalar gate.
        # Account for parameter, gradient and two Adam moments before allocating.
        head_parameters = (
            len(clients)
            * (hidden_size + 1)
            * sum(
                sum(torch.tensor(shape).prod().item() for shape in shapes)
                for shapes in projection_shapes.values()
            )
        )
        self.estimated_head_bytes = int(head_parameters * 4 * 4)
        if self.device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            if self.estimated_head_bytes > free_bytes * 0.8:
                raise MemoryError(
                    "UniFed-LoRA full output heads need about "
                    f"{self.estimated_head_bytes / 2**30:.1f} GiB "
                    "including Adam/gradients, beyond available GPU memory. Use an explicitly "
                    "configured hyper_device: cpu or a larger GPU; no scalar fallback."
                )
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        # Initialization must not reset the experiment's dropout/RNG stream.
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            self.module = nn.ModuleDict(
                {
                    "task_embedding": nn.Linear(descriptor_size, embedding_dim, bias=False),
                    "modality_embedding": nn.Embedding(1, embedding_dim),
                    "layer_embedding": nn.Embedding(max_layer + 1, embedding_dim),
                    "projection_embedding": nn.Embedding(len(self.projections), embedding_dim),
                }
            )
            layers = []
            features = 4 * embedding_dim
            for _ in range(4):
                layers.extend((nn.Linear(features, hidden_size), nn.ReLU()))
                features = hidden_size
            self.module["trunk"] = nn.Sequential(*layers)
            heads = nn.ModuleDict()
            self.head_keys = {}
            for ci, cid in enumerate(sorted(clients)):
                for pi, projection in enumerate(self.projections):
                    ashape, bshape = projection_shapes[projection]
                    count = self.initial[
                        next(
                            a
                            for m, (a, _) in self.pairs.items()
                            if self.module_info[m][1] == projection
                        )
                    ].numel()
                    count += self.initial[
                        next(
                            b
                            for m, (_, b) in self.pairs.items()
                            if self.module_info[m][1] == projection
                        )
                    ].numel()
                    key = f"client{ci}_module{pi}"
                    heads[key] = nn.Linear(hidden_size, count)
                    nn.init.zeros_(heads[key].weight)
                    nn.init.zeros_(heads[key].bias)
                    self.head_keys[cid, projection] = (key, ashape, bshape)
            self.module["heads"] = heads
            self.module.to(self.device)
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=learning_rate)

    def generate(self, client_id: str) -> dict[str, Any]:
        torch = self.torch
        task = self.clients[client_id]
        vector = torch.tensor(self.descriptors[task], device=self.device, dtype=torch.float32)
        task_embed = self.module["task_embedding"](vector)
        modality = self.module["modality_embedding"].weight[0]
        result = {}
        for module, (a, b) in self.pairs.items():
            depth, projection = self.module_info[module]
            features = torch.cat(
                (
                    task_embed,
                    modality,
                    self.module["layer_embedding"].weight[depth],
                    self.module["projection_embedding"].weight[self.projections.index(projection)],
                )
            )
            key, ashape, bshape = self.head_keys[client_id, projection]
            output = self.module["heads"][key](self.module["trunk"](features))
            offset = self.initial[a].numel()
            result[a] = self.initial[a] + output[:offset].reshape(ashape)
            result[b] = self.initial[b] + output[offset:].reshape(bshape)
        return result

    def begin_meta_step(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def accumulate_meta_gradient(
        self, client_id: str, trained: dict[str, Any], downloaded: dict[str, Any], weight: float
    ) -> float:
        # phi remains fixed for the entire synchronous meta-batch.
        generated = self.generate(client_id)
        objective = (
            sum(
                (
                    generated[k]
                    * (
                        downloaded[k].detach().to(self.device).float()
                        - trained[k].detach().to(self.device).float()
                    )
                ).sum()
                for k in generated
            )
            * weight
        )
        objective.backward()  # VJP -Delta: descent moves generated factors toward local factors.
        return float(objective.detach())

    def finish_meta_step(self) -> None:
        self.optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "module": {k: v.detach().clone() for k, v in self.module.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
        }
