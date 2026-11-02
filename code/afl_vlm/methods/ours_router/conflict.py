"""Exact recent SERVER-applied functional directions, isolated to conflict_router.

Design: 01a163c2-00b0-74b3-ab3c-6faef58610a9, 2026-11-02, L=3/T=24.
Each component change is rank <= 2. Cache its factor inner products, never BA,
sketches, concatenated parameter-delta cosines, or an EMA of directions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


class TaskConflictWindow:
    schema_version = 1

    def __init__(self, modules: Mapping, tasks: list[str], size: int = 3,
                 max_age: int = 24, eps: float = 1e-12, min_norm: float = 1e-12):
        self.modules = {m: dict(item) for m, item in modules.items()}
        self.tasks = sorted(set(tasks))
        if not self.modules or not self.tasks or type(size) is not int or size < 1:
            raise ValueError("Conflict cache requires modules, tasks and a positive window")
        if type(max_age) is not int or max_age < 0:
            raise ValueError("Conflict max_age must be a nonnegative integer")
        if not all(math.isfinite(v) and v > 0 for v in (eps, min_norm)):
            raise ValueError("Conflict numerical thresholds must be finite and positive")
        self.size, self.max_age, self.eps, self.min_norm = size, max_age, eps, min_norm
        self.clock = 0
        self.rows: list[dict] = []
        self.grams: dict[str, Any] = {}

    def _new(self) -> TaskConflictWindow:
        return TaskConflictWindow(
            self.modules, self.tasks, self.size, self.max_age, self.eps, self.min_norm
        )

    def append(self, before: Mapping, after: Mapping, selected: Mapping,
               task: str, update_id: str, accepted_at: int) -> TaskConflictWindow:
        """Build a copy-on-write candidate; publish it only AFTER server commit.

        E = kappa*(b+ a+ - b- a-)
          = [kappa*(b+ - b-), kappa*b-] @ [a+, a+ - a-].
        The second exact factorization avoids subtracting two large near-equal
        Gram terms. All historical factors stay immutable and on their device.
        """
        import torch

        if (task not in self.tasks or accepted_at != self.clock + 1
                or any(row["update_id"] == update_id for row in self.rows)):
            raise ValueError("Conflict records require unique consecutive successful updates")
        if set(selected) != set(self.modules):
            raise ValueError("Conflict selection differs from module schema")
        new = self._new()
        new.clock = accepted_at
        same = [i for i, row in enumerate(self.rows) if row["task"] == task]
        discard = set(same[:max(0, len(same) - self.size + 1)])
        keep = [i for i, row in enumerate(self.rows)
                if i not in discard and accepted_at - row["accepted_at"] <= self.max_age]
        new.rows = [self.rows[i] for i in keep]
        record = {"task": task, "update_id": update_id,
                  "accepted_at": accepted_at, "modules": {}}
        n = len(keep)
        with torch.no_grad():
            for name, item in self.modules.items():
                indices = list(selected[name])
                if (len(indices) != len(set(indices))
                        or any(type(j) is not int or not 0 <= j < item["rank"] for j in indices)):
                    raise ValueError("Invalid conflict component indices")
                a0, b0 = before[item["a_name"]], before[item["b_name"]]
                a1, b1 = after[item["a_name"]], after[item["b_name"]]
                index = torch.tensor(indices, device=a0.device, dtype=torch.long)
                dtype = torch.float64 if a0.dtype == torch.float64 else torch.float32
                aa = a1.index_select(0, index).to(dtype=dtype)
                bb = b1.index_select(1, index).T.to(dtype=dtype)
                da = aa - a0.index_select(0, index).to(dtype=dtype)
                bbase = b0.index_select(1, index).T.to(dtype=dtype)
                left = torch.stack((bb - bbase, bbase), dim=1) * float(item["scaling"])
                right = torch.stack((aa, da), dim=1)
                if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
                    raise ValueError("Non-finite applied functional direction")
                record["modules"][name] = {"indices": indices, "left": left, "right": right}
                gram = torch.zeros((item["rank"], n + 1, n + 1),
                                   device=a0.device, dtype=torch.float64)
                if keep:
                    positions = torch.tensor(keep, device=a0.device, dtype=torch.long)
                    gram[:, :n, :n] = self.grams[name].index_select(1, positions).index_select(
                        2, positions
                    )
                for old_position, old_record in enumerate(new.rows + [record]):
                    old = old_record["modules"][name]
                    positions = {j: p for p, j in enumerate(old["indices"])}
                    common = [j for j in indices if j in positions]
                    if not common:
                        continue
                    current_positions = {j: p for p, j in enumerate(indices)}
                    ci = torch.tensor([current_positions[j] for j in common], device=a0.device)
                    oi = torch.tensor([positions[j] for j in common], device=a0.device)
                    # Batched rank-two exact Frobenius products; double accumulation
                    # limits cancellation of multiple opposing historical changes.
                    current_left = left.index_select(0, ci).double()
                    r = right.index_select(0, ci).double()
                    ol = old["left"].index_select(0, oi).double()
                    oright = old["right"].index_select(0, oi).double()
                    products = ((current_left @ ol.transpose(1, 2)) *
                                (r @ oright.transpose(1, 2))).sum((1, 2))
                    if not bool(torch.isfinite(products).all()):
                        raise ValueError("Non-finite functional Gram entry")
                    global_indices = torch.tensor(common, device=a0.device)
                    gram[global_indices, n, old_position] = products
                    gram[global_indices, old_position, n] = products
                new.grams[name] = gram
        new.rows.append(record)
        return new

    def scores(self, task: str) -> dict:
        """Task-specific negative cosine, averaging only valid OTHER tasks.

        No own direction or no valid peer => unknown, zero penalty. Every task
        contributes ONE sum of recent changes, not one vote per upload.
        """
        import torch

        if task not in self.tasks:
            raise ValueError("Unknown conflict query task")
        result = {}
        task_index = self.tasks.index(task)
        for name, item in self.modules.items():
            if not self.rows:
                result[name] = {"penalty": [0.0] * item["rank"],
                                "valid": [False] * item["rank"],
                                "peer_count": [0] * item["rank"],
                                "own_direction_valid": [False] * item["rank"]}
                continue
            gram = self.grams[name]
            membership = gram.new_tensor([
                [float(row["task"] == q) for q in self.tasks] for row in self.rows
            ])
            task_gram = torch.einsum("ui,ruv,vj->rij", membership, gram, membership)
            energy = task_gram.diagonal(dim1=1, dim2=2).clamp_min(0)
            # Near cancellation relative to the component's individual changes
            # must not turn floating-point residue into a unit direction.
            scale = torch.einsum("ru,ui->ri", gram.diagonal(dim1=1, dim2=2).abs(), membership)
            valid = energy > torch.maximum(scale * 1e-12, energy.new_tensor(self.min_norm ** 2))
            own = valid[:, task_index]
            peer = valid & own[:, None]
            peer[:, task_index] = False
            counts = peer.sum(1)
            denominator = energy[:, task_index].sqrt()[:, None] * energy.sqrt() + self.eps
            cosine = (task_gram[:, task_index, :] / denominator).clamp(-1, 1)
            penalty = ((-cosine).clamp_min(0) * peer).sum(1) / counts.clamp_min(1)
            result[name] = {
                "penalty": penalty.cpu().tolist(), "valid": (counts > 0).cpu().tolist(),
                "peer_count": counts.cpu().tolist(), "own_direction_valid": own.cpu().tolist(),
            }
        return {"clock": self.clock, "modules": result, "window": self.summary()}

    def summary(self) -> dict:
        return {"per_task_limit": self.size, "max_age": self.max_age,
                "successful_updates": self.clock, "record_count": len(self.rows),
                "tasks": {q: [{"update_id": row["update_id"],
                               "accepted_at": row["accepted_at"],
                               "age": self.clock - row["accepted_at"]}
                              for row in self.rows if row["task"] == q]
                          for q in self.tasks},
                "factor_bytes": sum(v[side].numel() * v[side].element_size()
                                    for row in self.rows for v in row["modules"].values()
                                    for side in ("left", "right")),
                "gram_bytes": sum(g.numel() * g.element_size() for g in self.grams.values())}

    def state_dict(self) -> dict:
        import copy

        from afl_vlm.models.base import nested_to_device

        return {"schema_version": self.schema_version, "modules": copy.deepcopy(self.modules),
                "tasks": list(self.tasks), "size": self.size, "max_age": self.max_age,
                "eps": self.eps, "min_norm": self.min_norm, "clock": self.clock,
                "rows": copy.deepcopy(nested_to_device(self.rows, "cpu")),
                "grams": {m: v.detach().cpu().clone() for m, v in self.grams.items()}}

    def load_state_dict(self, source: Mapping, device: Any = "cpu") -> None:
        import copy

        from afl_vlm.models.base import nested_to_device

        if source.get("schema_version") != self.schema_version or any(
            source[k] != getattr(self, k)
            for k in ("modules", "tasks", "size", "max_age", "eps", "min_norm")
        ):
            raise ValueError("Incompatible conflict window schema/configuration")
        clock, rows = source["clock"], source["rows"]
        if type(clock) is not int or clock < 0 or any(
            type(row["accepted_at"]) is not int or not 1 <= row["accepted_at"] <= clock
            or clock - row["accepted_at"] > self.max_age or row["task"] not in self.tasks
            for row in rows
        ):
            raise ValueError("Invalid conflict record clock/task/age")
        if ([row["accepted_at"] for row in rows] != sorted({row["accepted_at"] for row in rows})
                or len({row["update_id"] for row in rows}) != len(rows)
                or any(sum(row["task"] == q for row in rows) > self.size for q in self.tasks)):
            raise ValueError("Invalid conflict record chronology or per-task window")
        if (clock == 0) != (not rows):
            raise ValueError("Conflict clock and retained records disagree")
        candidate = self._new()
        candidate.clock = clock
        candidate.rows = nested_to_device(copy.deepcopy(rows), device)
        candidate.grams = nested_to_device(copy.deepcopy(source["grams"]), device)
        import torch

        for row in candidate.rows:
            if set(row["modules"]) != set(self.modules):
                raise ValueError("Conflict record module set differs from the schema")
            for name, item in self.modules.items():
                value = row["modules"][name]
                indices = value["indices"]
                if (len(indices) != len(set(indices))
                        or any(type(j) is not int or not 0 <= j < item["rank"] for j in indices)
                        or tuple(value["left"].shape) != (len(indices), 2, item["out_features"])
                        or tuple(value["right"].shape) != (len(indices), 2, item["in_features"])
                        or not bool(torch.isfinite(value["left"]).all())
                        or not bool(torch.isfinite(value["right"]).all())):
                    raise ValueError("Invalid restored functional factor record")
        if set(candidate.grams) != (set(self.modules) if rows else set()):
            raise ValueError("Invalid conflict Gram module set")
        for name, gram in candidate.grams.items():
            if (tuple(gram.shape) != (self.modules[name]["rank"], len(rows), len(rows))
                    or not bool(torch.isfinite(gram).all())
                    or not torch.allclose(gram, gram.transpose(1, 2))):
                raise ValueError("Invalid conflict Gram shape/values")
        self.clock, self.rows, self.grams = candidate.clock, candidate.rows, candidate.grams

    def to(self, device: Any) -> None:
        """Move restored storage once at the first server/download boundary."""
        from afl_vlm.models.base import nested_to_device

        if self.rows and next(iter(self.grams.values())).device != device:
            self.rows = nested_to_device(self.rows, device)
            self.grams = nested_to_device(self.grams, device)
