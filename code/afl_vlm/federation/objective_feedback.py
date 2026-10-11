"""Execute an optional algorithm oracle on PRIVATE client training data only."""

from __future__ import annotations

import random
import time
from typing import Any

from afl_vlm.data.partition import sample_ids_hash


def client_objective_feedback(
    model: Any, samples: list[Any], request: dict[str, Any], max_text_length: int
) -> dict[str, Any]:
    started = time.monotonic()
    count = min(len(samples), int(request["num_samples"]))
    if count < 1:
        raise ValueError("Client objective feedback has no local training samples")
    selected = random.Random(int(request["seed"])).sample(samples, count)
    model._set_context(request["task"], request["client_id"])
    result = model.mixing_objective_feedback(
        request["left"],
        request["right"],
        float(request["weight"]),
        samples=selected,
        max_text_length=max_text_length,
    )
    return {
        **result,
        "client_id": request["client_id"],
        "server_version": request["server_version"],
        "probe_weight": request["weight"],
        "sample_ids_hash": sample_ids_hash(selected),
        "data_split": "client_train",
        "wall_time_seconds": time.monotonic() - started,
        "extra_work_not_in_train_plan": True,
    }
