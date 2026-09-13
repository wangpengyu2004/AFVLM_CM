import json
from copy import deepcopy
from pathlib import Path

import pytest
from afl_vlm.config import load_config, validate_config
from afl_vlm.data.fedmllm import FedMLLMTaskAdapter
from afl_vlm.runner import execute

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "tests" / "fixtures" / "tiny_config.yaml"


def record(sample_id: str, image: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is shown?"},
            {"from": "gpt", "value": "answer"},
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_shared_image_root_resolves_outside_nested_client_directory(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image = image_root / "FigureQA" / "images" / "42.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\nvalid-enough-for-path-test")
    client_file = tmp_path / "benchmark" / "clients" / "c0" / "train.jsonl"
    write_jsonl(client_file, [record("sample-0", "FigureQA/images/42.png")])
    adapter = FedMLLMTaskAdapter(
        "Task7",
        {
            "name": "figureqa_chart_vqa",
            "image_root": str(image_root),
            "require_images": True,
            "data_files": {},
        },
    )

    samples = adapter.load_file(client_file)

    assert samples[0].image == str(image.resolve())


def test_required_missing_image_fails_during_data_loading(tmp_path: Path) -> None:
    client_file = tmp_path / "clients" / "c0" / "train.jsonl"
    write_jsonl(client_file, [record("sample-0", "ImageNet-R/train/missing.jpg")])
    adapter = FedMLLMTaskAdapter(
        "Task1",
        {
            "name": "imagenet_r_classification",
            "image_root": str(tmp_path / "images"),
            "require_images": True,
            "data_files": {},
        },
    )

    with pytest.raises(FileNotFoundError, match="Image required by task 'Task1' is missing"):
        adapter.load_file(client_file)


def test_prepartitioned_configuration_accepts_per_client_train_files() -> None:
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["clients"]["data_partition"] = "prepartitioned"
    for assignment in config["clients"]["assignments"]:
        assignment["train_file"] = f"data/{assignment['id']}.jsonl"
    for experiment in config["experiments"]:
        experiment["enabled"] = experiment["type"] == "end_to_end"

    validate_config(config)


def test_prepartitioned_configuration_requires_every_train_file() -> None:
    config = deepcopy(load_config(DEFAULT_CONFIG))
    config["clients"]["data_partition"] = "prepartitioned"
    for assignment in config["clients"]["assignments"][1:]:
        assignment["train_file"] = f"data/{assignment['id']}.jsonl"
    for experiment in config["experiments"]:
        experiment["enabled"] = experiment["type"] == "end_to_end"

    with pytest.raises(ValueError, match="needs train_file"):
        validate_config(config)


def test_prepartitioned_clients_run_without_second_partition(tmp_path: Path) -> None:
    config = deepcopy(load_config(DEFAULT_CONFIG))
    image_root = tmp_path / "images"
    image = image_root / "shared" / "image.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\x89PNG\r\n\x1a\npath-test")
    for task_key, task in config["tasks"].items():
        probe = tmp_path / task_key / "probe.jsonl"
        final = tmp_path / task_key / "final.jsonl"
        write_jsonl(probe, [record(f"{task_key}-probe", "shared/image.png")])
        write_jsonl(final, [record(f"{task_key}-final", "shared/image.png")])
        task.update(
            {
                "allow_synthetic": False,
                "image_root": str(image_root),
                "require_images": True,
                "data_files": {"probe": str(probe), "final": str(final)},
            }
        )
    config["clients"]["data_partition"] = "prepartitioned"
    config["clients"]["upload_quota"] = 1
    for assignment in config["clients"]["assignments"]:
        train_file = tmp_path / assignment["id"] / "train.jsonl"
        write_jsonl(
            train_file,
            [record(f"{assignment['id']}-only-record", "shared/image.png")],
        )
        assignment["train_file"] = str(train_file)
    config["training"].update({"local_steps": 1, "grad_accumulation": 1})
    config["evaluation"].update(
        {"probe_samples_per_task": 1, "final_samples_per_task": 1, "window_size": 2}
    )
    for experiment in config["experiments"]:
        experiment["enabled"] = experiment["type"] == "end_to_end"
        if experiment["enabled"]:
            experiment["params"]["delay_scenarios"] = ["task_correlated"]
    config["run"]["output_root"] = str(tmp_path / "runs")

    rows = execute(config)

    assert len(rows) == 1
    assert rows[0]["received_updates"] == 4
