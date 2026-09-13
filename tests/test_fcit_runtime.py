from pathlib import Path

import yaml
from afl_vlm.config import load_config, validate_config
from afl_vlm.federation.types import ClientSpec, ServerContext, Update
from afl_vlm.methods.baselines.fedopt_sync import FedAdamSyncMethod, FedYogiSyncMethod
from afl_vlm.methods.registry import method_names
from afl_vlm.models.registry import model_names
from afl_vlm.scheduling.delay_models import PerClientDelay
from afl_vlm.scheduling.virtual_event import build_fedcompass_schedule


def _update(identifier: str, delta: float, examples: int = 1) -> Update:
    return Update(
        update_id=identifier,
        client_id=identifier,
        task="Task1",
        local_round=0,
        download_version=0,
        base_state={"x": [0.0]},
        delta={"x": [delta]},
        seed=42,
        sample_ids_hash="hash",
        losses=[1.0],
        optimizer_steps=1,
        metadata={"num_examples": examples},
    )


def test_fcit_and_baseline_registries_are_complete() -> None:
    assert "llava15_fcit" in model_names()
    assert {
        "async_additive",
        "staleness_decay",
        "fedavg_sync",
        "fedasync",
        "fedbuff",
        "fedadam_sync",
        "fedyogi_sync",
        "fedcompass_sim",
    } <= method_names()


def test_fedopt_adam_and_yogi_apply_server_adaptivity() -> None:
    context = ServerContext(global_state={"x": [0.0]}, version=0, expected_clients=1)
    params = {
        "server_lr": 1.0,
        "beta1": 0.0,
        "beta2": 0.0,
        "tau": 1.0,
        "sample_weighted": True,
    }
    for method in (FedAdamSyncMethod(params), FedYogiSyncMethod(params)):
        mutation = method.on_arrival(_update("u", 1.0), context)[0]
        assert mutation.new_state == {"x": [0.5]}
        assert mutation.metadata["sample_weights"] == [1.0]


def test_fedcompass_assigns_more_steps_to_faster_clients() -> None:
    clients = [ClientSpec("fast", "Task1", 1.0), ClientSpec("slow", "Task2", 4.0)]
    schedule = build_fedcompass_schedule(
        clients,
        upload_quota=1,
        delay_model=PerClientDelay({"fast": 1.0, "slow": 4.0}),
        seed=42,
        base_local_steps=5,
        min_local_steps=1,
        max_local_steps=5,
    )
    by_client = {event.client_id: event for event in schedule}
    assert by_client["fast"].local_steps == 5
    assert by_client["slow"].local_steps == 3
    assert by_client["fast"].virtual_duration < by_client["slow"].virtual_duration


def test_fcit_preset_hydrates_tasks_and_assignments(tmp_path: Path) -> None:
    variant = tmp_path / "benchmark" / "small_16_clients" / "balanced"
    variant.mkdir(parents=True)
    generated = {
        "tasks": {
            "Task1": {
                "backend": "fedmllm",
                "name": "imagenet_r_classification",
                "train_per_client": 1,
                "allow_synthetic": False,
                "image_root": "old",
                "require_images": True,
                "data_files": {"probe": "probe.jsonl", "final": "final.jsonl"},
            },
            "Task2": {
                "backend": "fedmllm",
                "name": "arxivqa_scientific_vqa",
                "train_per_client": 1,
                "allow_synthetic": False,
                "image_root": "old",
                "require_images": True,
                "data_files": {"probe": "probe2.jsonl", "final": "final2.jsonl"},
            },
        },
        "clients": {
            "count": 2,
            "assignments": [
                {
                    "id": "c1",
                    "task": "Task1",
                    "virtual_train_time": 1.0,
                    "train_file": "c1.jsonl",
                },
                {
                    "id": "c2",
                    "task": "Task2",
                    "virtual_train_time": 2.0,
                    "train_file": "c2.jsonl",
                },
            ],
        },
        "timing": {"train_delay": {"type": "per_task", "values": {"Task1": 1, "Task2": 2}}},
    }
    (variant / "framework_config.yaml").write_text(yaml.safe_dump(generated), encoding="utf-8")
    base = yaml.safe_load(Path("configs/fcit.yaml").read_text(encoding="utf-8"))
    base["dataset"]["root"] = "benchmark"
    base["dataset"]["image_root"] = "images"
    config_path = tmp_path / "configs" / "fcit.yaml"
    config_path.parent.mkdir()
    config_path.write_text(yaml.safe_dump(base), encoding="utf-8")

    resolved = load_config(config_path)
    validate_config(resolved)
    assert resolved["clients"]["count"] == 2
    assert resolved["clients"]["upload_quota"] == 10
    assert resolved["tasks"]["Task1"]["image_root"] == "images"
    assert resolved["run"]["output_root"] == "runs/fcit/small_16_clients/balanced"
