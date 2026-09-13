from afl_vlm.federation.types import ClientContext, ServerContext, Update
from afl_vlm.methods.baselines.fedasync import FedAsyncMethod
from afl_vlm.methods.baselines.fedbuff import FedBuffMethod
from afl_vlm.methods.custom.afvlm_cm import AFVLMCMMethod


def update(identifier: str, delta: float, version: int = 0) -> Update:
    return Update(
        update_id=identifier,
        client_id=identifier,
        task="fast",
        local_round=0,
        download_version=version,
        base_state={"x": [0.0]},
        delta={"x": [delta]},
        seed=42,
        sample_ids_hash="hash",
        losses=[1.0],
        optimizer_steps=1,
    )


def test_fedasync_interpolates_local_model_not_raw_delta() -> None:
    method = FedAsyncMethod({"alpha": 0.5, "decay_exponent": 0.0})
    mutations = method.on_arrival(
        update("u", 2.0), ServerContext({"x": [1.0]}, version=1, expected_clients=1)
    )
    assert mutations[0].new_state == {"x": [1.5]}


def test_fedbuff_waits_then_applies_mean_delta() -> None:
    method = FedBuffMethod({"server_lr": 0.5, "buffer_size": 2})
    context = ServerContext({"x": [0.0]}, version=0, expected_clients=2)
    assert method.on_arrival(update("u1", 1.0), context) == []
    mutation = method.on_arrival(update("u2", 3.0), context)[0]
    assert mutation.new_state == {"x": [1.0]}
    assert mutation.contributing_update_ids == ["u1", "u2"]


def test_afvlm_cm_removes_only_the_conflicting_stale_component() -> None:
    method = AFVLMCMMethod({"server_lr": 1.0, "staleness_exponent": 0.0, "download_strength": 0.1})
    stale = update("u", -2.0, version=0)
    mutation = method.on_arrival(stale, ServerContext({"x": [1.0]}, version=3, expected_clients=1))[
        0
    ]
    assert mutation.new_state == {"x": [1.0]}
    assert mutation.metadata["conflict_cosine"] == -1.0
    assert mutation.metadata["projection_removed"] == 2.0


def test_afvlm_cm_task_memory_changes_only_matching_task_download() -> None:
    method = AFVLMCMMethod({"server_lr": 1.0, "staleness_exponent": 0.0, "download_strength": 0.25})
    method.on_arrival(update("u", 2.0), ServerContext({"x": [0.0]}, 0, 1))
    same_task = ClientContext("c1", "fast", 1, 1, 42)
    other_task = ClientContext("c2", "slow", 1, 1, 42)
    assert method.prepare_download({"x": [1.0]}, same_task) == {"x": [1.5]}
    assert method.prepare_download({"x": [1.0]}, other_task) == {"x": [1.0]}

    restored = AFVLMCMMethod(method.params)
    restored.load_state_dict(method.state_dict())
    assert restored.prepare_download({"x": [1.0]}, same_task) == {"x": [1.5]}
