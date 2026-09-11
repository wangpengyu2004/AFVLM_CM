from afl_vlm.federation.types import ClientSpec
from afl_vlm.scheduling.delay_models import PerClientDelay
from afl_vlm.scheduling.virtual_event import build_schedule, shuffled_task_delays


def clients() -> list[ClientSpec]:
    return [
        ClientSpec("c0", "fast", 1.0),
        ClientSpec("c1", "fast", 1.2),
        ClientSpec("c2", "slow", 3.0),
        ClientSpec("c3", "slow", 3.2),
    ]


def test_same_seed_produces_identical_schedule() -> None:
    delay = PerClientDelay({client.id: client.virtual_train_time for client in clients()})
    first = build_schedule(clients(), 2, delay, 42)
    second = build_schedule(clients(), 2, delay, 42)
    assert first == second
    assert [(event.client_id, event.local_round) for event in first] == [
        ("c0", 0),
        ("c1", 0),
        ("c0", 1),
        ("c1", 1),
        ("c2", 0),
        ("c3", 0),
        ("c2", 1),
        ("c3", 1),
    ]


def test_task_shuffle_preserves_delay_multiset_and_mixes_speeds() -> None:
    shuffled = shuffled_task_delays(clients())
    assert sorted(item.virtual_train_time for item in shuffled) == [1.0, 1.2, 3.0, 3.2]
    by_task = {
        task: sorted(item.virtual_train_time for item in shuffled if item.task == task)
        for task in {"fast", "slow"}
    }
    assert all(values[0] < 2.0 < values[1] for values in by_task.values())
