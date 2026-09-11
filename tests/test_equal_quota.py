from collections import Counter

from afl_vlm.federation.types import ClientSpec
from afl_vlm.scheduling.delay_models import PerClientDelay
from afl_vlm.scheduling.virtual_event import build_schedule


def test_each_client_has_exactly_two_real_uploads() -> None:
    clients = [
        ClientSpec("c0", "fast", 1.0),
        ClientSpec("c1", "fast", 1.2),
        ClientSpec("c2", "slow", 3.0),
        ClientSpec("c3", "slow", 3.2),
    ]
    schedule = build_schedule(
        clients,
        2,
        PerClientDelay({item.id: item.virtual_train_time for item in clients}),
        42,
    )
    assert Counter(event.client_id for event in schedule) == {
        "c0": 2,
        "c1": 2,
        "c2": 2,
        "c3": 2,
    }
    assert len(schedule) == 8
