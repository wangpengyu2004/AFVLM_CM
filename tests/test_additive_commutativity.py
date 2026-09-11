from afl_vlm.models.base import add_scaled


def test_fixed_additive_updates_commute() -> None:
    initial = {"adapter": [0.0, 1.0, -1.0]}
    first = {"adapter": [0.2, -0.3, 0.4]}
    second = {"adapter": [-0.7, 0.1, 0.5]}
    forward = add_scaled(add_scaled(initial, first, 0.5), second, 0.5)
    reverse = add_scaled(add_scaled(initial, second, 0.5), first, 0.5)
    assert forward == reverse
