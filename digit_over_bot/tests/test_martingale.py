"""
Run with: python3 -m pytest tests/test_martingale.py -q
"""
from __future__ import annotations

from src.martingale import MartingaleStaking


def test_disabled_always_returns_base_stake() -> None:
    m = MartingaleStaking(base_stake=1.0, factor=2.5, max_steps=3, enabled=False)
    for _ in range(5):
        m.record_result("R_10", win=False)
    assert m.stake_for("R_10") == 1.0


def test_escalates_on_consecutive_losses_and_resets_on_win() -> None:
    m = MartingaleStaking(base_stake=1.0, factor=2.5, max_steps=3, enabled=True)
    assert m.stake_for("R_10") == 1.0  # step 0

    m.record_result("R_10", win=False)
    assert m.stake_for("R_10") == 2.5  # step 1

    m.record_result("R_10", win=False)
    assert m.stake_for("R_10") == 6.25  # step 2

    m.record_result("R_10", win=True)
    assert m.stake_for("R_10") == 1.0  # reset to step 0


def test_resets_after_max_steps_even_without_a_win() -> None:
    m = MartingaleStaking(base_stake=1.0, factor=2.5, max_steps=3, enabled=True)
    m.record_result("R_10", win=False)  # step 1
    m.record_result("R_10", win=False)  # step 2
    m.record_result("R_10", win=False)  # would be step 3 == max_steps -> reset to 0
    assert m.step_for("R_10") == 0
    assert m.stake_for("R_10") == 1.0


def test_symbols_are_tracked_independently() -> None:
    m = MartingaleStaking(base_stake=1.0, factor=2.5, max_steps=3, enabled=True)
    m.record_result("R_10", win=False)
    assert m.stake_for("R_10") == 2.5
    assert m.stake_for("R_25") == 1.0


if __name__ == "__main__":
    test_disabled_always_returns_base_stake()
    test_escalates_on_consecutive_losses_and_resets_on_win()
    test_resets_after_max_steps_even_without_a_win()
    test_symbols_are_tracked_independently()
    print("PASS")
