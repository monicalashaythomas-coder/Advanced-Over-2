"""
Run with: python3 -m tests.test_learner
"""
from __future__ import annotations

import random

from src.learner import Learner


def test_weight_shifts_toward_better_model() -> None:
    learner = Learner(["good", "bad"], learning_rate=0.5, calibration_window=500)
    rng = random.Random(1)
    p_fair = 0.7
    for _ in range(500):
        outcome_over = rng.random() < 0.85
        # "good" model tracks the true rate closely, "bad" model is overconfident wrongly.
        learner.observe_tick({"good": 0.85, "bad": 0.3}, p_fair, outcome_over)
    weights = learner.weights_snapshot()
    assert weights["good"] > weights["bad"]


def test_calibration_pause_fires_when_ensemble_is_worse_than_baseline() -> None:
    learner = Learner(["m"], learning_rate=0.1, calibration_window=200, calibration_pause_threshold=0.05)
    rng = random.Random(2)
    p_fair = 0.7
    for _ in range(200):
        outcome_over = rng.random() < 0.7  # truly fair -- no exploitable edge
        # ensemble overconfidently claims 0.95 every time -- badly miscalibrated
        learner.observe_ensemble(0.95, outcome_over)
        learner.observe_tick({}, p_fair, outcome_over)  # feeds the baseline_brier deque
    paused, reason = learner.should_pause()
    assert paused, "should have paused given a badly miscalibrated ensemble"
    assert reason is not None


def test_no_pause_with_insufficient_history() -> None:
    learner = Learner(["m"], calibration_window=200)
    for _ in range(10):
        learner.observe_ensemble(0.95, True)
        learner.observe_tick({}, 0.7, True)
    paused, _ = learner.should_pause()
    assert not paused


def test_no_pause_when_ensemble_matches_baseline() -> None:
    learner = Learner(["m"], calibration_window=200, calibration_pause_threshold=0.05)
    rng = random.Random(3)
    for _ in range(200):
        outcome_over = rng.random() < 0.7
        learner.observe_ensemble(0.7, outcome_over)  # ensemble just predicts fair odds -- matches baseline
        learner.observe_tick({}, 0.7, outcome_over)
    paused, _ = learner.should_pause()
    assert not paused


if __name__ == "__main__":
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
