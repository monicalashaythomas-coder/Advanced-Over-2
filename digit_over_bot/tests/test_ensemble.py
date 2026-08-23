"""
Run with: python3 -m tests.test_ensemble
"""
from __future__ import annotations

import random

from src.digit_buffer import RollingDigitBuffer
from src.ensemble import Ensemble
from src.stats.markov import MarkovLayer


def make_ensemble() -> Ensemble:
    return Ensemble(
        barrier=2,
        alpha=0.01,
        min_edge=0.03,
        min_edge_sigma_multiple=2.0,
        min_models_agreeing=3,
        min_markov_state_count=20,
    )


def run_stream(digits: list[int], barrier: int = 2):
    buffer = RollingDigitBuffer(maxlen=1000)
    markov = MarkovLayer(orders=(1, 2, 3))
    ensemble = make_ensemble()
    weights = {"zscore": 1.0, "markov_order_1": 1.0, "markov_order_2": 1.0, "markov_order_3": 1.0}
    trade_count = 0
    total_ticks = 0
    last_result = None
    for d in digits:
        history_before = buffer.window()
        markov.observe(history_before, d)
        buffer.push(d)
        result = ensemble.evaluate("TEST", buffer.window(), markov, weights)
        total_ticks += 1
        if result.should_trade:
            trade_count += 1
        last_result = result
    return trade_count, total_ticks, last_result


def test_uniform_random_rarely_trades() -> None:
    rng = random.Random(123)
    digits = [rng.randint(0, 9) for _ in range(3000)]
    trade_count, total_ticks, _ = run_stream(digits)
    rate = trade_count / total_ticks
    # With alpha=0.01, min_edge floor, sigma-multiple, AND a 3-model agreement
    # requirement stacked together, the false "should_trade" rate on genuinely
    # uniform data should be far below the nominal 1% of any single test.
    assert rate < 0.02, f"traded too often on pure noise: {rate:.4f} ({trade_count}/{total_ticks})"


def test_strongly_biased_stream_eventually_trades() -> None:
    # Digit is > 2 with probability 0.85 (vs fair 0.70) -- a strong, stable edge.
    rng = random.Random(9)
    digits = []
    for _ in range(3000):
        if rng.random() < 0.85:
            digits.append(rng.choice([3, 4, 5, 6, 7, 8, 9]))
        else:
            digits.append(rng.choice([0, 1, 2]))
    trade_count, total_ticks, last_result = run_stream(digits)
    assert trade_count > 0, "ensemble never fired on a strong, stable, 3000-tick bias"
    assert last_result is not None


def test_no_trade_without_enough_data() -> None:
    ensemble = make_ensemble()
    markov = MarkovLayer(orders=(1, 2, 3))
    weights = {"zscore": 1.0, "markov_order_1": 1.0, "markov_order_2": 1.0, "markov_order_3": 1.0}
    result = ensemble.evaluate("TEST", [3, 4, 5, 6, 7, 8, 9, 3, 4, 5], markov, weights)
    assert not result.should_trade
    assert "insufficient" in result.reasons[0] or result.combined_edge is None


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
