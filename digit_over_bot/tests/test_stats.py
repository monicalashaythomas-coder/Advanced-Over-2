"""
Run with: python3 -m tests.test_stats
No network required -- everything here is synthetic digit sequences.
"""
from __future__ import annotations

import random

from src.digit_buffer import RollingDigitBuffer
from src.stats.chi_square import goodness_of_fit
from src.stats.markov import MarkovLayer
from src.stats.zscore import over_proportion_test


def uniform_digits(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(0, 9) for _ in range(n)]


def biased_over_digits(n: int, seed: int, p_over: float, barrier: int) -> list[int]:
    """Digits where P(digit > barrier) = p_over, uniform within each side."""
    rng = random.Random(seed)
    over_pool = list(range(barrier + 1, 10))
    under_pool = list(range(0, barrier + 1))
    out = []
    for _ in range(n):
        if rng.random() < p_over:
            out.append(rng.choice(over_pool))
        else:
            out.append(rng.choice(under_pool))
    return out


def test_chi_square_uniform_mostly_not_significant() -> None:
    false_positives = 0
    trials = 200
    for seed in range(trials):
        digits = uniform_digits(1000, seed)
        result = goodness_of_fit(digits, alpha=0.01, barrier=2)
        assert result is not None
        if result.significant:
            false_positives += 1
    rate = false_positives / trials
    # at alpha=0.01 we expect ~1% false positive rate; allow generous slack
    assert rate < 0.05, f"false positive rate too high: {rate}"


def test_chi_square_detects_strong_bias() -> None:
    digits = biased_over_digits(1000, seed=1, p_over=0.85, barrier=2)
    result = goodness_of_fit(digits, alpha=0.01, barrier=2)
    assert result is not None
    assert result.significant
    assert result.over_direction_score > 0


def test_zscore_uniform_rarely_significant() -> None:
    false_positives = 0
    trials = 300
    for seed in range(trials):
        digits = uniform_digits(1000, seed + 5000)
        result = over_proportion_test(digits, barrier=2, alpha=0.01)
        assert result is not None
        if result.significant:
            false_positives += 1
    rate = false_positives / trials
    assert rate < 0.05, f"false positive rate too high: {rate}"


def test_zscore_detects_bias_direction() -> None:
    digits = biased_over_digits(1000, seed=2, p_over=0.85, barrier=2)
    result = over_proportion_test(digits, barrier=2, alpha=0.01)
    assert result is not None
    assert result.significant
    assert result.edge > 0


def test_zscore_fair_data_gives_near_zero_edge() -> None:
    digits = biased_over_digits(2000, seed=3, p_over=0.70, barrier=2)  # exactly fair
    result = over_proportion_test(digits, barrier=2, alpha=0.01)
    assert result is not None
    assert abs(result.edge) < 0.05


def test_markov_layer_learns_biased_transition() -> None:
    # Build a stream where, whenever the previous digit is even, the next
    # digit is heavily biased to be > 2; whenever odd, it's biased <= 2.
    rng = random.Random(42)
    digits: list[int] = [rng.randint(0, 9)]
    for _ in range(5000):
        prev = digits[-1]
        if prev % 2 == 0:
            nxt = rng.choice([3, 4, 5, 6, 7, 8, 9, 3, 4, 5])  # skewed over
        else:
            nxt = rng.choice([0, 1, 2, 0, 1, 2, 3])  # skewed under/borderline
        digits.append(nxt)

    layer = MarkovLayer(orders=(1, 2, 3))
    history: list[int] = []
    for d in digits:
        layer.observe(history, d)
        history.append(d)

    est_even = layer.predict_per_order(history=[0], min_count=10)[1]  # last digit 0 (even) -> expect over-bias
    est_odd = layer.predict_per_order(history=[1], min_count=10)[1]  # last digit 1 (odd) -> expect under-bias

    assert est_even is not None and est_odd is not None
    p_over_even = est_even.p_over(barrier=2)
    p_over_odd = est_odd.p_over(barrier=2)
    assert p_over_even > 0.6, p_over_even
    assert p_over_odd < 0.4, p_over_odd


def test_markov_backoff_falls_back_when_state_unseen() -> None:
    layer = MarkovLayer(orders=(1, 2, 3))
    history: list[int] = []
    rng = random.Random(7)
    for _ in range(200):
        d = rng.randint(0, 9)
        layer.observe(history, d)
        history.append(d)
    # order-3 state (1,2,3) almost certainly has near-zero samples in 200 draws
    per_order = layer.predict_per_order(history=[1, 2, 3], min_count=30)
    assert per_order[3] is None  # correctly refuses instead of trusting 0-1 samples
    backoff = layer.predict_backoff(history=[1, 2, 3], min_count=30)
    assert backoff.order in (0, 1, 2)


def test_rolling_buffer_window_and_full() -> None:
    buf = RollingDigitBuffer(maxlen=10)
    for i in range(15):
        buf.push(i % 10)
    assert buf.full()
    assert len(buf.window()) == 10
    assert len(buf.window(5)) == 5


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
