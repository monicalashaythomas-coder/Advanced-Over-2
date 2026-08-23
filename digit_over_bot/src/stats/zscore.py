"""
One-sample z-test on the proportion of "digit > barrier" outcomes.

This IS a directional signal: it directly estimates whether the observed
win-rate for an Over-`barrier` bet in the current window differs from the
fair/theoretical rate, and which way.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ZScoreResult:
    n: int
    p_hat: float
    p_fair: float
    z: float
    p_value: float  # two-sided
    edge: float  # p_hat - p_fair, signed (positive => favors Over)
    significant: bool


def _normal_sf_two_sided(z: float) -> float:
    """Two-sided p-value from a standard-normal z statistic, no scipy needed."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def over_proportion_test(digits: list[int], barrier: int, alpha: float) -> ZScoreResult | None:
    n = len(digits)
    if n < 30:
        return None

    p_fair = (9 - barrier) / 10.0  # e.g. barrier=2 -> 7 winning digits / 10 -> 0.7
    hits = sum(1 for d in digits if d > barrier)
    p_hat = hits / n

    se = math.sqrt(p_fair * (1 - p_fair) / n)
    if se == 0:
        return None
    z = (p_hat - p_fair) / se
    p_value = _normal_sf_two_sided(z)

    return ZScoreResult(
        n=n,
        p_hat=p_hat,
        p_fair=p_fair,
        z=z,
        p_value=p_value,
        edge=p_hat - p_fair,
        significant=p_value < alpha,
    )
