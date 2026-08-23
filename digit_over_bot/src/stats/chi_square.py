"""
Chi-square goodness-of-fit test for the last-digit distribution.

This answers one question only: "does the observed digit distribution in
this window depart from uniform by more than sampling noise explains?" It is
a GATE, not a directional signal -- a significant chi-square tells you
*something* in the full 10-way distribution looks off, not which way. To get
a direction usable for an Over-2 bet, we look at the standardized residuals
per digit and check whether the excess mass sits on the "over 2" side
(digits 3-9) or the "under/equal 2" side (digits 0-2).
"""
from __future__ import annotations

from dataclasses import dataclass

from scipy import stats as sp_stats

OVER_DIGITS = tuple(range(3, 10))  # for barrier == 2; recomputed per-barrier below


@dataclass
class ChiSquareResult:
    n: int
    chi2_stat: float
    p_value: float
    df: int
    residuals: list[float]  # standardized residual per digit, index 0-9
    over_direction_score: float  # >0 => excess mass on the "over" side
    significant: bool


def goodness_of_fit(digits: list[int], alpha: float, barrier: int) -> ChiSquareResult | None:
    n = len(digits)
    if n < 30:
        # Chi-square with sparse expected counts (<5 per cell is the classic
        # rule of thumb -- n=30 gives expected 3/cell, already thin) is
        # unreliable; refuse to answer rather than return a noisy one.
        return None

    observed = [0] * 10
    for d in digits:
        observed[d] += 1
    expected = n / 10.0

    chi2_stat, p_value = sp_stats.chisquare(observed, f_exp=[expected] * 10)
    # Standardized (Pearson) residual per cell: (O-E)/sqrt(E*(1-1/10))
    # using sqrt(E) is the simplest classic form; good enough for direction.
    residuals = [(o - expected) / (expected**0.5) for o in observed]

    over_digits = tuple(d for d in range(10) if d > barrier)
    under_digits = tuple(d for d in range(10) if d <= barrier)
    over_score = sum(residuals[d] for d in over_digits) / len(over_digits)
    under_score = sum(residuals[d] for d in under_digits) / len(under_digits)
    # Positive => the "over" bucket is carrying more excess mass than the
    # "under" bucket, i.e. structure (if real) favors an Over bet.
    over_direction_score = over_score - under_score

    return ChiSquareResult(
        n=n,
        chi2_stat=chi2_stat,
        p_value=p_value,
        df=9,
        residuals=residuals,
        over_direction_score=over_direction_score,
        significant=p_value < alpha,
    )
