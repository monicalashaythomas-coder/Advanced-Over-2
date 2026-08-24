"""
Monte-Carlo-guided duration selection.

Rather than trading every signal at one fixed duration, this estimates --
via block bootstrap over the recent digit window, not a parametric
multi-step Markov projection -- the probability that the digit *k* ticks
from now clears the barrier, for each candidate duration. Block bootstrap
(resampling contiguous chunks, not individual digits one at a time) is used
specifically so that whatever short-range dependency actually exists in the
observed window is preserved in the simulated forward paths, rather than
assumed away by treating digits as i.i.d.

This deliberately does NOT weight by payout/EV -- it purely picks the
duration with the highest simulated win probability. The existing per-trade
EV check against the live quoted payout (see executor.py) still runs
afterwards, using whichever duration this returns, and can still veto the
trade regardless of which duration was chosen here.

Honest expectation: because block-bootstrap resampling washes out
short-range memory as k grows, per-candidate win probabilities should
generally drift toward p_fair the further out you look. In practice this
will usually favor the shortest candidate that still carries a real edge --
that's the correct, non-overfit behavior, not a bug.
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class DurationChoice:
    duration_ticks: int
    win_prob: float
    per_candidate: dict[int, float]


def select_duration(
    digits_window: list[int],
    candidates: list[int],
    barrier: int,
    n_samples: int = 500,
    block_size: int = 10,
    rng: random.Random | None = None,
) -> DurationChoice | None:
    """Block-bootstrap the recent digit window forward `n_samples` times and
    pick whichever candidate duration has the highest simulated P(win).

    Returns None if the window is too short to bootstrap meaningfully (the
    caller should fall back to a fixed default duration in that case).
    """
    if not candidates:
        return None
    max_k = max(candidates)
    if block_size < 1 or len(digits_window) < max(2 * block_size, max_k, 1):
        return None

    rng = rng or random.Random()
    hits = {k: 0 for k in candidates}

    for _ in range(n_samples):
        path: list[int] = []
        while len(path) < max_k:
            start = rng.randrange(0, len(digits_window) - block_size + 1)
            path.extend(digits_window[start : start + block_size])
        for k in candidates:
            if path[k - 1] > barrier:
                hits[k] += 1

    per_candidate = {k: hits[k] / n_samples for k in candidates}
    best_k = max(per_candidate, key=per_candidate.get)
    return DurationChoice(duration_ticks=best_k, win_prob=per_candidate[best_k], per_candidate=per_candidate)
