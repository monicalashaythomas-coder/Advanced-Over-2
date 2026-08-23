"""
Rolling last-digit buffer.

Two buffers are kept per symbol, deliberately separate:

- `detect`: a fixed-length rolling window (default 1000) used by the
  structure-detection layer (chi-square / z-score / Markov). Because it's a
  rolling window, old ticks age OUT -- this is what lets the bot react if the
  exploitable structure (if any) drifts or disappears.
- `cumulative`: an exponentially-decayed running count used only to warm up
  the higher-order Markov tables faster than a 1000-tick window allows (order
  3 has 1000 possible conditioning states, so a single rolling window of 1000
  ticks gives ~1 sample per state on average -- nowhere near enough). The
  decay means old evidence still fades, just slower than the rolling window,
  which is what lets Markov tables reach usable sample sizes without
  pretending the process is stationary forever.
"""
from __future__ import annotations

from collections import deque


class RollingDigitBuffer:
    def __init__(self, maxlen: int = 1000) -> None:
        self.maxlen = maxlen
        self._digits: deque[int] = deque(maxlen=maxlen)

    def push(self, digit: int) -> None:
        if not 0 <= digit <= 9:
            raise ValueError(f"digit out of range: {digit}")
        self._digits.append(digit)

    def __len__(self) -> int:
        return len(self._digits)

    def window(self, n: int | None = None) -> list[int]:
        """Most recent `n` digits (or the whole buffer if n is None/too large)."""
        if n is None or n >= len(self._digits):
            return list(self._digits)
        return list(self._digits)[-n:]

    def full(self) -> bool:
        return len(self._digits) == self.maxlen


class DecayedMarkovCounts:
    """
    Exponentially-decayed transition counts for a single Markov order.

    counts[state][next_digit] accumulates with each observation, and every
    prior count is decayed by `decay` first so that evidence from far in the
    past contributes less than recent evidence, without a hard cutoff. This
    is what lets the Markov layer build up enough samples per state (states
    for order 3 = 1000 possible histories) while still tracking drift over a
    session.
    """

    def __init__(self, order: int, decay: float = 0.999) -> None:
        self.order = order
        self.decay = decay
        self.counts: dict[tuple[int, ...], list[float]] = {}

    def observe(self, history: tuple[int, ...], next_digit: int) -> None:
        if len(history) != self.order:
            return
        row = self.counts.setdefault(history, [0.0] * 10)
        for i in range(10):
            row[i] *= self.decay
        row[next_digit] += 1.0

    def get(self, history: tuple[int, ...]) -> list[float] | None:
        return self.counts.get(history)

    def state_count(self, history: tuple[int, ...]) -> float:
        row = self.counts.get(history)
        return sum(row) if row else 0.0
