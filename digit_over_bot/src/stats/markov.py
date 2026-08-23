"""
Markov chain layer, orders 1-3, over the last-digit stream.

Each order k conditions on the last k digits and predicts a distribution
over the next digit. Higher orders are more specific (and more useful if the
generator genuinely isn't memoryless) but also need exponentially more data
per state (10^k possible histories) -- order 3 alone has 1000 states, so a
window of 1000 digits gives ~1 sample per state on average, which is not
enough to trust. Two mechanisms handle this:

1. Decayed cumulative counts (see DecayedMarkovCounts) let a state accumulate
   evidence across a session rather than only the last 1000 ticks, while
   still discounting stale evidence.
2. Backoff: when a given order's state has too few observations, fall back
   to a lower order (eventually the unconditional digit frequency) rather
   than trusting a thin estimate. This is classic Katz backoff.

Two different outputs are exposed on purpose:

- `predict_per_order`: each order's OWN estimate, returned only when that
  order individually has enough samples at the current state -- used so the
  ensemble can ask "do order 1, order 2, and order 3 independently agree?"
  without one high-order estimate silently just being a copy of a low-order
  one via backoff.
- `predict_backoff`: a single best-effort distribution using backoff -- used
  as a convenience estimate, not as one of the "independent" votes.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.digit_buffer import DecayedMarkovCounts


@dataclass
class MarkovEstimate:
    order: int
    state_count: float
    probs: list[float]  # length 10, P(next digit == i)

    def p_over(self, barrier: int) -> float:
        return sum(self.probs[d] for d in range(10) if d > barrier)


class MarkovLayer:
    def __init__(self, orders: tuple[int, ...] = (1, 2, 3), decay: float = 0.999) -> None:
        self.orders = orders
        self.models = {o: DecayedMarkovCounts(o, decay) for o in orders}
        self.base = DecayedMarkovCounts(0, decay)

    def observe(self, history_before: list[int], next_digit: int) -> None:
        """`history_before` = digits seen so far, oldest first, NOT including next_digit."""
        self.base.observe((), next_digit)
        for o, model in self.models.items():
            if len(history_before) >= o:
                model.observe(tuple(history_before[-o:]), next_digit)

    @staticmethod
    def _smoothed(row: list[float], alpha: float = 1.0) -> list[float]:
        total = sum(row)
        return [(c + alpha) / (total + 10 * alpha) for c in row]

    def predict_per_order(
        self, history: list[int], min_count: float, alpha_smooth: float = 1.0
    ) -> dict[int, MarkovEstimate | None]:
        out: dict[int, MarkovEstimate | None] = {}
        for o, model in self.models.items():
            if len(history) < o:
                out[o] = None
                continue
            hist = tuple(history[-o:])
            row = model.get(hist)
            cnt = model.state_count(hist)
            if row is None or cnt < min_count:
                out[o] = None
                continue
            out[o] = MarkovEstimate(order=o, state_count=cnt, probs=self._smoothed(row, alpha_smooth))
        return out

    def predict_backoff(
        self, history: list[int], min_count: float, alpha_smooth: float = 1.0
    ) -> MarkovEstimate:
        for o in sorted(self.models.keys(), reverse=True):
            if len(history) < o:
                continue
            model = self.models[o]
            hist = tuple(history[-o:])
            row = model.get(hist)
            cnt = model.state_count(hist)
            if row is not None and cnt >= min_count:
                return MarkovEstimate(order=o, state_count=cnt, probs=self._smoothed(row, alpha_smooth))
        row = self.base.get(()) or [0.0] * 10
        cnt = sum(row)
        return MarkovEstimate(order=0, state_count=cnt, probs=self._smoothed(row, alpha_smooth))

    def export_state(self) -> dict[int, dict[str, list[float]]]:
        """Serialize the cumulative decayed counts for every order, for
        persistence across restarts -- this is what lets order 2/3 warm up
        instead of resetting to zero every deploy."""
        return {o: model.to_dict() for o, model in self.models.items()}

    def import_state(self, state: dict[int, dict[str, list[float]]]) -> None:
        """Restore cumulative counts previously produced by export_state()."""
        for o, data in state.items():
            o = int(o)
            if o in self.models:
                self.models[o].load_dict(data)
