"""
Online relearning loop.

Two independent things are learned continuously, on EVERY tick that had an
opinion to check (not only on ticks where a trade actually fired -- that
would starve learning of data, since trades are rare by design):

1. Per-model reliability weights, via a multiplicative-weights / Hedge-style
   update on each model's Brier score against the realized outcome. A model
   that has been running well-calibrated recently gets upweighted in the
   ensemble's inverse-variance combination; one that's been off gets
   downweighted. This is what "relearning as the pattern develops" means
   concretely here -- weights drift with recent performance, they are not
   fixed at start.

2. The ensemble's OWN rolling calibration, benchmarked against the trivial
   baseline of just always predicting the fair probability. If the
   ensemble's combined estimate has recently been a *worse* predictor than
   simply assuming nothing is exploitable, that is a direct, falsifiable
   signal that whatever structure it thought it found isn't real (or has
   drifted away) -- and trading is paused rather than continuing on
   overfit confidence. This is the "self-doubt" breaker: it does not
   assume the model stays valid indefinitely just because it looked
   sound when it was built.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


def _clip01(x: float) -> float:
    return min(max(x, 1e-6), 1 - 1e-6)


@dataclass
class LearnerState:
    recent_brier: dict[str, deque] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    ensemble_brier: deque = field(default_factory=lambda: deque(maxlen=200))
    baseline_brier: deque = field(default_factory=lambda: deque(maxlen=200))


class Learner:
    def __init__(
        self,
        model_names: list[str],
        learning_rate: float = 0.1,
        calibration_window: int = 200,
        calibration_pause_threshold: float = 0.08,
        per_model_window: int = 300,
    ) -> None:
        self.learning_rate = learning_rate
        self.calibration_pause_threshold = calibration_pause_threshold
        self.per_model_window = per_model_window
        self.state = LearnerState(weights={name: 1.0 for name in model_names})
        self.state.recent_brier = {name: deque(maxlen=per_model_window) for name in model_names}
        self.state.ensemble_brier = deque(maxlen=calibration_window)
        self.state.baseline_brier = deque(maxlen=calibration_window)

    def weights_snapshot(self) -> dict[str, float]:
        return dict(self.state.weights)

    def observe_tick(
        self,
        per_model_p_over: dict[str, float | None],
        p_fair: float,
        outcome_over: bool,
    ) -> None:
        """
        Update per-model weights given each model's predicted P(over) this
        tick. Deliberately NOT a multiplicative running-product update --
        that would make a model's weight depend on how MANY times it has
        fired, not just how ACCURATE it's been (a model that only has
        enough samples to speak rarely, like a high Markov order early on,
        would otherwise end up looking artificially "better" purely for
        being quiet, since a monotonically-shrinking product only shrinks
        when a model actually speaks). Instead each model's weight is
        recomputed fresh each time from the MEAN Brier score over its own
        last `per_model_window` observations -- directly comparable across
        models regardless of how often each one has had an opinion.
        """
        y = 1.0 if outcome_over else 0.0
        for name, p in per_model_p_over.items():
            if p is None or name not in self.state.recent_brier:
                continue
            p = _clip01(p)
            brier = (p - y) ** 2
            self.state.recent_brier[name].append(brier)

        for name, dq in self.state.recent_brier.items():
            if dq:
                mean_brier = sum(dq) / len(dq)
                self.state.weights[name] = math.exp(-self.learning_rate * mean_brier)
            # else: no observations yet at all -- leave at the initial 1.0
        self._renormalize()

        baseline_brier = (p_fair - y) ** 2
        self.state.baseline_brier.append(baseline_brier)

    def observe_ensemble(self, p_predicted: float | None, outcome_over: bool) -> None:
        if p_predicted is None:
            return
        y = 1.0 if outcome_over else 0.0
        p = _clip01(p_predicted)
        self.state.ensemble_brier.append((p - y) ** 2)

    def _renormalize(self) -> None:
        # Keep the mean weight at 1.0 so `min_edge_sigma_multiple` etc. in the
        # ensemble config stay meaningful over long runs instead of drifting
        # toward 0 or infinity.
        vals = list(self.state.weights.values())
        if not vals:
            return
        mean = sum(vals) / len(vals)
        if mean <= 0:
            return
        for k in self.state.weights:
            self.state.weights[k] /= mean

    def should_pause(self) -> tuple[bool, str | None]:
        """
        Self-doubt circuit breaker: pause if the ensemble's recent live
        calibration is meaningfully worse than the trivial "always predict
        fair odds" baseline. Requires a minimum sample size before it will
        ever fire, so it can't trip on the first handful of noisy ticks.
        """
        eb, bb = self.state.ensemble_brier, self.state.baseline_brier
        if len(eb) < max(50, eb.maxlen // 2) or len(bb) < max(50, bb.maxlen // 2):
            return False, None
        ensemble_mean = sum(eb) / len(eb)
        baseline_mean = sum(bb) / len(bb)
        gap = ensemble_mean - baseline_mean
        if gap > self.calibration_pause_threshold:
            return True, (
                f"ensemble Brier {ensemble_mean:.4f} is {gap:.4f} worse than the "
                f"fair-odds baseline {baseline_mean:.4f} over the last {len(eb)} ticks -- "
                f"pausing until recent calibration recovers"
            )
        return False, None
