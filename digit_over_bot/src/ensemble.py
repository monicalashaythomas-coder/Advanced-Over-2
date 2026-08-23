"""
Ensemble decision layer.

Guiding principle (as given, plus the refinements noted below):

    Do not predict a digit with certainty. Estimate the probability
    distribution of the next last digit, detect whether the current state
    contains exploitable structure, and trade only when multiple independent
    models agree.

Refinements applied here:

1. Every model outputs an edge estimate WITH a standard error, never a bare
   point prediction -- "how sure" is carried through the whole pipeline, not
   just "which way".
2. Chi-square is used as a structure GATE (is the full 10-way distribution
   non-uniform at all?) plus a directional corroboration vote, not as a
   weighted quantitative estimate -- its residuals aren't safely convertible
   to a calibrated probability without extra assumptions, so it shouldn't be
   allowed to swing the combined edge by magnitude, only by direction.
3. The quantitative combination (z-score + Markov orders 1-3) is done by
   inverse-variance weighting, further scaled by a learned per-model
   reliability weight (see learner.py) -- a model that has been miscalibrated
   recently gets down-weighted automatically, it isn't just trusted forever
   because the math above looks right.
4. Multiple-testing discipline: because this evaluates several models across
   several symbols on every tick, a naive alpha=0.05 would produce frequent
   false positives over time purely from repeated testing. Default alpha is
   0.01 and, more importantly, a real trade additionally requires (a) an
   absolute edge floor (min_edge) so a "significant" but tiny effect can't
   fire on its own, (b) the combined edge to clear several multiples of its
   own combined standard error, and (c) a minimum number of independently-
   available models to agree in direction -- a single model alone, however
   significant, is never sufficient.
5. A live EV check against the ACTUAL quoted payout (not just the estimated
   fair probability) happens in bot.py right before firing, since the fair
   probability alone doesn't account for the broker's built-in margin.
6. A calibration circuit breaker (see learner.py) can veto trading outright
   if the ensemble's own recent predictions have been running poorly
   calibrated against realized outcomes -- the system distrusts itself
   rather than assuming past validity holds indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.stats.chi_square import ChiSquareResult, goodness_of_fit
from src.stats.markov import MarkovEstimate, MarkovLayer
from src.stats.zscore import ZScoreResult, over_proportion_test


@dataclass
class ModelVote:
    name: str
    available: bool
    edge: float | None = None          # p_hat - p_fair, signed
    se: float | None = None            # standard error of that edge estimate
    significant: bool = False          # this model's OWN significance test
    weight: float = 1.0                # learned reliability weight (learner.py)


@dataclass
class EnsembleResult:
    symbol: str
    barrier: int
    p_fair: float
    n: int
    votes: list[ModelVote]
    combined_edge: float | None
    combined_se: float | None
    agreement_count: int
    votes_available: int
    should_trade: bool
    reasons: list[str] = field(default_factory=list)
    chi2: ChiSquareResult | None = None
    zscore: ZScoreResult | None = None
    markov_per_order: dict[int, MarkovEstimate | None] = field(default_factory=dict)


class Ensemble:
    def __init__(
        self,
        barrier: int,
        alpha: float,
        min_edge: float,
        min_edge_sigma_multiple: float,
        min_models_agreeing: int,
        min_markov_state_count: float,
    ) -> None:
        self.barrier = barrier
        self.alpha = alpha
        self.min_edge = min_edge
        self.min_edge_sigma_multiple = min_edge_sigma_multiple
        self.min_models_agreeing = min_models_agreeing
        self.min_markov_state_count = min_markov_state_count

    def evaluate(
        self,
        symbol: str,
        digits_window: list[int],
        markov: MarkovLayer,
        model_weights: dict[str, float],
    ) -> EnsembleResult:
        p_fair = (9 - self.barrier) / 10.0
        n = len(digits_window)
        reasons: list[str] = []

        chi2 = goodness_of_fit(digits_window, self.alpha, self.barrier)
        zscore = over_proportion_test(digits_window, self.barrier, self.alpha)
        markov_orders = markov.predict_per_order(digits_window, self.min_markov_state_count)

        votes: list[ModelVote] = []

        # --- z-score: quantitative vote ---
        if zscore is not None:
            votes.append(
                ModelVote(
                    name="zscore",
                    available=True,
                    edge=zscore.edge,
                    se=(zscore.p_fair * (1 - zscore.p_fair) / zscore.n) ** 0.5,
                    significant=zscore.significant,
                    weight=model_weights.get("zscore", 1.0),
                )
            )
        else:
            votes.append(ModelVote(name="zscore", available=False))

        # --- markov orders 1-3: quantitative votes, only when individually well-sampled ---
        for order in sorted(markov_orders.keys()):
            est = markov_orders[order]
            name = f"markov_order_{order}"
            if est is None:
                votes.append(ModelVote(name=name, available=False))
                continue
            p_hat = est.p_over(self.barrier)
            se = max((p_hat * (1 - p_hat) / max(est.state_count, 1.0)) ** 0.5, 1e-6)
            edge = p_hat - p_fair
            # individual significance: edge clears ~2 of its own SE
            significant = abs(edge) >= 2.0 * se
            votes.append(
                ModelVote(
                    name=name,
                    available=True,
                    edge=edge,
                    se=se,
                    significant=significant,
                    weight=model_weights.get(name, 1.0),
                )
            )

        # --- inverse-variance combination of the quantitative votes ---
        quant_votes = [v for v in votes if v.available and v.edge is not None and v.se]
        if quant_votes:
            precision_sum = sum(v.weight / (v.se**2) for v in quant_votes)
            combined_edge = sum(v.weight * v.edge / (v.se**2) for v in quant_votes) / precision_sum
            combined_se = (1.0 / precision_sum) ** 0.5
        else:
            combined_edge = None
            combined_se = None

        # --- chi-square: gate + directional corroboration only, never magnitude ---
        chi2_vote_agrees = False
        if chi2 is not None:
            chi2_direction_over = chi2.over_direction_score > 0
            proposed_over = (combined_edge or 0.0) > 0
            chi2_vote_agrees = chi2.significant and (chi2_direction_over == proposed_over)

        # --- agreement count across everything with a defined direction ---
        proposed_over = (combined_edge or 0.0) > 0
        agreement_count = 0
        votes_available = 0
        for v in votes:
            if not v.available or v.edge is None:
                continue
            votes_available += 1
            same_direction = (v.edge > 0) == proposed_over
            if same_direction and v.significant:
                agreement_count += 1
        if chi2 is not None:
            votes_available += 1
            if chi2_vote_agrees:
                agreement_count += 1

        should_trade = False
        if combined_edge is None or combined_se is None:
            reasons.append("insufficient data for a combined estimate")
        elif not proposed_over:
            reasons.append("combined edge favors Under, not Over -- this bot only trades Over")
        elif combined_edge < self.min_edge:
            reasons.append(f"combined edge {combined_edge:.4f} below floor {self.min_edge:.4f}")
        elif combined_edge < self.min_edge_sigma_multiple * combined_se:
            reasons.append(
                f"combined edge {combined_edge:.4f} doesn't clear "
                f"{self.min_edge_sigma_multiple:g}x its own SE ({combined_se:.4f})"
            )
        elif agreement_count < self.min_models_agreeing:
            reasons.append(
                f"only {agreement_count}/{votes_available} independent models agree "
                f"(need {self.min_models_agreeing})"
            )
        else:
            should_trade = True
            reasons.append(
                f"{agreement_count}/{votes_available} models agree, "
                f"edge={combined_edge:.4f} ({combined_edge / combined_se:.1f} SE)"
            )

        return EnsembleResult(
            symbol=symbol,
            barrier=self.barrier,
            p_fair=p_fair,
            n=n,
            votes=votes,
            combined_edge=combined_edge,
            combined_se=combined_se,
            agreement_count=agreement_count,
            votes_available=votes_available,
            should_trade=should_trade,
            reasons=reasons,
            chi2=chi2,
            zscore=zscore,
            markov_per_order=markov_orders,
        )
