"""
Turns an ensemble "should_trade" result into an actual Deriv contract
purchase -- but only after re-checking expected value against the ACTUAL
quoted payout, using a conservative (not point-estimate) probability.

The ensemble's `combined_edge` is an estimate with a standard error attached;
using the raw point estimate for the EV check would let the bot fire on
the optimistic tail of its own uncertainty. Instead the EV check uses
`p_fair + combined_edge - combined_se`, one standard error worse than the
point estimate, floored at p_fair. If a trade doesn't clear EV even under
that more conservative assumption, it doesn't fire.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.deriv_client import DerivClient
from src.ensemble import EnsembleResult

logger = logging.getLogger("executor")


@dataclass
class TradeAttempt:
    executed: bool
    reason: str
    contract_id: int | None = None
    stake: float | None = None
    payout: float | None = None
    conservative_p: float | None = None
    expected_value: float | None = None
    duration_ticks: int | None = None


class Executor:
    def __init__(
        self,
        client: DerivClient,
        stake: float,
        currency: str,
        duration_ticks: int,
        max_price_slippage_pct: float,
        min_ev_margin: float = 0.0,
    ) -> None:
        self.client = client
        self.stake = stake
        self.currency = currency
        self.duration_ticks = duration_ticks
        self.max_price_slippage_pct = max_price_slippage_pct
        self.min_ev_margin = min_ev_margin

    async def try_execute(
        self, result: EnsembleResult, duration_ticks: int | None = None, stake: float | None = None
    ) -> TradeAttempt:
        if result.combined_edge is None or result.combined_se is None:
            return TradeAttempt(executed=False, reason="no combined estimate")

        duration = duration_ticks if duration_ticks is not None else self.duration_ticks
        trade_stake = stake if stake is not None else self.stake
        conservative_p = max(result.p_fair, result.p_fair + result.combined_edge - result.combined_se)

        try:
            proposal = await self.client.proposal(
                symbol=result.symbol,
                contract_type="DIGITOVER",
                barrier=result.barrier,
                amount=trade_stake,
                currency=self.currency,
                duration_ticks=duration,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: proposal failed: %s", result.symbol, exc)
            return TradeAttempt(executed=False, reason=f"proposal failed: {exc}", duration_ticks=duration)

        payout = float(proposal["payout"])
        ask_price = float(proposal["ask_price"])
        proposal_id = proposal["id"]

        # Expected value of a single contract: p * payout - stake
        expected_value = conservative_p * payout - ask_price
        if expected_value <= self.min_ev_margin * ask_price:
            return TradeAttempt(
                executed=False,
                reason=(
                    f"EV {expected_value:.4f} (conservative p={conservative_p:.3f}, "
                    f"payout={payout:.2f}, stake={ask_price:.2f}) does not clear margin"
                ),
                payout=payout,
                conservative_p=conservative_p,
                expected_value=expected_value,
                duration_ticks=duration,
            )

        # Deriv rejects buy prices with more than 2 decimal places
        # (error code InvalidPrice). ask_price is always a clean 2-decimal
        # money value, but multiplying by a slippage percentage routinely
        # produces 3-4 decimal places (e.g. stake=2.50 * 1.05 = 2.625) --
        # round the final price back down to cents before sending it, or
        # every buy past a non-"nice" stake (any martingale step beyond the
        # base stake, in particular) fails outright.
        buy_price = round(ask_price * (1 + self.max_price_slippage_pct / 100), 2)
        try:
            buy_resp = await self.client.buy(proposal_id, price=buy_price)
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: buy failed: %s", result.symbol, exc)
            return TradeAttempt(executed=False, reason=f"buy failed: {exc}", duration_ticks=duration)

        contract_id = int(buy_resp["contract_id"])
        logger.info(
            "%s: BOUGHT DIGITOVER barrier=%d duration=%dt contract_id=%s stake=%.2f payout=%.2f "
            "conservative_p=%.3f EV=%.4f",
            result.symbol, result.barrier, duration, contract_id, ask_price, payout, conservative_p, expected_value,
        )
        return TradeAttempt(
            executed=True,
            reason="executed",
            contract_id=contract_id,
            stake=ask_price,
            payout=payout,
            conservative_p=conservative_p,
            expected_value=expected_value,
            duration_ticks=duration,
        )
