from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.deriv_client import DerivClient
from src.digit_buffer import RollingDigitBuffer
from src.ensemble import Ensemble, EnsembleResult
from src.executor import Executor
from src.learner import Learner
from src.risk import RiskManager
from src.stats.markov import MarkovLayer
from src.storage.supabase_client import SupabaseStore

logger = logging.getLogger("bot")

LOG_EVERY_N_EVALUATIONS = 20  # sample non-trade evaluations to Supabase to bound write volume


def last_digit(quote: float | str) -> int:
    s = f"{quote}"
    return int(s[-1])


@dataclass
class PendingPrediction:
    per_model_p_over: dict[str, float | None]
    p_fair: float
    combined_p: float | None
    result: EnsembleResult


class DigitOverBot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        t = settings.trading
        self.client = DerivClient(settings.deriv.app_id, settings.deriv.api_token, settings.deriv.endpoint)
        self.store = SupabaseStore(settings.supabase.url, settings.supabase.service_key, settings.supabase.enabled)
        self.risk = RiskManager(
            starting_balance=100.0,  # overwritten by an authorize/balance call in start()
            max_daily_loss_pct=t.max_daily_loss_pct,
            max_consecutive_losses=t.max_consecutive_losses,
            max_concurrent_open=t.max_concurrent_open,
            cooldown_after_trade_s=t.cooldown_after_trade_s,
        )
        self.executor = Executor(
            client=self.client,
            stake=t.stake,
            currency=t.currency,
            duration_ticks=t.duration_ticks,
            max_price_slippage_pct=t.max_price_slippage_pct,
        )
        self.ensemble = Ensemble(
            barrier=t.barrier,
            alpha=t.alpha,
            min_edge=t.min_edge,
            min_edge_sigma_multiple=t.min_edge_sigma_multiple,
            min_models_agreeing=t.min_models_agreeing,
            min_markov_state_count=t.min_markov_state_count,
        )

        self.buffers: dict[str, RollingDigitBuffer] = {}
        self.markov_layers: dict[str, MarkovLayer] = {}
        self.learners: dict[str, Learner] = {}
        self.pending: dict[str, PendingPrediction | None] = {}
        self._eval_counter: dict[str, int] = {}

        for symbol in t.symbols:
            self.buffers[symbol] = RollingDigitBuffer(t.buffer_size)
            self.markov_layers[symbol] = MarkovLayer(orders=tuple(t.markov_orders))
            model_names = ["zscore"] + [f"markov_order_{o}" for o in t.markov_orders]
            self.learners[symbol] = Learner(
                model_names,
                learning_rate=t.weight_learning_rate,
                calibration_window=t.calibration_window,
                calibration_pause_threshold=t.calibration_pause_threshold,
            )
            self.pending[symbol] = None
            self._eval_counter[symbol] = 0

    async def start(self) -> None:
        await self.client.connect()
        for symbol in self.settings.trading.symbols:
            await self.client.subscribe_ticks(symbol, self._tick_handler(symbol))
        logger.info("subscribed to: %s", ", ".join(self.settings.trading.symbols))

    def _tick_handler(self, symbol: str):
        async def handler(tick: dict[str, Any]) -> None:
            try:
                await self._on_tick(symbol, tick)
            except Exception:  # noqa: BLE001
                logger.exception("%s: unhandled error in _on_tick", symbol)
        return handler

    async def _on_tick(self, symbol: str, tick: dict[str, Any]) -> None:
        quote = tick.get("quote")
        if quote is None:
            return
        digit = last_digit(quote)
        buffer = self.buffers[symbol]
        markov = self.markov_layers[symbol]
        learner = self.learners[symbol]
        barrier = self.settings.trading.barrier

        # (1) Score the PREVIOUS tick's prediction against the digit that just
        # arrived -- this is the continuous relearning signal, generated on
        # every tick regardless of whether a trade fired.
        pending = self.pending.get(symbol)
        if pending is not None:
            outcome_over = digit > barrier
            learner.observe_tick(pending.per_model_p_over, pending.p_fair, outcome_over)
            learner.observe_ensemble(pending.combined_p, outcome_over)

        # (2) Update the Markov tables using the state as it was BEFORE this
        # digit, then push the new digit into the rolling window.
        history_before = buffer.window()
        markov.observe(history_before, digit)
        buffer.push(digit)

        # (3) Generate a fresh prediction for the NEXT digit.
        weights = learner.weights_snapshot()
        result = self.ensemble.evaluate(symbol, buffer.window(), markov, weights)

        self._eval_counter[symbol] += 1
        if result.should_trade or self._eval_counter[symbol] % LOG_EVERY_N_EVALUATIONS == 0:
            asyncio.create_task(self._log_evaluation(symbol, result))

        per_model_p_over: dict[str, float | None] = {}
        for v in result.votes:
            per_model_p_over[v.name] = (result.p_fair + v.edge) if (v.available and v.edge is not None) else None
        combined_p = (result.p_fair + result.combined_edge) if result.combined_edge is not None else None
        self.pending[symbol] = PendingPrediction(per_model_p_over, result.p_fair, combined_p, result)

        # (4) Self-doubt breaker: if recent live calibration is worse than the
        # trivial fair-odds baseline, keep learning but stop trading.
        paused, pause_reason = learner.should_pause()
        if paused:
            logger.warning("%s: trading paused -- %s", symbol, pause_reason)
            return

        if not result.should_trade:
            return

        can_trade, why_not = self.risk.can_trade(symbol)
        if not can_trade:
            logger.info("%s: signal fired but risk manager blocked it: %s", symbol, why_not)
            return

        attempt = await self.executor.try_execute(result)
        if not attempt.executed:
            logger.info("%s: signal fired but did not execute: %s", symbol, attempt.reason)
            return

        self.risk.record_open(symbol)
        await self.store.insert(
            "digit_trades",
            {
                "symbol": symbol,
                "contract_id": attempt.contract_id,
                "barrier": barrier,
                "stake": attempt.stake,
                "payout": attempt.payout,
                "p_estimate": attempt.conservative_p,
                "combined_edge": result.combined_edge,
                "agreement_count": result.agreement_count,
                "votes_available": result.votes_available,
                "reasons": "; ".join(result.reasons),
                "opened_epoch": tick.get("epoch"),
            },
        )
        await self.client.subscribe_contract(attempt.contract_id, self._settlement_handler(symbol, attempt.contract_id))

    def _settlement_handler(self, symbol: str, contract_id: int):
        async def handler(poc: dict[str, Any]) -> None:
            if not poc.get("is_sold"):
                return
            profit = float(poc.get("profit", 0.0))
            self.risk.record_settlement(symbol, profit)
            self.client.unsubscribe_contract(contract_id)
            await self.store.update(
                "digit_trades",
                match={"contract_id": contract_id},
                patch={
                    "profit": profit,
                    "win": profit > 0,
                    "closed_epoch": poc.get("date_expiry") or poc.get("sell_time"),
                },
            )
            logger.info("%s: contract %s settled, profit=%.2f, balance=%.2f", symbol, contract_id, profit, self.risk.state.balance)
        return handler

    async def _log_evaluation(self, symbol: str, result: EnsembleResult) -> None:
        markov_edges = {}
        for order, est in result.markov_per_order.items():
            markov_edges[order] = (est.p_over(result.barrier) - result.p_fair) if est is not None else None
        await self.store.insert(
            "digit_ensemble_log",
            {
                "symbol": symbol,
                "n": result.n,
                "p_fair": result.p_fair,
                "combined_edge": result.combined_edge,
                "combined_se": result.combined_se,
                "agreement_count": result.agreement_count,
                "votes_available": result.votes_available,
                "should_trade": result.should_trade,
                "chi2_stat": result.chi2.chi2_stat if result.chi2 else None,
                "chi2_p": result.chi2.p_value if result.chi2 else None,
                "zscore_z": result.zscore.z if result.zscore else None,
                "zscore_p": result.zscore.p_value if result.zscore else None,
                "markov1_edge": markov_edges.get(1),
                "markov2_edge": markov_edges.get(2),
                "markov3_edge": markov_edges.get(3),
                "reasons": "; ".join(result.reasons),
            },
        )

    async def run_forever(self) -> None:
        await self.start()
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            await self.client.close()
            await self.store.close()
