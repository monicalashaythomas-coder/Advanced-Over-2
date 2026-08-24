from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from src.config import Settings
from src.deriv_client import DerivClient
from src.digit_buffer import RollingDigitBuffer
from src.duration_selector import select_duration
from src.ensemble import Ensemble, EnsembleResult, describe
from src.executor import Executor, TradeAttempt
from src.learner import Learner
from src.pnl_ledger import PnLLedger
from src.risk import RiskManager
from src.stats.markov import MarkovLayer
from src.storage.supabase_client import SupabaseStore

logger = logging.getLogger("bot")

LOG_EVERY_N_EVALUATIONS = 20  # sample non-trade evaluations to Supabase to bound write volume
MARKOV_SAVE_INTERVAL_S = 60  # how often cumulative markov counts get flushed to Supabase


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

        self.pnl = PnLLedger(starting_balance=100.0)  # overwritten once connect() reports the real balance

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

        self._markov_save_task: asyncio.Task | None = None

    async def start(self) -> None:
        await self.client.connect()
        if self.client.initial_balance:
            self.risk.state.starting_balance = self.client.initial_balance
            self.risk.state.balance = self.client.initial_balance
            self.pnl.starting_balance = self.client.initial_balance
        for symbol in self.settings.trading.symbols:
            had_persisted_state = await self._load_markov_state(symbol)
            if not had_persisted_state:
                await self._seed_markov_state(symbol)
        for symbol in self.settings.trading.symbols:
            await self.client.subscribe_ticks(symbol, self._tick_handler(symbol))
        logger.info("subscribed to: %s", ", ".join(self.settings.trading.symbols))

    async def _load_markov_state(self, symbol: str) -> bool:
        """Restore cumulative markov_order_2/3 counts from Supabase so they
        don't start from zero on every restart -- without this, those orders
        can take tens of thousands of ticks per symbol to warm back up.
        Returns True if any persisted state was found and restored."""
        rows = await self.store.select("digit_markov_state", {"symbol": symbol})
        if not rows:
            logger.info("%s: no persisted markov state found (fresh start for this symbol)", symbol)
            return False
        state = {row["order_"]: row["counts"] for row in rows}
        self.markov_layers[symbol].import_state(state)
        logger.info("%s: restored markov state for orders %s", symbol, sorted(state.keys()))
        return True

    async def _seed_markov_state(self, symbol: str, history_count: int = 5000) -> None:
        """One-time cold-start seed: replay recent tick history through the
        same observe()/push() path live ticks use, so markov_order_1/2 (and
        to a lesser extent order_3) don't have to wait through minutes-to-
        hours of live ticks before becoming available. Only runs when
        Supabase had no persisted state at all for this symbol -- a restart
        with real persisted history is never overwritten by a replay."""
        try:
            prices = await self.client.ticks_history(symbol, count=history_count)
        except Exception:
            logger.exception("%s: markov seed failed, continuing without it", symbol)
            return
        markov = self.markov_layers[symbol]
        buffer = self.buffers[symbol]
        for price in prices:
            digit = last_digit(price)
            markov.observe(buffer.window(), digit)
            buffer.push(digit)
        logger.info("%s: seeded markov state from %d historical ticks", symbol, len(prices))

    async def _save_markov_state(self, symbol: str) -> None:
        state = self.markov_layers[symbol].export_state()
        saved_orders = []
        for order, counts in state.items():
            if not counts:
                continue
            await self.store.upsert(
                "digit_markov_state",
                {"symbol": symbol, "order_": order, "counts": counts},
                on_conflict="symbol,order_",
            )
            saved_orders.append(order)
        logger.info(
            "%s: markov save cycle -- wrote orders %s, skipped empty %s",
            symbol, saved_orders, sorted(set(state.keys()) - set(saved_orders)),
        )

    async def _markov_save_loop(self) -> None:
        while True:
            await asyncio.sleep(MARKOV_SAVE_INTERVAL_S)
            for symbol in self.settings.trading.symbols:
                await self._save_markov_state(symbol)

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

        # Console visibility into what every layer/model is saying and
        # whether they agree, on every tick by default (LOG_EVERY_EVALUATION).
        # Trade fires always log regardless of that setting.
        if self.settings.log_every_evaluation or result.should_trade:
            logger.info(describe(result))

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

        t = self.settings.trading
        duration_choice = select_duration(
            buffer.window(),
            t.duration_candidates,
            barrier,
            n_samples=t.duration_mc_samples,
            block_size=t.duration_mc_block_size,
        )
        chosen_duration = duration_choice.duration_ticks if duration_choice is not None else None
        if duration_choice is not None:
            logger.info(
                "%s: MC duration select -> %dt (p_win=%.3f) candidates=%s",
                symbol, duration_choice.duration_ticks, duration_choice.win_prob,
                {k: round(v, 3) for k, v in duration_choice.per_candidate.items()},
            )
        else:
            logger.info("%s: MC duration select -> window too short, falling back to DURATION_TICKS", symbol)

        attempt = await self.executor.try_execute(result, duration_ticks=chosen_duration)
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
                "duration_ticks": attempt.duration_ticks,
            },
        )
        await self.client.subscribe_contract(attempt.contract_id, self._settlement_handler(symbol, attempt))

    def _settlement_handler(self, symbol: str, attempt: TradeAttempt):
        contract_id = attempt.contract_id

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
            logger.info(
                "%s: contract %s settled, profit=%.2f, balance=%.2f",
                symbol, contract_id, profit, self.risk.state.balance,
            )
            self.pnl.record(
                symbol=symbol,
                contract_id=contract_id,
                stake=attempt.stake or 0.0,
                payout=attempt.payout or 0.0,
                profit=profit,
                balance_after=self.risk.state.balance,
            )
            logger.info("P&L\n%s", self.pnl.render())
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
        self._markov_save_task = asyncio.create_task(self._markov_save_loop())
        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            if self._markov_save_task is not None:
                self._markov_save_task.cancel()
            for symbol in self.settings.trading.symbols:
                await self._save_markov_state(symbol)
            await self.client.close()
            await self.store.close()
