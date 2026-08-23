"""
Risk manager: independent of the statistical ensemble on purpose.

Even a well-calibrated ensemble should never be trusted to also manage its
own exposure -- these breakers fire on realized P&L and position state, not
on anything the ensemble believes about digit structure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SymbolRiskState:
    open_contracts: int = 0
    consecutive_losses: int = 0
    last_trade_ts: float = 0.0
    halted: bool = False
    halt_reason: str | None = None


@dataclass
class RiskState:
    starting_balance: float
    balance: float
    daily_pnl: float = 0.0
    per_symbol: dict[str, SymbolRiskState] = field(default_factory=dict)


class RiskManager:
    def __init__(
        self,
        starting_balance: float,
        max_daily_loss_pct: float,
        max_consecutive_losses: int,
        max_concurrent_open: int,
        cooldown_after_trade_s: float,
    ) -> None:
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_concurrent_open = max_concurrent_open
        self.cooldown_after_trade_s = cooldown_after_trade_s
        self.state = RiskState(starting_balance=starting_balance, balance=starting_balance)

    def _sym(self, symbol: str) -> SymbolRiskState:
        return self.state.per_symbol.setdefault(symbol, SymbolRiskState())

    def can_trade(self, symbol: str) -> tuple[bool, str | None]:
        sym = self._sym(symbol)

        daily_loss_pct = -self.state.daily_pnl / self.state.starting_balance * 100.0
        if daily_loss_pct >= self.max_daily_loss_pct:
            return False, f"daily loss {daily_loss_pct:.1f}% >= limit {self.max_daily_loss_pct:.1f}%"

        if sym.consecutive_losses >= self.max_consecutive_losses:
            return False, f"{sym.consecutive_losses} consecutive losses on {symbol}"

        if sym.open_contracts >= self.max_concurrent_open:
            return False, f"{sym.open_contracts} contract(s) already open on {symbol}"

        if self.cooldown_after_trade_s > 0:
            elapsed = time.time() - sym.last_trade_ts
            if elapsed < self.cooldown_after_trade_s:
                return False, f"cooldown: {self.cooldown_after_trade_s - elapsed:.1f}s remaining"

        return True, None

    def record_open(self, symbol: str) -> None:
        sym = self._sym(symbol)
        sym.open_contracts += 1
        sym.last_trade_ts = time.time()

    def record_settlement(self, symbol: str, profit: float) -> None:
        sym = self._sym(symbol)
        sym.open_contracts = max(0, sym.open_contracts - 1)
        self.state.balance += profit
        self.state.daily_pnl += profit
        if profit < 0:
            sym.consecutive_losses += 1
        else:
            sym.consecutive_losses = 0

    def reset_daily(self) -> None:
        self.state.daily_pnl = 0.0
        for sym in self.state.per_symbol.values():
            sym.consecutive_losses = 0
