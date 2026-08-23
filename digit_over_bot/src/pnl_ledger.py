"""
In-memory trade ledger for console P&L reporting.

Not a source of truth (Supabase's digit_trades table remains that) -- this
exists purely to render a running P&L table to the logs after every settled
trade, so performance is visible at a glance without a dashboard. State is
lost on restart; that's fine, it rebuilds from the next trade onward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TradeRecord:
    symbol: str
    contract_id: int
    stake: float
    payout: float
    profit: float
    win: bool
    balance_after: float
    closed_at: str


@dataclass
class PnLLedger:
    starting_balance: float = 0.0
    max_display_rows: int = 15
    trades: list[TradeRecord] = field(default_factory=list)

    def record(
        self, symbol: str, contract_id: int, stake: float, payout: float, profit: float, balance_after: float
    ) -> TradeRecord:
        rec = TradeRecord(
            symbol=symbol,
            contract_id=contract_id,
            stake=stake,
            payout=payout,
            profit=profit,
            win=profit > 0,
            balance_after=balance_after,
            closed_at=datetime.now(timezone.utc).strftime("%H:%M:%S"),
        )
        self.trades.append(rec)
        return rec

    def render(self) -> str:
        total = len(self.trades)
        if total == 0:
            return "no trades settled yet"

        wins = sum(1 for t in self.trades if t.win)
        losses = total - wins
        win_rate = wins / total * 100.0
        total_pnl = sum(t.profit for t in self.trades)
        balance = self.trades[-1].balance_after

        per_symbol: dict[str, list[float]] = {}
        for t in self.trades:
            per_symbol.setdefault(t.symbol, []).append(t.profit)

        rows = self.trades[-self.max_display_rows :]
        header = (
            f"{'time':<9}{'symbol':<10}{'contract_id':<13}{'stake':>8}"
            f"{'payout':>8}{'profit':>9}{'result':>7}{'balance':>11}"
        )
        sep = "-" * len(header)
        lines = [sep, header, sep]
        if len(rows) < total:
            lines.append(f"... ({total - len(rows)} earlier trade(s) omitted) ...")
        for t in rows:
            lines.append(
                f"{t.closed_at:<9}{t.symbol:<10}{t.contract_id!s:<13}{t.stake:>8.2f}"
                f"{t.payout:>8.2f}{t.profit:>+9.2f}{('WIN' if t.win else 'LOSS'):>7}{t.balance_after:>11.2f}"
            )
        lines.append(sep)
        lines.append(
            f"TOTAL: {total} trades | {wins}W/{losses}L ({win_rate:.1f}% win rate) | "
            f"net P&L: {total_pnl:+.2f} | balance: {balance:.2f} "
            f"({balance - self.starting_balance:+.2f} vs starting {self.starting_balance:.2f})"
        )
        if len(per_symbol) > 1:
            by_symbol = ", ".join(
                f"{sym}: {sum(1 for p in pl if p > 0)}W/{sum(1 for p in pl if p <= 0)}L net {sum(pl):+.2f}"
                for sym, pl in sorted(per_symbol.items())
            )
            lines.append(f"by symbol: {by_symbol}")
        return "\n".join(lines)
