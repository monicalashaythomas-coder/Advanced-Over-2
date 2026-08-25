"""
Per-symbol martingale staking.

Off by default -- this bot's default is flat staking (see README), on the
philosophy that the ensemble's edge shouldn't need bet-sizing tricks to pay
for itself. This module exists for when MARTINGALE_ENABLED is explicitly
turned on.

Mirrors the design already used in the expiryrange-quiet-bot: a step
counter per symbol, multiplying the base stake by `factor` for each
consecutive loss, resetting to step 0 on a win OR once `max_steps` is
reached (whichever comes first). Driven purely by the step counter, never
by account balance.

State is in-memory only and resets on process restart -- same as
expiryrange-quiet-bot's martingale. If you want it to survive a Railway
redeploy mid-sequence, it would need a Supabase table the same way
digit_markov_state is persisted; not done here since it wasn't asked for.
"""
from __future__ import annotations


class MartingaleStaking:
    def __init__(self, base_stake: float, factor: float, max_steps: int, enabled: bool) -> None:
        self.base_stake = base_stake
        self.factor = factor
        self.max_steps = max_steps
        self.enabled = enabled
        self._steps: dict[str, int] = {}

    def step_for(self, symbol: str) -> int:
        return self._steps.get(symbol, 0)

    def stake_for(self, symbol: str) -> float:
        if not self.enabled:
            return self.base_stake
        step = self._steps.get(symbol, 0)
        return round(self.base_stake * (self.factor ** step), 2)

    def record_result(self, symbol: str, win: bool) -> None:
        """Call once per settled contract with the actual win/loss outcome."""
        if not self.enabled:
            return
        if win:
            self._steps[symbol] = 0
            return
        step = self._steps.get(symbol, 0) + 1
        if step >= self.max_steps:
            step = 0
        self._steps[symbol] = step
