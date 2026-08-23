"""
Drives DigitOverBot._on_tick directly over a synthetic, strongly-biased tick
stream, with a fake Deriv client standing in for the real websocket -- no
network access required or used.

Run with: python3 -m tests.smoke_test
"""
from __future__ import annotations

import asyncio
import itertools
import os
import random

os.environ.setdefault("DERIV_APP_ID", "test")
os.environ.setdefault("DERIV_API_TOKEN", "test")
os.environ.setdefault("SUPABASE_ENABLED", "false")
os.environ.setdefault("SYMBOLS", "TEST_SYM")
os.environ.setdefault("MIN_MODELS_AGREEING", "3")

from src.bot import DigitOverBot  # noqa: E402
from src.config import SETTINGS  # noqa: E402


class FakeDerivClient:
    """Stands in for src.deriv_client.DerivClient -- no real websocket."""

    def __init__(self) -> None:
        self.contract_ids = itertools.count(1000)
        self.bought = []
        self.subscribed_contracts = {}

    async def proposal(self, symbol, contract_type, barrier, amount, currency, duration_ticks):
        # Fair-ish payout for "Over 2": ~70% win prob, modest payout.
        return {"id": "proposal-1", "ask_price": amount, "payout": amount * 1.38}

    async def buy(self, proposal_id, price):
        cid = next(self.contract_ids)
        self.bought.append(cid)
        return {"contract_id": cid}

    async def subscribe_contract(self, contract_id, handler):
        self.subscribed_contracts[contract_id] = handler
        # Immediately "settle" the contract as a win, synchronously enough for the test.
        await handler({"is_sold": True, "profit": 0.38, "date_expiry": 12345})

    def unsubscribe_contract(self, contract_id):
        self.subscribed_contracts.pop(contract_id, None)

    async def close(self):
        pass


async def main() -> None:
    bot = DigitOverBot(SETTINGS)
    fake_client = FakeDerivClient()
    bot.client = fake_client
    bot.executor.client = fake_client

    symbol = SETTINGS.trading.symbols[0]
    rng = random.Random(11)
    ticks_processed = 0
    for i in range(4000):
        if rng.random() < 0.85:
            digit = rng.choice([3, 4, 5, 6, 7, 8, 9])
        else:
            digit = rng.choice([0, 1, 2])
        quote = f"1000.{digit}"
        await bot._on_tick(symbol, {"quote": quote, "epoch": 1_800_000_000 + i})
        ticks_processed += 1

    print(f"Ticks processed: {ticks_processed}")
    print(f"Trades executed: {len(fake_client.bought)}")
    print(f"Final balance:    {bot.risk.state.balance:.2f}")
    print(f"Learner weights:  {bot.learners[symbol].weights_snapshot()}")

    assert ticks_processed == 4000
    assert len(fake_client.bought) > 0, "expected at least one trade on a strong, stable bias"
    print("\nPIPELINE RAN WITHOUT ERRORS ✅")


if __name__ == "__main__":
    asyncio.run(main())
