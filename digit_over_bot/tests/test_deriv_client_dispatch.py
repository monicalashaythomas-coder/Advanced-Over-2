"""
Regression test for the tick-handler / proposal-response deadlock.

Before the fix: _dispatch() awaited tick handlers inline, on the same loop
that is the only thing able to read a proposal()/buy() response off the
socket. A tick handler that calls proposal() (exactly what every real
trade signal does) would therefore block forever waiting on a response
that could only arrive via the very call it was blocking -- guaranteeing
a timeout on every single trade attempt, which is exactly what showed up
in production logs as "proposal failed: no response" on 100% of trades,
plus periodic 1011 keepalive-ping-timeout disconnects from the resulting
back-pressure.

This test reproduces that shape directly against DerivClient: a tick
handler that itself calls client.proposal(), fed by a hand-driven
_dispatch() loop standing in for _recv_pump (so no real network is
needed). It asserts the proposal call resolves well within its 15s
timeout instead of timing out.

Run with: python3 -m pytest tests/test_deriv_client_dispatch.py -q
"""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault("DERIV_APP_ID", "test")
os.environ.setdefault("DERIV_API_TOKEN", "test")

from src.deriv_client import DerivClient  # noqa: E402


class _RecordingWS:
    """Stands in for the real websockets connection. Records every frame
    the client tries to send so the test can react to it (e.g. answer a
    proposal request) exactly the way a real server would -- out-of-band,
    not from inside the handler that sent the request."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        import json
        self.sent.append(json.loads(raw))


async def _run() -> None:
    client = DerivClient(app_id="test", api_token="test", endpoint="wss://example.invalid")

    # Wire up just enough transport state to drive _send_with_id() and
    # _dispatch() without a real connect() (no network in this test).
    ws = _RecordingWS()
    client._ws = ws
    client._send_queue = asyncio.Queue()
    client._send_task = asyncio.create_task(client._send_pump())

    proposal_result: dict = {}

    async def tick_handler(tick: dict) -> None:
        # This is the part that deadlocked: calling proposal() from inside
        # a tick handler that _dispatch() is (or, pre-fix, was) awaiting
        # directly.
        resp = await client.proposal(
            symbol="R_10",
            contract_type="DIGITOVER",
            barrier=2,
            amount=1.0,
            currency="USD",
            duration_ticks=1,
        )
        proposal_result.update(resp)

    client._tick_handlers["R_10"] = tick_handler
    client._ensure_tick_worker("R_10")

    # Simulate _recv_pump: feed in a tick, then -- as a real server would,
    # asynchronously and without waiting for the client to "finish"
    # handling the tick -- feed in the proposal response once the request
    # actually hits the wire.
    await client._dispatch({"msg_type": "tick", "tick": {"symbol": "R_10", "quote": "1234.5"}})

    # The tick dispatch above must return immediately (it only enqueues),
    # leaving this coroutine free to keep "reading the socket" -- exactly
    # what _recv_pump needs to be able to do.
    for _ in range(200):
        if ws.sent:
            break
        await asyncio.sleep(0.01)
    assert ws.sent, "proposal request was never sent -- tick worker never ran"

    proposal_req = ws.sent[-1]
    assert proposal_req["proposal"] == 1
    assert proposal_req["underlying_symbol"] == "R_10"

    await client._dispatch(
        {
            "req_id": proposal_req["req_id"],
            "proposal": {"id": "abc123", "ask_price": 1.0, "payout": 1.95},
        }
    )

    # The proposal() call inside tick_handler should now resolve promptly
    # -- not after the 15s _send_with_id timeout.
    for _ in range(200):
        if proposal_result:
            break
        await asyncio.sleep(0.01)

    client._send_task.cancel()
    for task in client._tick_workers.values():
        task.cancel()

    assert proposal_result.get("id") == "abc123", (
        "tick handler's proposal() call never resolved -- recv pump is deadlocked"
    )


def test_tick_handler_proposal_call_does_not_deadlock() -> None:
    asyncio.run(asyncio.wait_for(_run(), timeout=5.0))


if __name__ == "__main__":
    test_tick_handler_proposal_call_does_not_deadlock()
    print("PASS")
