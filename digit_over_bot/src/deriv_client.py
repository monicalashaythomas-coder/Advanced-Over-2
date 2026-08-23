"""
Minimal async Deriv WebSocket API client.

Covers exactly what this bot needs: authorize, tick subscription, proposal,
buy, and proposal_open_contract (settlement) subscription. Built directly
against the documented v3 API (wss://ws.derivws.com/websockets/v3?app_id=...,
request/response correlated by req_id, contract_type "DIGITOVER" with an
integer barrier). Verify against https://developers.deriv.com before trusting
this in a live account -- the API does evolve, and this was written from
current docs, not from a live-tested session (no network path to Deriv from
the environment this was built in).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
from typing import Any, Awaitable, Callable

import websockets

logger = logging.getLogger("deriv_client")

TickHandler = Callable[[dict[str, Any]], Awaitable[None]]
ContractHandler = Callable[[dict[str, Any]], Awaitable[None]]


class DerivApiError(RuntimeError):
    pass


class DerivClient:
    def __init__(self, app_id: str, api_token: str, endpoint: str) -> None:
        self.app_id = app_id
        self.api_token = api_token
        self.endpoint = endpoint
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._req_id = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_handlers: dict[str, TickHandler] = {}
        self._contract_handlers: dict[int, ContractHandler] = {}
        self._listen_task: asyncio.Task | None = None
        self._ka_task: asyncio.Task | None = None
        self._closing = False

    async def connect(self) -> None:
        url = f"{self.endpoint}?app_id={self.app_id}"
        self._ws = await websockets.connect(url, ping_interval=None)
        self._listen_task = asyncio.create_task(self._listen())
        self._ka_task = asyncio.create_task(self._heartbeat())
        if self.api_token:
            await self.authorize()

    async def close(self) -> None:
        self._closing = True
        for task in (self._listen_task, self._ka_task):
            if task:
                task.cancel()
        if self._ws:
            await self._ws.close()

    async def _heartbeat(self) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                if self._ws:
                    await self._send({"ping": 1})
        except asyncio.CancelledError:
            pass

    async def _listen(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                await self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except websockets.ConnectionClosed:
            if not self._closing:
                logger.warning("Deriv websocket closed unexpectedly; reconnect logic should handle this")

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("req_id")
        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                fut.set_result(msg)
            return

        msg_type = msg.get("msg_type")
        if msg_type == "tick":
            tick = msg.get("tick", {})
            symbol = tick.get("symbol")
            handler = self._tick_handlers.get(symbol)
            if handler:
                await handler(tick)
        elif msg_type == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            contract_id = poc.get("contract_id")
            handler = self._contract_handlers.get(contract_id)
            if handler:
                await handler(poc)
        elif msg.get("error"):
            logger.error("Deriv API error (unsolicited): %s", msg["error"])

    async def _send(self, payload: dict[str, Any], expect_response: bool = False) -> dict[str, Any] | None:
        if self._ws is None:
            raise DerivApiError("not connected")
        req_id = next(self._req_id)
        payload = {**payload, "req_id": req_id}
        fut: asyncio.Future | None = None
        if expect_response:
            fut = asyncio.get_event_loop().create_future()
            self._pending[req_id] = fut
        await self._ws.send(json.dumps(payload))
        if fut is None:
            return None
        try:
            return await asyncio.wait_for(fut, timeout=15.0)
        finally:
            self._pending.pop(req_id, None)

    async def authorize(self) -> dict[str, Any]:
        resp = await self._send({"authorize": self.api_token}, expect_response=True)
        if resp and resp.get("error"):
            raise DerivApiError(f"authorize failed: {resp['error']}")
        return resp or {}

    async def subscribe_ticks(self, symbol: str, handler: TickHandler) -> None:
        self._tick_handlers[symbol] = handler
        resp = await self._send({"ticks": symbol, "subscribe": 1}, expect_response=True)
        if resp and resp.get("error"):
            raise DerivApiError(f"subscribe_ticks({symbol}) failed: {resp['error']}")

    async def proposal(
        self, symbol: str, contract_type: str, barrier: int, amount: float, currency: str, duration_ticks: int
    ) -> dict[str, Any]:
        resp = await self._send(
            {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": currency,
                "duration": duration_ticks,
                "duration_unit": "t",
                "symbol": symbol,
                "barrier": str(barrier),
            },
            expect_response=True,
        )
        if resp is None or resp.get("error"):
            raise DerivApiError(f"proposal failed: {resp.get('error') if resp else 'no response'}")
        return resp["proposal"]

    async def buy(self, proposal_id: str, price: float) -> dict[str, Any]:
        resp = await self._send({"buy": proposal_id, "price": price}, expect_response=True)
        if resp is None or resp.get("error"):
            raise DerivApiError(f"buy failed: {resp.get('error') if resp else 'no response'}")
        return resp["buy"]

    async def subscribe_contract(self, contract_id: int, handler: ContractHandler) -> None:
        self._contract_handlers[contract_id] = handler
        resp = await self._send(
            {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1},
            expect_response=True,
        )
        if resp and resp.get("error"):
            raise DerivApiError(f"subscribe_contract({contract_id}) failed: {resp['error']}")

    def unsubscribe_contract(self, contract_id: int) -> None:
        self._contract_handlers.pop(contract_id, None)
