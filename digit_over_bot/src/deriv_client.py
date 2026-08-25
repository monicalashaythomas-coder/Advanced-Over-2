"""
Async Deriv WebSocket API client.

Ported from the connection layer used in the expiryrange-quiet-bot
project (bot.py's DerivClient), which is live-tested against a real
account. That port was needed because this client's old approach --
connect straight to wss://ws.derivws.com/websockets/v3?app_id=... and
send {"authorize": api_token} as the first message -- was failing with
"server rejected WebSocket connection: HTTP 401" at the handshake
itself, before any authorize message could even be sent.

The fix is a different auth model entirely (Deriv's newer Options API):
the api_token is exchanged over REST for a personalized, already-
authenticated WebSocket URL (an "OTP" URL scoped to one account), and
the client connects to *that* URL directly. There is no separate
{"authorize": ...} step afterwards -- the URL itself carries the auth,
so the first message sent (a balance check) doubles as both the
connectivity check and the auth check.

Covers exactly what this bot needs: connect (REST OTP + WS), tick
subscription, proposal, buy, and proposal_open_contract (settlement)
subscription. Verify against https://developers.deriv.com before
trusting this in a live account if Deriv's API surface has moved on --
this was last confirmed working against the reference bot, not
re-verified from scratch here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, ConnectionClosedOK

logger = logging.getLogger("deriv_client")

TickHandler = Callable[[dict[str, Any]], Awaitable[None]]
ContractHandler = Callable[[dict[str, Any]], Awaitable[None]]

REST_BASE = os.environ.get("DERIV_REST_BASE", "https://api.derivws.com")


class DerivApiError(RuntimeError):
    pass


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class DerivClient:
    def __init__(
        self,
        app_id: str,
        api_token: str,
        endpoint: str,
        account_id: str = "",
        use_real_account: bool | None = None,
        ws_ping_interval: float = 30,
    ) -> None:
        self.app_id = app_id
        self.api_token = api_token
        # Legacy WS endpoint -- no longer connected to directly (see module
        # docstring), kept only so the constructor stays call-compatible
        # with existing callers that pass settings.deriv.endpoint.
        self.endpoint = endpoint
        self.account_id = account_id or os.environ.get("DERIV_ACCOUNT_ID", "") or None
        self.use_real_account = (
            use_real_account if use_real_account is not None else _env_bool("DERIV_USE_REAL", False)
        )
        self.ws_ping_interval = ws_ping_interval

        self.ws_url: str | None = None
        self._ws = None
        self._send_queue: asyncio.Queue | None = None
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._req_id_counter = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._tick_handlers: dict[str, TickHandler] = {}
        self._contract_handlers: dict[int, ContractHandler] = {}
        self.initial_balance: float = 0.0
        self._closing = False
        self._reconnecting = False

        # Per-symbol tick queues + worker tasks -- see _dispatch()/_tick_worker()
        # for why ticks must never be awaited directly from the recv pump.
        self._tick_queues: dict[str, asyncio.Queue] = {}
        self._tick_workers: dict[str, asyncio.Task] = {}

    # ---- REST: token -> personalized, pre-authenticated WS URL ----

    def _rest_request(self, path: str, method: str = "GET") -> dict:
        req = urllib.request.Request(
            f"{REST_BASE}{path}",
            method=method,
            headers={
                "Deriv-App-ID": self.app_id,
                "Authorization": f"Bearer {self.api_token}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DerivApiError(f"HTTP {exc.code} from {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise DerivApiError(f"Network error calling {path}: {exc.reason}") from exc

    def _resolve_account_id(self) -> str:
        payload = self._rest_request("/trading/v1/options/accounts")
        accounts = payload.get("data") or payload.get("accounts") or []
        if not accounts:
            raise DerivApiError("No accounts returned for this API token")
        wanted = "real" if self.use_real_account else "demo"
        for acc in accounts:
            t = str(acc.get("type") or acc.get("account_type") or "").lower()
            if t == wanted:
                return acc.get("account_id") or acc.get("id")
        first = accounts[0]
        return first.get("account_id") or first.get("id")

    def _fetch_ws_url(self) -> str:
        if not self.account_id:
            self.account_id = self._resolve_account_id()
        payload = self._rest_request(f"/trading/v1/options/accounts/{self.account_id}/otp", method="POST")
        url = (payload.get("data") or {}).get("url")
        if not url:
            raise DerivApiError(f"OTP response missing url: {payload}")
        return url

    # ---- connection lifecycle ----

    async def _teardown_transport(self) -> None:
        """Cancel any live pump tasks and close any live socket before a
        (re)connect. Without this, a reconnect retry that opens a new
        websocket but then fails at resubscribe leaves the previous
        socket's recv loop running forever against a stale connection --
        `async for raw in self._ws` captures that socket once and keeps
        reading it even after self._ws is reassigned, so failed retries
        silently accumulate extra live sessions on the account."""
        for task in (self._send_task, self._recv_task):
            if task and not task.done():
                task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None

    async def connect(self) -> None:
        await self._teardown_transport()

        loop = asyncio.get_event_loop()
        self.ws_url = await loop.run_in_executor(None, self._fetch_ws_url)

        safe = self.ws_url.split("?")[0]
        logger.info("connecting -> %s (account %s)", safe, self.account_id)
        self._ws = await websockets.connect(
            self.ws_url,
            ping_interval=self.ws_ping_interval,
            ping_timeout=20,
            close_timeout=10,
        )
        self._closing = False
        self._send_queue = asyncio.Queue()
        self._req_id_counter = 1
        self._pending = {}
        self._recv_task = asyncio.create_task(self._recv_pump(), name="deriv_recv_pump")
        self._send_task = asyncio.create_task(self._send_pump(), name="deriv_send_pump")

        # The OTP URL is already pre-authenticated -- no separate
        # {"authorize": token} message is needed or accepted here. A
        # balance check both confirms the connection is genuinely
        # authenticated and gives us a starting balance for free.
        resp = await self._send_with_id({"balance": 1}, timeout=15)
        if resp is None or resp.get("error"):
            err = (resp or {}).get("error", {}).get("message", "timeout")
            raise DerivApiError(f"post-connect balance check failed: {err}")
        bal = resp.get("balance", {})
        self.initial_balance = float(bal.get("balance", 0) or 0)
        logger.info(
            "connected | account %s | balance %.2f %s",
            self.account_id, self.initial_balance, bal.get("currency", ""),
        )

    async def close(self) -> None:
        self._closing = True
        await self._teardown_transport()
        self._fail_pending("connection closed")
        for task in self._tick_workers.values():
            task.cancel()

    # ---- I/O pumps ----
    # Writes go through a single queue/task so concurrent callers (e.g. a
    # proposal fetch racing a buy) never write to the socket at the same
    # time. Reads are similarly centralized in one task and fan out by
    # req_id (to a waiting Future) or msg_type (to a registered tick/
    # contract handler) -- this is what lets subscribe_ticks callbacks and
    # request/response calls like proposal()/buy() share one connection
    # safely.

    async def _send_pump(self) -> None:
        while True:
            data, fut = await self._send_queue.get()
            try:
                await self._ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_queue.task_done()

    async def _recv_pump(self) -> None:
        try:
            async for raw in self._ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                await self._dispatch(msg)
        except asyncio.CancelledError:
            pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            self._fail_pending("connection closed")
            if not self._closing:
                logger.warning("Deriv websocket closed unexpectedly; reconnecting")
                asyncio.create_task(self._reconnect(), name="deriv_reconnect")
        except Exception as exc:
            logger.error("recv pump error: %s", exc)
            self._fail_pending(str(exc))
            if not self._closing:
                asyncio.create_task(self._reconnect(), name="deriv_reconnect")

    async def _reconnect(self) -> None:
        """Re-establish the websocket after an unexpected close and
        resubscribe everything the caller had registered -- tick streams
        and open-contract watches -- so the bot doesn't go silently dead
        after a routine keepalive/ping-timeout disconnect."""
        if self._reconnecting or self._closing:
            return
        self._reconnecting = True
        try:
            symbols = list(self._tick_handlers.keys())
            contract_ids = list(self._contract_handlers.keys())
            delay = 2.0
            for attempt in range(1, 6):
                if self._closing:
                    return
                await asyncio.sleep(delay)
                try:
                    logger.info("reconnect attempt %d/5", attempt)
                    await self.connect()
                    for symbol in symbols:
                        resp = await self._send_with_id({"ticks": symbol, "subscribe": 1})
                        if resp is None or resp.get("error"):
                            raise DerivApiError(f"resubscribe ticks({symbol}) failed: {(resp or {}).get('error')}")
                    for contract_id in contract_ids:
                        resp = await self._send_with_id(
                            {"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1}
                        )
                        if resp is None or resp.get("error"):
                            raise DerivApiError(
                                f"resubscribe contract({contract_id}) failed: {(resp or {}).get('error')}"
                            )
                    logger.info(
                        "reconnected -- resubscribed to %d symbol(s), %d contract(s)",
                        len(symbols), len(contract_ids),
                    )
                    return
                except Exception as exc:
                    logger.warning("reconnect attempt %d/5 failed: %s", attempt, exc)
                    delay = min(delay * 2, 30)
            logger.error("reconnect exhausted 5 attempts; websocket will stay down until process restart")
        finally:
            self._reconnecting = False

    def _fail_pending(self, reason: str) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(DerivApiError(reason))
        self._pending.clear()

    async def _dispatch(self, msg: dict[str, Any]) -> None:
        req_id = msg.get("req_id")
        if req_id is not None and req_id in self._pending:
            fut = self._pending.pop(req_id)
            if not fut.done():
                fut.set_result(msg)
            return

        msg_type = msg.get("msg_type")
        if msg_type == "tick":
            # IMPORTANT: never `await handler(tick)` here. _recv_pump calls
            # _dispatch() in a tight loop that is the ONLY thing capable of
            # reading proposal/buy responses off the socket (see
            # _send_with_id). If a tick handler is awaited inline and it
            # turns around and calls proposal()/buy() (which it does, on
            # every trade signal), the handler blocks waiting for a response
            # that only this same loop can deliver -- a self-deadlock that
            # times every trade out after exactly the _send_with_id timeout,
            # and starves the underlying websocket's ping/pong handling
            # badly enough to trigger server-side keepalive-timeout closes.
            # Handing the tick to a per-symbol queue/worker keeps this loop
            # free to keep reading (and keeps per-symbol tick ordering
            # intact, since each symbol's worker processes its queue
            # strictly one tick at a time).
            tick = msg.get("tick", {})
            symbol = tick.get("symbol")
            queue = self._tick_queues.get(symbol)
            if queue is not None:
                queue.put_nowait(tick)
        elif msg_type == "proposal_open_contract":
            poc = msg.get("proposal_open_contract", {})
            handler = self._contract_handlers.get(poc.get("contract_id"))
            if handler:
                asyncio.create_task(handler(poc), name=f"poc_handler_{poc.get('contract_id')}")
        elif msg.get("error"):
            logger.error("Deriv API error (unsolicited): %s", msg["error"])

    def _ensure_tick_worker(self, symbol: str) -> None:
        existing = self._tick_workers.get(symbol)
        if existing is not None and not existing.done():
            return
        queue: asyncio.Queue = asyncio.Queue()
        self._tick_queues[symbol] = queue
        self._tick_workers[symbol] = asyncio.create_task(
            self._tick_worker(symbol, queue), name=f"tick_worker_{symbol}"
        )

    async def _tick_worker(self, symbol: str, queue: asyncio.Queue) -> None:
        """Processes one symbol's ticks strictly in arrival order, off the
        recv pump. A slow tick (e.g. one that fires a trade and waits on
        proposal()/buy()) only blocks this symbol's own queue -- it never
        blocks _recv_pump, other symbols, or that same proposal's response
        from arriving."""
        while True:
            tick = await queue.get()
            try:
                handler = self._tick_handlers.get(symbol)
                if handler:
                    await handler(tick)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s: tick worker handler error", symbol)
            finally:
                queue.task_done()

    # ---- request/response plumbing ----

    def _next_req_id(self) -> int:
        rid = self._req_id_counter
        self._req_id_counter += 1
        return rid

    async def _send(self, data: dict) -> None:
        if self._send_queue is None:
            raise DerivApiError("not connected")
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        await self._send_queue.put((data, fut))
        await fut

    async def _send_with_id(self, data: dict, timeout: float = 15.0) -> dict[str, Any] | None:
        loop = asyncio.get_event_loop()
        rid = self._next_req_id()
        fut = loop.create_future()
        self._pending[rid] = fut
        try:
            await self._send({**data, "req_id": rid})
        except Exception:
            self._pending.pop(rid, None)
            if not fut.done():
                fut.cancel()
            raise
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=timeout)
        except asyncio.TimeoutError:
            self._pending.pop(rid, None)
            if not fut.done():
                fut.cancel()
            return None

    # ---- public API (unchanged surface -- callers need no changes) ----

    async def subscribe_ticks(self, symbol: str, handler: TickHandler) -> None:
        self._tick_handlers[symbol] = handler
        self._ensure_tick_worker(symbol)
        resp = await self._send_with_id({"ticks": symbol, "subscribe": 1})
        if resp is None or resp.get("error"):
            raise DerivApiError(f"subscribe_ticks({symbol}) failed: {(resp or {}).get('error')}")

    async def ticks_history(self, symbol: str, count: int = 5000) -> list[float]:
        """Pull recent tick history for cold-start seeding of the cumulative
        Markov tables -- returns quotes in chronological order, ready to be
        replayed through the same last_digit()/observe() path live ticks use.
        """
        resp = await self._send_with_id(
            {
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": count,
                "end": "latest",
                "style": "ticks",
            },
            timeout=20.0,
        )
        if resp is None or resp.get("error"):
            raise DerivApiError(f"ticks_history({symbol}) failed: {(resp or {}).get('error')}")
        return [float(p) for p in resp["history"]["prices"]]

    async def proposal(
        self, symbol: str, contract_type: str, barrier: int, amount: float, currency: str, duration_ticks: int
    ) -> dict[str, Any]:
        resp = await self._send_with_id(
            {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type,
                "currency": currency,
                "duration": duration_ticks,
                "duration_unit": "t",
                "underlying_symbol": symbol,
                "barrier": str(barrier),
            }
        )
        if resp is None or resp.get("error"):
            raise DerivApiError(f"proposal failed: {resp.get('error') if resp else 'no response'}")
        return resp["proposal"]

    async def buy(self, proposal_id: str, price: float) -> dict[str, Any]:
        resp = await self._send_with_id({"buy": proposal_id, "price": price})
        if resp is None or resp.get("error"):
            raise DerivApiError(f"buy failed: {resp.get('error') if resp else 'no response'}")
        return resp["buy"]

    async def subscribe_contract(self, contract_id: int, handler: ContractHandler) -> None:
        self._contract_handlers[contract_id] = handler
        resp = await self._send_with_id({"proposal_open_contract": 1, "contract_id": contract_id, "subscribe": 1})
        if resp is None or resp.get("error"):
            raise DerivApiError(f"subscribe_contract({contract_id}) failed: {(resp or {}).get('error')}")

    def unsubscribe_contract(self, contract_id: int) -> None:
        self._contract_handlers.pop(contract_id, None)
