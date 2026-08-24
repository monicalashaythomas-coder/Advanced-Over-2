"""
Thin async Supabase (PostgREST) client -- deliberately just `httpx` against
the REST endpoint rather than the `supabase-py` SDK, so this doesn't depend
on tracking that SDK's API across versions. Requires `schema.sql` (repo
root) to have been applied to the Supabase project first.

Every call is best-effort: a Supabase outage should never be able to stop
the bot from trading or block a settlement from being processed -- it can
only mean that run's audit trail has a gap, which is logged loudly.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("supabase")


class SupabaseStore:
    def __init__(self, url: str, service_key: str, enabled: bool = True) -> None:
        self.enabled = enabled and bool(url) and bool(service_key)
        self._base = url.rstrip("/") + "/rest/v1"
        self._headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client

    async def insert(self, table: str, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            client = await self._get_client()
            resp = await client.post(f"{self._base}/{table}", headers=self._headers, json=row)
            if resp.status_code >= 300:
                logger.warning("Supabase insert into %s failed: %s %s", table, resp.status_code, resp.text)
        except Exception:
            logger.exception("Supabase insert into %s raised", table)

    async def select(self, table: str, match: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        try:
            client = await self._get_client()
            params = {k: f"eq.{v}" for k, v in match.items()}
            resp = await client.get(f"{self._base}/{table}", headers=self._headers, params=params)
            if resp.status_code >= 300:
                logger.warning("Supabase select on %s failed: %s %s", table, resp.status_code, resp.text)
                return []
            return resp.json()
        except Exception:
            logger.exception("Supabase select on %s raised", table)
            return []

    async def upsert(self, table: str, row: dict[str, Any], on_conflict: str) -> None:
        logger.info("upsert() called for %s (enabled=%s)", table, self.enabled)
        if not self.enabled:
            return
        try:
            client = await self._get_client()
            headers = {**self._headers, "Prefer": "resolution=merge-duplicates"}
            params = {"on_conflict": on_conflict}
            resp = await client.post(f"{self._base}/{table}", headers=headers, params=params, json=row)
            if resp.status_code >= 300:
                logger.warning("Supabase upsert into %s failed: %s %s", table, resp.status_code, resp.text)
            else:
                logger.info("Supabase upsert into %s succeeded: %s", table, resp.status_code)
        except Exception:
            logger.exception("Supabase upsert into %s raised", table)

    async def update(self, table: str, match: dict[str, Any], patch: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            client = await self._get_client()
            params = {k: f"eq.{v}" for k, v in match.items()}
            resp = await client.patch(f"{self._base}/{table}", headers=self._headers, params=params, json=patch)
            if resp.status_code >= 300:
                logger.warning("Supabase update on %s failed: %s %s", table, resp.status_code, resp.text)
        except Exception:
            logger.exception("Supabase update on %s raised", table)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
