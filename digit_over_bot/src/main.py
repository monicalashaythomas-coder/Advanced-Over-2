from __future__ import annotations

import asyncio
import logging
import sys

from src.bot import DigitOverBot
from src.config import SETTINGS

RECONNECT_DELAY_S = 5


def setup_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, SETTINGS.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


async def main() -> None:
    setup_logging()
    logger = logging.getLogger("main")

    if not SETTINGS.deriv.app_id or not SETTINGS.deriv.api_token:
        logger.error("DERIV_APP_ID / DERIV_API_TOKEN are not set. Refusing to start.")
        sys.exit(1)

    # Diagnostic only -- never logs the actual secret values, just whether
    # each piece SupabaseStore.enabled depends on is actually present, and
    # the same for martingale, since "enabled=False" alone (from
    # supabase_client.py's upsert log line) doesn't say *which* of
    # SUPABASE_URL / SUPABASE_SERVICE_KEY / SUPABASE_ENABLED is missing.
    sb = SETTINGS.supabase
    logger.info(
        "config check -- SUPABASE_URL set=%s len=%d | SUPABASE_SERVICE_KEY set=%s len=%d | "
        "SUPABASE_ENABLED=%s | resolved enabled=%s",
        bool(sb.url), len(sb.url), bool(sb.service_key), len(sb.service_key),
        sb.enabled, sb.enabled and bool(sb.url) and bool(sb.service_key),
    )
    t = SETTINGS.trading
    logger.info(
        "config check -- MARTINGALE_ENABLED=%s factor=%s max_steps=%s | MIN_MODELS_AGREEING=%d",
        t.martingale_enabled, t.martingale_factor, t.martingale_max_steps, t.min_models_agreeing,
    )

    while True:
        bot = DigitOverBot(SETTINGS)
        try:
            await bot.run_forever()
        except Exception:  # noqa: BLE001
            logger.exception("bot crashed, reconnecting in %ss", RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    asyncio.run(main())
