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

    while True:
        bot = DigitOverBot(SETTINGS)
        try:
            await bot.run_forever()
        except Exception:  # noqa: BLE001
            logger.exception("bot crashed, reconnecting in %ss", RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)


if __name__ == "__main__":
    asyncio.run(main())
