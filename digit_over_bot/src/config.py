"""
Central configuration for the digit-over bot.

Everything here is env-driven so the same image runs on Railway (or anywhere
else) with no code changes -- only environment variables differ between a
demo-account deployment and a real-account one. There is deliberately no
"live" vs "demo" switch in code: that distinction is entirely a property of
which API token you hand it (Deriv demo tokens only ever touch a demo
account). Start with a demo token.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_int_list(name: str, default: str) -> list[int]:
    raw = os.getenv(name, default)
    return [int(s.strip()) for s in raw.split(",") if s.strip()]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DerivConfig:
    app_id: str = field(default_factory=lambda: os.getenv("DERIV_APP_ID", ""))
    api_token: str = field(default_factory=lambda: os.getenv("DERIV_API_TOKEN", ""))
    endpoint: str = field(
        default_factory=lambda: os.getenv("DERIV_WS_ENDPOINT", "wss://ws.derivws.com/websockets/v3")
    )


@dataclass(frozen=True)
class SupabaseConfig:
    url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    service_key: str = field(default_factory=lambda: os.getenv("SUPABASE_SERVICE_KEY", ""))
    enabled: bool = field(default_factory=lambda: _env_bool("SUPABASE_ENABLED", True))


@dataclass(frozen=True)
class TradingConfig:
    symbols: list[str] = field(
        default_factory=lambda: _env_list("SYMBOLS", "R_10,R_25,R_75,")
    )
    barrier: int = field(default_factory=lambda: _env_int("BARRIER", 2))  # "Digit Over 2"
    stake: float = field(default_factory=lambda: _env_float("STAKE", 0.35))
    currency: str = field(default_factory=lambda: os.getenv("CURRENCY", "USD"))
    duration_ticks: int = field(default_factory=lambda: _env_int("DURATION_TICKS", 1))
    max_price_slippage_pct: float = field(default_factory=lambda: _env_float("MAX_PRICE_SLIPPAGE_PCT", 5.0))

    # Monte-Carlo-guided duration selection (replaces the fixed duration_ticks
    # above for live trade firing -- duration_ticks stays as the fallback used
    # only when the digit window is too short to bootstrap yet). See
    # src/duration_selector.py.
    duration_candidates: list[int] = field(default_factory=lambda: _env_int_list("DURATION_CANDIDATES", "1,3,5"))
    duration_mc_samples: int = field(default_factory=lambda: _env_int("DURATION_MC_SAMPLES", 1000))
    duration_mc_block_size: int = field(default_factory=lambda: _env_int("DURATION_MC_BLOCK_SIZE", 10))

    # Rolling window sizes used by the structure-detection layer.
    window_sizes: list[int] = field(default_factory=lambda: [100, 250, 500, 1000])
    buffer_size: int = field(default_factory=lambda: _env_int("BUFFER_SIZE", 1000))
    markov_orders: list[int] = field(default_factory=lambda: [1, 2, 3])

    # Cold-start Markov seed. A single ticks_history call has, in practice,
    # returned exactly `markov_seed_batch_size` ticks for these symbols
    # regardless of the requested count -- these knobs chain multiple calls
    # (paging backward via `end`) to build a larger seed than one call alone
    # can provide, mainly so the higher Markov orders warm up faster. See
    # src/bot.py: _seed_markov_state.
    markov_seed_target_ticks: int = field(default_factory=lambda: _env_int("MARKOV_SEED_TARGET_TICKS", 20000))
    markov_seed_batch_size: int = field(default_factory=lambda: _env_int("MARKOV_SEED_BATCH_SIZE", 1000))
    markov_seed_max_batches: int = field(default_factory=lambda: _env_int("MARKOV_SEED_MAX_BATCHES", 25))
    markov_seed_batch_delay_s: float = field(
        default_factory=lambda: _env_float("MARKOV_SEED_BATCH_DELAY_S", 0.5)
    )

    # Statistical gating. See src/ensemble.py for how these combine.
    alpha: float = field(default_factory=lambda: _env_float("ALPHA", 0.01))
    min_edge: float = field(default_factory=lambda: _env_float("MIN_EDGE", 0.03))
    min_edge_sigma_multiple: float = field(default_factory=lambda: _env_float("MIN_EDGE_SIGMA_MULTIPLE", 2.0))
    min_models_agreeing: int = field(default_factory=lambda: _env_int("MIN_MODELS_AGREEING", 3))
    min_markov_state_count: int = field(default_factory=lambda: _env_int("MIN_MARKOV_STATE_COUNT", 30))

    # Risk / circuit breakers.
    max_daily_loss_pct: float = field(default_factory=lambda: _env_float("MAX_DAILY_LOSS_PCT", 10.0))
    max_consecutive_losses: int = field(default_factory=lambda: _env_int("MAX_CONSECUTIVE_LOSSES", 6))
    max_concurrent_open: int = field(default_factory=lambda: _env_int("MAX_CONCURRENT_OPEN", 1))
    cooldown_after_trade_s: float = field(default_factory=lambda: _env_float("COOLDOWN_AFTER_TRADE_S", 0.0))

    # Online relearning.
    weight_learning_rate: float = field(default_factory=lambda: _env_float("WEIGHT_LEARNING_RATE", 0.1))
    calibration_window: int = field(default_factory=lambda: _env_int("CALIBRATION_WINDOW", 200))
    calibration_pause_threshold: float = field(
        default_factory=lambda: _env_float("CALIBRATION_PAUSE_THRESHOLD", 0.08)
    )

    # Martingale staking (off by default -- see src/martingale.py). Same
    # shape as the expiryrange-quiet-bot's: a per-symbol step counter that
    # multiplies STAKE by martingale_factor**step after each loss, resets
    # to step 0 on a win or once martingale_max_steps is reached, and is
    # driven purely by the step counter (never by account balance).
    martingale_enabled: bool = field(default_factory=lambda: _env_bool("MARTINGALE_ENABLED", True))
    martingale_factor: float = field(default_factory=lambda: _env_float("MARTINGALE_FACTOR", 1.28))
    martingale_max_steps: int = field(default_factory=lambda: _env_int("MARTINGALE_MAX_STEPS", 3))


@dataclass(frozen=True)
class Settings:
    deriv: DerivConfig = field(default_factory=DerivConfig)
    supabase: SupabaseConfig = field(default_factory=SupabaseConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    dry_run: bool = field(default_factory=lambda: _env_bool("DRY_RUN", False))
    # Log every model's edge/se/significance/weight plus the agreement
    # verdict on every single tick evaluation (not just when a trade fires
    # or on the sampled Supabase cadence). Off this, only trade fires and
    # the periodic sampled line still log. On by default since it's the
    # main way to see what the layers are doing live.
    log_every_evaluation: bool = field(default_factory=lambda: _env_bool("LOG_EVERY_EVALUATION", True))


SETTINGS = Settings()
