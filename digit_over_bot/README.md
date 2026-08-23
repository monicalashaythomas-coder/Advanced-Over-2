# digit-over-bot

A standalone Deriv bot that trades **Digit Over `barrier`** contracts (default
barrier = 2, i.e. wins if the last digit is 3-9) across multiple volatility
index symbols in parallel, using a statistically-disciplined ensemble rather
than a single indicator.

## Honest starting point

Deriv's synthetic indices run on an independently audited RNG designed to
produce IID uniform last digits -- there is no publicly known mechanism by
which the digit stream has memory. That means:

- Any chi-square/z-score/Markov reading you see WILL fluctuate tick to tick.
  That's sampling variance, not necessarily structure.
- The only honest way to use these tools is to make the bot's own confidence
  falsifiable: log every prediction against what actually happens next, and
  let the system distrust itself the moment its calibration stops holding up
  (see "Self-doubt breaker" below). `tests/test_ensemble.py` demonstrates
  this discipline directly: on 3000 ticks of genuinely uniform random
  digits, the ensemble's false "should_trade" rate stays under 2% even
  though the underlying statistical tests nominally allow up to ~1% each --
  the multi-model-agreement + edge-floor + sigma-multiple requirements
  compound to make spurious firing rare, not the reverse.
- Treat this as a rigorously-instrumented experiment that will tell you the
  truth about whether it works, not as a guaranteed edge.

## Guiding principle (as given, plus refinements)

> Do not predict a digit with certainty. Estimate the probability
> distribution of the next last digit, detect whether the current state
> contains exploitable structure, and trade only when multiple independent
> models agree.

Refinements implemented on top of that:

1. **Every model outputs an edge + standard error, never a bare point
   guess** -- uncertainty is carried through the whole pipeline.
2. **Chi-square is a gate, not a weighted vote.** It answers "is the full
   10-way digit distribution non-uniform at all in this window?" and
   contributes a directional corroboration (via its standardized residuals)
   but never swings the combined edge's *magnitude* -- its residuals aren't
   safely convertible to a calibrated probability without extra assumptions.
3. **Inverse-variance combination** of the quantitative votes (z-score +
   Markov orders 1-3), each additionally scaled by a **learned reliability
   weight** that reflects that specific model's own recent calibration
   (rolling mean Brier score -- see `src/learner.py`), not just its
   textbook variance.
4. **Multiple-testing discipline**: alpha defaults to 0.01, not 0.05, and a
   trade additionally requires (a) an absolute edge floor (`MIN_EDGE`), (b)
   the combined edge to clear several multiples of its own combined standard
   error (`MIN_EDGE_SIGMA_MULTIPLE`), and (c) a minimum count of
   independently-available models agreeing in direction
   (`MIN_MODELS_AGREEING`) -- no single model, however significant on its
   own, is ever sufficient.
5. **Live EV check against the actual quoted payout**, using a
   *conservative* probability (point estimate minus one combined standard
   error, floored at the fair rate) -- not the optimistic point estimate --
   right before firing (`src/executor.py`).
6. **Self-doubt breaker** (`src/learner.py: should_pause`): the ensemble's
   own rolling live calibration (Brier score) is compared against the
   trivial "always predict fair odds" baseline. If the ensemble has recently
   been a *worse* predictor than assuming nothing is exploitable, trading
   pauses (learning continues) until that recovers. This is what "keep
   learning and relearning as the pattern develops and changes" means
   concretely: weights and the pause state both move with recent
   performance, not just at startup.

## Architecture

```
src/
  config.py          env-driven settings
  digit_buffer.py     rolling window (recency) + decayed cumulative Markov counts (sample-size)
  stats/
    chi_square.py     structure gate + directional residual corroboration
    zscore.py         directional edge estimate vs fair Over-probability
    markov.py         orders 1-3, Laplace-smoothed, Katz backoff
  ensemble.py         inverse-variance combination + agreement + significance gating
  learner.py          per-model reliability weights + self-doubt calibration breaker
  risk.py             daily loss / consecutive-loss / concurrency circuit breakers
  deriv_client.py     minimal async Deriv WS v3 client (ticks, proposal, buy, settlement)
  executor.py         live EV check (conservative probability) then buy
  storage/
    supabase_client.py  thin async PostgREST client (best-effort, never blocks trading)
  bot.py              per-symbol orchestration, wiring it all together
  main.py             entrypoint + reconnect loop
schema.sql            Supabase tables (run once in the SQL editor)
tests/                unit + ensemble + learner tests, and a network-free smoke test
```

### Why two different digit buffers per symbol?

- A **rolling window** (default 1000, `BUFFER_SIZE`) feeds chi-square,
  z-score, and "which order-k state are we in right now" -- old ticks age
  OUT, so the bot can react if structure drifts or disappears.
- **Decayed cumulative Markov counts** (`digit_buffer.py:
  DecayedMarkovCounts`) let the higher Markov orders (order 3 = 1000
  possible conditioning states) accumulate enough samples per state to be
  trustworthy at all -- a 1000-tick rolling window alone gives ~1 sample per
  order-3 state on average, nowhere near enough. Evidence still decays
  (default 0.999/tick), just more slowly than the rolling window, so it
  doesn't assume the process is stationary forever either.

### Per-tick flow (`bot.py: _on_tick`)

1. Score the **previous** tick's prediction against the digit that just
   arrived (this happens on every tick, whether or not a trade fired --
   learning must not be starved by how rarely the bot actually trades).
2. Update the Markov tables with the state as it was *before* this digit.
3. Push the new digit into the rolling window.
4. Generate a fresh prediction for the *next* digit, log it (sampled, plus
   always when `should_trade`), and stash it to be scored next tick.
5. Check the self-doubt breaker; if paused, stop here (still learning).
6. If `should_trade`, check the risk manager's circuit breakers, then hand
   off to the executor for a live EV check and (if it clears) purchase.

## Setup

1. Create a Deriv app at <https://developers.deriv.com> (do not use the
   legacy demo app_id 1089) and get an API token for a **demo account**
   first.
2. Create a Supabase project, then run `schema.sql` in its SQL editor.
3. Copy `.env.example` to `.env` and fill in `DERIV_APP_ID`,
   `DERIV_API_TOKEN`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`.
4. `pip install -r requirements.txt`
5. `python -m tests.test_stats && python -m tests.test_ensemble && python -m tests.test_learner && python -m tests.smoke_test`
6. `python -m src.main`

## Deploying to Railway

`railway.json` / `Procfile` are both included (`worker: python -m src.main`).
Set the same env vars from `.env.example` in the Railway project's
Variables tab. This is a long-running worker, not a web service -- no port
needs to be exposed.

## Key config knobs (`.env.example` has the full list)

| Var | What it controls |
|---|---|
| `SYMBOLS` | comma-separated symbols traded in parallel |
| `BARRIER` | Over-`barrier` contract (default 2) |
| `ALPHA` | significance threshold for individual model tests |
| `MIN_EDGE` | absolute floor on the combined edge over fair odds |
| `MIN_EDGE_SIGMA_MULTIPLE` | how many combined-SEs the edge must clear |
| `MIN_MODELS_AGREEING` | how many of {z-score, Markov 1/2/3, chi-square} must agree |
| `MIN_MARKOV_STATE_COUNT` | min samples before an individual Markov order is trusted |
| `WEIGHT_LEARNING_RATE` | how fast per-model reliability weights adapt |
| `CALIBRATION_PAUSE_THRESHOLD` | how much worse than baseline before pausing |
| `MAX_DAILY_LOSS_PCT` / `MAX_CONSECUTIVE_LOSSES` / `MAX_CONCURRENT_OPEN` | risk breakers |

## What isn't done for you

- This trades **Over only** (as asked). The ensemble already computes a
  signed combined edge, so adding a symmetric Under path when the edge
  favors the other side is a small extension in `ensemble.py` /
  `executor.py` if you want it.
- The Deriv client was written directly from current API docs
  (`developers.deriv.com`), not against a live session (no network path to
  Deriv from the environment this was built in) -- confirm the `proposal`/
  `buy` field names still match before trusting it with a funded account.
- Position sizing is flat (`STAKE`); there's no Kelly/martingale sizing --
  intentional, given the whole point of the ensemble is to avoid needing
  to make up for a low edge with bet-sizing tricks.
