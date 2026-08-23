-- Run this once against your Supabase project (SQL editor) before starting the bot.

create table if not exists digit_trades (
    id bigint generated always as identity primary key,
    symbol text not null,
    contract_id bigint,
    barrier int not null,
    stake numeric not null,
    payout numeric,
    profit numeric,
    win boolean,
    p_estimate numeric,
    combined_edge numeric,
    agreement_count int,
    votes_available int,
    reasons text,
    opened_epoch bigint,
    closed_epoch bigint,
    created_at timestamptz not null default now()
);

create table if not exists digit_ensemble_log (
    id bigint generated always as identity primary key,
    symbol text not null,
    n int,
    p_fair numeric,
    combined_edge numeric,
    combined_se numeric,
    agreement_count int,
    votes_available int,
    should_trade boolean,
    chi2_stat numeric,
    chi2_p numeric,
    zscore_z numeric,
    zscore_p numeric,
    markov1_edge numeric,
    markov2_edge numeric,
    markov3_edge numeric,
    reasons text,
    created_at timestamptz not null default now()
);

create table if not exists digit_weight_snapshots (
    id bigint generated always as identity primary key,
    weights_json jsonb not null,
    ensemble_brier numeric,
    baseline_brier numeric,
    paused boolean not null default false,
    pause_reason text,
    created_at timestamptz not null default now()
);

create index if not exists idx_digit_trades_symbol on digit_trades (symbol, created_at desc);
create index if not exists idx_digit_ensemble_log_symbol on digit_ensemble_log (symbol, created_at desc);
