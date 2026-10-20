"""Tests for the latency-aware backtest engine -- the trade-off mechanism."""

from __future__ import annotations

import copy

import numpy as np

from src.backtest import run_backtest


def _with(cfg: dict, **bt_overrides) -> dict:
    """Deep-copy a config and override keys in its ``backtest`` block."""
    c = copy.deepcopy(cfg)
    c["backtest"].update(bt_overrides)
    return c


def test_zero_latency_perfect_accuracy_is_profitable(market, small_config):
    """Sanity: a perfect, instant model must make money.

    accuracy = 1.0 and latency = 0 means every call is correct about the latent
    move and we are filled immediately, so net PnL must be strictly positive.
    """
    seed = int(small_config["seed"])
    res = run_backtest(market, accuracy=1.0, latency_ms=0.0, cfg=small_config, seed=seed)
    assert res.net_pnl > 0.0
    assert res.hit_rate > 0.5


def test_hit_rate_decreases_with_latency(market, small_config):
    """For fixed nominal accuracy, the execution-time hit rate falls as latency
    grows -- the model is increasingly 'right about the past'."""
    seed = int(small_config["seed"])
    latencies = [0.0, 2.0, 5.0, 10.0, 20.0, 30.0]
    hit_rates = [
        run_backtest(market, 1.0, L, small_config, seed).hit_rate for L in latencies
    ]
    # Non-increasing (allow a tiny numerical tolerance), and clearly lower overall.
    for earlier, later in zip(hit_rates, hit_rates[1:]):
        assert later <= earlier + 1e-9
    assert hit_rates[0] - hit_rates[-1] > 0.1


def test_net_pnl_decreases_with_latency_at_fixed_accuracy(market, small_config):
    """The latency penalty erodes profit even when accuracy is held high."""
    seed = int(small_config["seed"])
    pnl_fast = run_backtest(market, 0.75, 0.0, small_config, seed).net_pnl
    pnl_slow = run_backtest(market, 0.75, 30.0, small_config, seed).net_pnl
    assert pnl_slow < pnl_fast


def test_higher_accuracy_is_more_skilful(market, small_config):
    """At fixed latency, a higher-accuracy model is right about the coming move
    at least as often (believed hit rate is monotone in nominal accuracy)."""
    seed = int(small_config["seed"])
    lo = run_backtest(market, 0.60, 0.0, small_config, seed)
    hi = run_backtest(market, 0.80, 0.0, small_config, seed)
    assert hi.believed_hit_rate >= lo.believed_hit_rate


def test_per_decision_latency_array_supported(market, small_config):
    """An adaptive (per-decision) latency array runs and reports its average.

    The array spans every *decision* (one latency per decision opportunity);
    gating then selects which of those become trades.
    """
    seed = int(small_config["seed"])
    base = run_backtest(market, 0.75, 5.0, small_config, seed)
    n = base.n_decisions
    lat = np.full(n, 5.0)
    res = run_backtest(market, 0.75, lat, small_config, seed)
    assert res.n_decisions == n
    assert abs(res.avg_latency_ms - 5.0) < 1e-9


# --- Realistic signal-gating behaviour ------------------------------------- #
def test_bot_trades_selectively(market, small_config):
    """A skilled model trades on SOME but not ALL decisions (it waits for a
    strong enough signal instead of trading every tick)."""
    seed = int(small_config["seed"])
    res = run_backtest(market, 0.72, 5.0, small_config, seed)
    assert 0 < res.n_trades < res.n_decisions
    assert 0.0 < res.trade_rate < 1.0


def test_coinflip_model_stays_flat(market, small_config):
    """A no-skill model (accuracy ~ 0.5) believes it has no edge, so it never
    trades -- exactly what a sensible real system would do."""
    seed = int(small_config["seed"])
    res = run_backtest(market, 0.50, 1.0, small_config, seed)
    assert res.n_trades == 0
    assert res.net_pnl == 0.0


def test_believed_exceeds_realized_under_latency(market, small_config):
    """The crux: at high latency the model is still 'right about the past'
    (high believed hit rate) but loses at execution (low realised hit rate)."""
    seed = int(small_config["seed"])
    res = run_backtest(market, 0.77, 28.0, small_config, seed)
    assert res.believed_hit_rate > res.hit_rate
    # It is genuinely skilful about the move it bet on...
    assert res.believed_hit_rate > 0.5
    # ...yet latency drags its realised hit rate down toward (or below) a coin flip.
    assert res.hit_rate < res.believed_hit_rate - 0.05


# --- Bid/ask spread -------------------------------------------------------- #
def test_spread_reduces_pnl(market, small_config):
    """Paying the bid/ask spread on every fill lowers net PnL vs a zero-spread
    market (same bot, same trades)."""
    seed = int(small_config["seed"])
    no_spread = _with(small_config, spread_bps=0.0)
    wide_spread = _with(small_config, spread_bps=5.0)
    pnl_tight = run_backtest(market, 0.72, 5.0, no_spread, seed).net_pnl
    pnl_wide = run_backtest(market, 0.72, 5.0, wide_spread, seed).net_pnl
    assert pnl_wide < pnl_tight


# --- Latency jitter -------------------------------------------------------- #
def test_jitter_raises_average_latency_above_nominal(market, small_config):
    """With fat-tailed jitter on, a bot's realised AVERAGE latency exceeds its
    nominal value (the tail spikes pull the mean up); with jitter off it equals
    the nominal exactly."""
    seed = int(small_config["seed"])
    jit_off = _with(small_config, latency_jitter={"enabled": False})
    assert abs(run_backtest(market, 0.72, 10.0, jit_off, seed).avg_latency_ms - 10.0) < 1e-9

    jit_on = _with(small_config, latency_jitter={
        "enabled": True, "rel_sigma": 0.3, "tail_prob": 0.03, "tail_mult": 4.0})
    assert run_backtest(market, 0.72, 10.0, jit_on, seed).avg_latency_ms > 10.0
