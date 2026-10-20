"""Tests for the experiment drivers -- especially the inverted-U shape."""

from __future__ import annotations

import numpy as np

from src.experiments import (
    run_frontier_sweep,
    run_frontier_sweep_multiseed,
    run_gap_analysis,
    run_heatmap,
)


def test_multiseed_band_is_well_formed_and_robust(small_config):
    """The multi-seed sweep yields an ordered confidence band, and the hump is
    robust: its peak mean PnL is positive across independent markets."""
    seed = int(small_config["seed"])
    df = run_frontier_sweep_multiseed(small_config, seed, n_seeds=5)
    assert len(df) == small_config["experiments"]["frontier_sweep"]["n_points"]
    # Band ordering holds at every bot.
    assert (df["net_pnl_lo"] <= df["net_pnl"]).all()
    assert (df["net_pnl"] <= df["net_pnl_hi"]).all()
    assert (df["net_pnl_p10"] <= df["net_pnl_p90"]).all()
    # The hump survives averaging over markets.
    assert df["net_pnl"].max() > 0.0


def test_frontier_sweep_is_hump_shaped(market, small_config):
    """The headline result: net PnL vs latency peaks at an INTERIOR point.

    Both extremes (fast-but-dumb and slow-but-accurate) must be worse than the
    interior optimum -- i.e. the argmax is neither the first nor the last point.
    """
    seed = int(small_config["seed"])
    df = run_frontier_sweep(small_config, market, seed).sort_values("latency_ms")
    pnl = df["net_pnl"].to_numpy()

    i_peak = int(np.argmax(pnl))
    assert 0 < i_peak < len(pnl) - 1, "profit peak must be interior (hump-shaped)"
    # The peak strictly beats both endpoints.
    assert pnl[i_peak] > pnl[0]
    assert pnl[i_peak] > pnl[-1]


def test_accuracy_rises_monotonically_along_frontier(market, small_config):
    """Accuracy increases with complexity/latency along the frontier."""
    seed = int(small_config["seed"])
    df = run_frontier_sweep(small_config, market, seed).sort_values("latency_ms")
    acc = df["accuracy"].to_numpy()
    assert np.all(np.diff(acc) >= -1e-12)


def test_heatmap_high_latency_loses_even_at_high_accuracy(market, small_config):
    """At the highest latency, even top accuracy cannot beat a modest-accuracy,
    low-latency model -- the core message of the decoupled heatmap."""
    seed = int(small_config["seed"])
    acc, lat, pnl = run_heatmap(small_config, market, seed)
    # pnl shape: (n_accuracy, n_latency). Highest accuracy = last row,
    # highest latency = last column.
    high_acc_high_lat = pnl[-1, -1]
    mid_acc_low_lat = pnl[pnl.shape[0] // 2, 0]
    assert high_acc_high_lat < mid_acc_low_lat


def test_gap_widens_with_latency(market, small_config):
    """At fixed accuracy, the gap between the believed and realised hit rate
    grows with latency -- isolating the pure damage done by being late."""
    seed = int(small_config["seed"])
    df = run_gap_analysis(small_config, market, seed).sort_values("latency_ms")
    gap = (df["believed_hit_rate"] - df["realized_hit_rate"]).to_numpy()
    # Believed stays ~flat (accuracy is held fixed); realised falls, so the gap
    # at the highest latency clearly exceeds the gap at zero latency.
    assert gap[-1] > gap[0] + 0.05
    # Believed hit rate stays well above a coin flip throughout.
    assert df["believed_hit_rate"].min() > 0.5
