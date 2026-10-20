"""Tests for the synthetic market generator."""

from __future__ import annotations

import numpy as np

from src.market import generate_market, ms_to_ticks


def test_market_is_deterministic_under_fixed_seed(small_config):
    """Same seed + same config => byte-identical market (reproducibility)."""
    seed = int(small_config["seed"])
    a = generate_market(small_config, seed)
    b = generate_market(small_config, seed)
    assert np.array_equal(a.price, b.price)
    assert np.array_equal(a.latent, b.latent)


def test_different_seeds_differ(small_config):
    a = generate_market(small_config, 1)
    b = generate_market(small_config, 2)
    assert not np.array_equal(a.price, b.price)


def test_market_shapes_and_mean_reversion(market, small_config):
    """Price length matches config and the OU series stays near its mean."""
    assert market.n_ticks == small_config["market"]["n_ticks"]
    assert market.price.shape == market.latent.shape
    # Mean-reverting around mu: the long-run average should sit near mu.
    assert abs(float(np.mean(market.latent)) - small_config["market"]["mu"]) < 2.0


def test_ms_to_ticks():
    assert ms_to_ticks(0.0, 1.0) == 0
    assert ms_to_ticks(0.4, 1.0) == 0   # sub-half-tick rounds down to instant
    assert ms_to_ticks(0.6, 1.0) == 1
    assert ms_to_ticks(10.0, 1.0) == 10
    assert ms_to_ticks(10.0, 2.0) == 5  # 2 ms per tick
