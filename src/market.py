"""Synthetic high-frequency market generator.

We model the *latent fair value* of an asset as a mean-reverting
Ornstein-Uhlenbeck (OU) process and the *observed mid price* as that fair
value plus i.i.d. microstructure noise (bid/ask bounce), with optional rare
jumps. This is deliberately lightweight and pedagogical -- it is NOT an
agent-based limit-order-book simulator.

Why OU? Real high-frequency prices mean-revert at short horizons. That
short-horizon mean reversion is what makes the near-future direction *partly
predictable* (so a fast, accurate model can profit) and, at the same time,
what makes a *late* entry harmful (a slow model gets filled after the move it
predicted has already started to revert). That single property is the engine
behind the inverted-U profit-vs-latency curve this project demonstrates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Market:
    """A generated market path.

    Attributes
    ----------
    latent:
        The latent OU "fair value" series ``X[t]`` (length ``n_ticks``).
    price:
        The observed mid price series ``P[t] = X[t] + noise`` (length
        ``n_ticks``). This is what a trader actually transacts at.
    ms_per_tick:
        Wall-clock milliseconds represented by one tick. Used to convert a
        latency in milliseconds into an integer number of ticks.
    """

    latent: np.ndarray
    price: np.ndarray
    ms_per_tick: float

    @property
    def n_ticks(self) -> int:
        return int(self.price.shape[0])


def generate_market(cfg: dict, seed: int) -> Market:
    """Generate a synthetic market path from the ``market`` config block.

    The latent fair value follows the discrete OU update

        X[t+1] = X[t] + theta * (mu - X[t]) + sigma * z[t],   z ~ N(0, 1)

    and the observed price adds independent Gaussian microstructure noise

        P[t] = X[t] + epsilon[t],                             epsilon ~ N(0, noise_sigma).

    Rare jumps (if enabled) add an occasional large shock to the latent value,
    which makes the latency penalty bite harder around fast-moving moments.

    Parameters
    ----------
    cfg:
        The full parsed ``config.yaml`` dictionary (we read ``cfg['market']``).
    seed:
        Global random seed. Generation is fully deterministic in ``seed``.

    Returns
    -------
    Market
        The generated latent and observed price series.
    """
    m = cfg["market"]
    n: int = int(m["n_ticks"])
    rng = np.random.default_rng(seed)

    theta = float(m["theta"])
    mu = float(m["mu"])
    sigma = float(m["sigma"])

    # OU innovations and (optional) jump shocks for the latent fair value.
    z = rng.standard_normal(n)
    shocks = sigma * z

    jumps = m.get("jumps", {})
    if jumps.get("enabled", False):
        # A jump occurs on a tick with probability `prob_per_tick`; its signed
        # magnitude is Gaussian with std `size`. Jumps are added to the latent
        # process so they propagate forward (they are not pure noise).
        jump_mask = rng.random(n) < float(jumps["prob_per_tick"])
        jump_vals = rng.standard_normal(n) * float(jumps["size"])
        shocks = shocks + jump_mask * jump_vals

    # Integrate the OU process iteratively. n_ticks is ~1e5-1e6; a Python loop
    # would dominate runtime, so we use an explicit recurrence that numpy can
    # run quickly enough, but keep it readable. (theta is small, so the simple
    # Euler update is stable.)
    latent = np.empty(n, dtype=np.float64)
    latent[0] = float(m["x0"])
    one_minus_theta = 1.0 - theta
    theta_mu = theta * mu
    for t in range(1, n):
        # X[t] = X[t-1] + theta*(mu - X[t-1]) + shock = (1-theta)*X[t-1] + theta*mu + shock
        latent[t] = one_minus_theta * latent[t - 1] + theta_mu + shocks[t - 1]

    # Observed mid = latent fair value + microstructure noise.
    epsilon = rng.standard_normal(n) * float(m["noise_sigma"])
    price = latent + epsilon

    return Market(latent=latent, price=price, ms_per_tick=float(m["ms_per_tick"]))


def ms_to_ticks(latency_ms: float, ms_per_tick: float) -> int:
    """Convert a latency in milliseconds to an integer (non-negative) tick count.

    A latency below half a tick rounds to 0 ticks (instant fill); otherwise we
    round to the nearest whole tick because the market is sampled discretely.
    """
    return int(max(0, round(latency_ms / ms_per_tick)))
