"""Standalone figure: three bots' cumulative PnL over time -- one per region of
the inverted-U.

We pick three models off the same accuracy/latency frontier used everywhere
else and run each over the SAME single example market (the global seed, the one
Figure 0 plots), then draw each bot's cumulative net PnL as trades accumulate
through the day:

  * Fast & inaccurate (left of the hump):  barely finds an edge -> stays ~flat.
  * Sweet spot (top of the hump):          climbs steadily -> profitable.
  * Slow & accurate (right of the hump):   right about moves but filled late ->
                                           drifts negative.

This is the inverted-U told as three time series instead of one curve.

    python make_bot_paths.py
"""

from __future__ import annotations

import numpy as np
import yaml

from src.market import generate_market
from src.model import frontier_points
from src.backtest import run_backtest
from src import plotting
from src.plotting import OKABE_ITO


# Which bots (indices into the n_points frontier sweep) represent each region,
# and how to label / colour them.
REGIONS = [
    {"idx": 2,  "label": "Fast & inaccurate",  "color": OKABE_ITO["sky"]},
    {"idx": 10, "label": "Sweet spot",          "color": OKABE_ITO["green"]},
    {"idx": 29, "label": "Slow & accurate",     "color": OKABE_ITO["vermillion"]},
]


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["seed"])
    market = generate_market(cfg, seed)
    ms_per_tick = float(cfg["market"]["ms_per_tick"])

    n_points = int(cfg["experiments"]["frontier_sweep"]["n_points"])
    bots = frontier_points(cfg, n_points)

    plotting.apply_paper_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for r in REGIONS:
        spec = bots[r["idx"]]
        res = run_backtest(market, spec.accuracy, spec.latency_ms, cfg, seed,
                           return_timeline=True)

        # Cumulative net PnL vs wall-clock time (ticks -> ms -> seconds).
        t_s = res.trade_ticks * ms_per_tick / 1000.0
        cum = np.cumsum(res.trade_pnl)
        # Anchor the line at the origin so all bots start flat at 0.
        t_s = np.concatenate(([0.0], t_s))
        cum = np.concatenate(([0.0], cum))

        # Report the realised average latency (after jitter) to match Fig 1 and
        # the frontier figure, not the pre-jitter nominal value.
        ax.plot(t_s, cum, color=r["color"], lw=2.0,
                label=(f"{r['label']}  "
                       f"(L={res.avg_latency_ms:.1f} ms, p={spec.accuracy:.2f}, "
                       f"end={cum[-1]:+.1f})"))

    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Time  (seconds of trading on the example market)")
    ax.set_ylabel("Cumulative net PnL  (sum of per-trade returns)")
    ax.set_title("Three bots, one market: the inverted-U as PnL over time")
    ax.legend(loc="upper left", framealpha=0.9)

    paths = plotting._save(fig, cfg["output"]["figures_dir"], "fig_bot_paths")
    for p in paths:
        print(f"[written] {p}")


if __name__ == "__main__":
    main()
