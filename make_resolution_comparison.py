#!/usr/bin/env python3
"""Side-by-side: the latency sweep at coarse (1 ms) vs fine (0.25 ms) ticks.

The zigzag in the headline figure is a *discretisation artifact*: latency is
rounded to a whole number of ticks before the fill, while accuracy varies
continuously along the frontier. On steps where the rounded latency is unchanged
PnL ticks up (accuracy gained, latency flat); on steps where it crosses into the
next tick PnL ticks down (latency penalty jumps) -- hence the alternating saw.

This script samples the SAME wall-clock experiment 4x more finely so latency
rounds to 0.25 ms steps and the saw shrinks to near-invisibility.

IMPORTANT -- a fair refinement does NOT just change `ms_per_tick`. The OU
dynamics and the horizon are defined *per tick*, so to hold the millisecond
physics fixed while quartering the tick we must also rescale:

    theta  -> theta / s          (mean-reversion rate per ms unchanged)
    sigma  -> sigma / sqrt(s)     (variance accrued per ms unchanged)
    jump prob/tick -> /s          (jumps per ms unchanged)
    forward_horizon, decision_stride, n_ticks -> * s   (same ms span / trade count)

with s = (old ms_per_tick) / (new ms_per_tick) = 4. Stationary variance
sigma^2/(2*theta), the mean-reversion half-life in ms, the horizon in ms, and
every latency in ms are then all identical -- only the sampling grid is finer.
"""

from __future__ import annotations

import copy
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.experiments import run_frontier_sweep
from src.market import generate_market
from src import plotting

ROOT = os.path.dirname(os.path.abspath(__file__))


def refine_config(cfg: dict, new_ms_per_tick: float) -> dict:
    """Return a copy of ``cfg`` sampled at ``new_ms_per_tick`` with identical
    millisecond-scale physics (see module docstring)."""
    s = cfg["market"]["ms_per_tick"] / new_ms_per_tick
    fine = copy.deepcopy(cfg)
    m = fine["market"]
    m["ms_per_tick"] = new_ms_per_tick
    m["theta"] = cfg["market"]["theta"] / s
    m["sigma"] = cfg["market"]["sigma"] / np.sqrt(s)
    m["n_ticks"] = int(round(cfg["market"]["n_ticks"] * s))
    if m.get("jumps", {}).get("enabled", False):
        m["jumps"]["prob_per_tick"] = cfg["market"]["jumps"]["prob_per_tick"] / s
    fine["backtest"]["forward_horizon"] = int(round(cfg["backtest"]["forward_horizon"] * s))
    fine["backtest"]["decision_stride"] = int(round(cfg["backtest"]["decision_stride"] * s))
    return fine


# (ms_per_tick, panel title, "draw markers?") from coarse -> fine.
GRIDS = [
    (5.00, "Coarse: 1 tick = 5.00 ms\n(chunky staircase)", True),
    (1.00, "Current: 1 tick = 1.00 ms\n(zigzag)", True),
    (0.25, "Fine: 1 tick = 0.25 ms\n(smooth)", False),
]


def main() -> int:
    with open(os.path.join(ROOT, "config.yaml")) as fh:
        cfg = yaml.safe_load(fh)
    seed = int(cfg["seed"])

    plotting.apply_paper_style()
    fig, axes = plt.subplots(1, len(GRIDS), figsize=(14.0, 4.4), sharey=True)
    c_blue = plotting.OKABE_ITO["blue"]
    c_red = plotting.OKABE_ITO["vermillion"]

    for ax, (mpt, title, markers) in zip(axes, GRIDS):
        # mpt == base resolution -> use config as-is; else rescale physics.
        run_cfg = cfg if mpt == cfg["market"]["ms_per_tick"] else refine_config(cfg, mpt)
        print(f"Running sweep at 1 tick = {mpt:.2f} ms "
              f"(theta={run_cfg['market']['theta']:.4g}, sigma={run_cfg['market']['sigma']:.4g}, "
              f"h={run_cfg['backtest']['forward_horizon']} ticks, n_ticks={run_cfg['market']['n_ticks']:,})...")
        df = run_frontier_sweep(run_cfg, generate_market(run_cfg, seed), seed).sort_values("latency_ms")
        i = int(df["net_pnl"].to_numpy().argmax())
        n_steps = df["latency_ms"].map(lambda L: round(L / mpt)).nunique()
        print(f"  peak L={df['latency_ms'].iloc[i]:.2f} ms  PnL={df['net_pnl'].iloc[i]:.3f}  "
              f"| {n_steps} distinct latency steps across the sweep")

        if markers:
            ax.plot(df["latency_ms"], df["net_pnl"], color=c_blue, marker="o", ms=3.5)
        else:
            ax.plot(df["latency_ms"], df["net_pnl"], color=c_blue, lw=1.8)
        ax.scatter([df["latency_ms"].iloc[i]], [df["net_pnl"].iloc[i]],
                   color=c_red, marker="*", s=130, zorder=6)
        ax.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax.set_title(title)
        ax.set_xlabel("Model latency  L  (ms)")
    axes[0].set_ylabel("Net PnL")

    fig.suptitle("Same wall-clock experiment at three tick resolutions: "
                 "the saw is a discretisation artifact, not economics", fontsize=12)
    figs_dir = os.path.join(ROOT, cfg["output"]["figures_dir"])
    paths = plotting._save(fig, figs_dir, "fig_resolution_comparison")
    print("wrote:", *[os.path.relpath(p, ROOT) for p in paths])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
