#!/usr/bin/env python3
"""CLI entry point. ``python run.py --all`` reproduces every figure and table.

This is the single reproducibility command for the project: with a fixed seed in
``config.yaml`` it regenerates all outputs from scratch, deterministically.

Examples
--------
    python run.py --all                 # everything (figures + tables + console summary)
    python run.py --experiment 1        # just the inverted-U figure
    python run.py --config config.yaml  # use an alternative config
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

# Allow running both as `python run.py` and `python -m run` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import (  # noqa: E402
    run_frontier_sweep,
    run_frontier_sweep_multiseed,
    run_gap_analysis,
    run_heatmap,
    run_sensitivity,
)
from src.market import generate_market  # noqa: E402
from src import plotting  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _abspath(root: str, rel: str) -> str:
    return rel if os.path.isabs(rel) else os.path.join(root, rel)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=os.path.join(ROOT, "config.yaml"),
                        help="path to config.yaml")
    parser.add_argument("--all", action="store_true",
                        help="run all experiments and regenerate all outputs")
    parser.add_argument("--experiment", type=int, choices=[1, 2, 3, 4],
                        help="run a single experiment (1=inverted-U, 2=heatmap, "
                             "3=signal-gating/'right about the past', 4=sensitivity)")
    parser.add_argument("--sample-path", action="store_true",
                        help="also plot a sample synthetic market path")
    args = parser.parse_args(argv)

    if not args.all and args.experiment is None:
        parser.error("nothing to do: pass --all or --experiment N")

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    figures_dir = _abspath(ROOT, cfg["output"]["figures_dir"])
    tables_dir = _abspath(ROOT, cfg["output"]["tables_dir"])
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    plotting.apply_paper_style()

    print(f"[config] {args.config}")
    print(f"[seed]   {seed}")
    print("[market] generating synthetic OU market "
          f"({cfg['market']['n_ticks']:,} ticks)...")
    market = generate_market(cfg, seed)

    do = (lambda n: args.all or args.experiment == n)
    written: list[str] = []

    if args.all or args.sample_path:
        written += plotting.plot_sample_path(market, figures_dir)

    # ---- Experiment 1: inverted-U (with multi-seed confidence band) -------
    if do(1):
        # Rich single-seed sweep (per-bot metrics: trade rate, hit rates, ...),
        # also reused by experiment 3's selectivity panel.
        frontier_df = run_frontier_sweep(cfg, market, seed)
        frontier_df.to_csv(os.path.join(tables_dir, "results.csv"), index=False)
        written.append(os.path.join(tables_dir, "results.csv"))

        # Multi-seed sweep across independent markets for the headline band.
        n_seeds = int(cfg["experiments"]["frontier_sweep"].get("n_seeds", 1))
        print(f"[exp 1]  inverted-U sweep across {n_seeds} market realisations...")
        band_df = run_frontier_sweep_multiseed(cfg, seed, n_seeds)
        band_df.to_csv(os.path.join(tables_dir, "results_multiseed.csv"), index=False)
        written.append(os.path.join(tables_dir, "results_multiseed.csv"))
        written += plotting.plot_inverted_u(band_df, figures_dir)

        i = int(band_df["net_pnl"].to_numpy().argmax())
        peak = band_df.iloc[i]
        print(f"         optimum at L={peak['latency_ms']:.2f} ms, "
              f"p={peak['accuracy']:.3f}, mean net PnL={peak['net_pnl']:.3f} "
              f"(95% CI [{peak['net_pnl_lo']:.3f}, {peak['net_pnl_hi']:.3f}])")
    else:
        frontier_df = None

    # ---- Experiment 2: heatmap -------------------------------------------
    if do(2):
        print("[exp 2]  decoupled (accuracy, latency) heatmap...")
        acc, lat, pnl = run_heatmap(cfg, market, seed)
        written += plotting.plot_heatmap(acc, lat, pnl, figures_dir)

    # ---- Experiment 3: signal gating / "right about the past" ------------
    if do(3):
        print("[exp 3]  signal gating: selectivity + believed-vs-realised gap...")
        if frontier_df is None:  # the left panel needs the frontier's trade rates
            frontier_df = run_frontier_sweep(cfg, market, seed)
        gap_df = run_gap_analysis(cfg, market, seed)
        gap_df.to_csv(os.path.join(tables_dir, "gap_analysis.csv"), index=False)
        written.append(os.path.join(tables_dir, "gap_analysis.csv"))
        written += plotting.plot_signal_gating(frontier_df, gap_df, figures_dir)
        # Report where latency drags the realised hit rate below a coin flip.
        below = gap_df[gap_df["realized_hit_rate"] < 0.5]
        if not below.empty:
            L0 = below.iloc[0]["latency_ms"]
            print(f"         realised hit rate falls below 50% at L >= {L0:.1f} ms "
                  f"(believed stays ~{gap_df['believed_hit_rate'].mean():.2f})")

    # ---- Experiment 4: sensitivity ---------------------------------------
    if do(4):
        print("[exp 4]  volatility sensitivity...")
        sweeps = run_sensitivity(cfg, seed)
        written += plotting.plot_sensitivity(sweeps, figures_dir)
        # Record how the sweet spot moves with volatility.
        rows = []
        for mult, df in sorted(sweeps.items()):
            j = int(df["net_pnl"].to_numpy().argmax())
            r = df.iloc[j]
            rows.append({"sigma_multiplier": mult,
                         "optimal_latency_ms": r["latency_ms"],
                         "optimal_accuracy": r["accuracy"],
                         "peak_net_pnl": r["net_pnl"]})
        sens = pd.DataFrame(rows)
        sens.to_csv(os.path.join(tables_dir, "sensitivity.csv"), index=False)
        written.append(os.path.join(tables_dir, "sensitivity.csv"))
        print(sens.to_string(index=False))

    print("\n[done] wrote:")
    for p in written:
        print("   ", os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
