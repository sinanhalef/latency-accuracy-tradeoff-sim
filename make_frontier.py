"""Standalone figure: the accuracy<->latency FRONTIER itself.

This plots the raw trade-off that the whole experiment assumes: more complex
models are more accurate but slower. Each bot is a point ``(latency, accuracy)``
sampled along the complexity knob ``c in [0, 1]`` (same frontier the backtest
uses). No PnL, no trading -- just the constraint.

    python make_frontier.py
"""

from __future__ import annotations

import numpy as np
import yaml

from src.model import accuracy_from_complexity, latency_from_complexity
from src import plotting
from src.plotting import OKABE_ITO


def main() -> None:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    plotting.apply_paper_style()
    import matplotlib.pyplot as plt

    fr = cfg["frontier"]
    n_points = int(cfg["experiments"]["frontier_sweep"]["n_points"])

    # Smooth frontier curve (many points) + the actual sampled bots (n_points).
    c_dense = np.linspace(0.0, 1.0, 400)
    L_dense = latency_from_complexity(c_dense, fr)
    p_dense = accuracy_from_complexity(c_dense, fr)

    c_bots = np.linspace(0.0, 1.0, n_points)
    L_bots = latency_from_complexity(c_bots, fr)
    p_bots = accuracy_from_complexity(c_bots, fr)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    # The frontier curve.
    ax.plot(L_dense, p_dense, color=OKABE_ITO["green"], lw=2.2, zorder=2,
            label="Accuracy-latency frontier")
    # The individual bots that race in Figure 1.
    ax.scatter(L_bots, p_bots, s=28, color=OKABE_ITO["green"],
               edgecolor="white", linewidth=0.6, zorder=3,
               label=f"Sampled models (n={n_points})")

    # Label the two extremes so the trade-off reads instantly.
    ax.annotate("fast but inaccurate\n(barely better than a coin)",
                xy=(L_bots[0], p_bots[0]), xytext=(L_bots[0] + 4.0, p_bots[0] + 0.005),
                fontsize=8.5, color="grey", va="bottom")
    ax.annotate("accurate but slow",
                xy=(L_bots[-1], p_bots[-1]), xytext=(L_bots[-1] - 0.5, p_bots[-1] - 0.01),
                fontsize=8.5, color="grey", ha="right", va="top")

    ax.set_xlabel("Model latency  L  (ms)")
    ax.set_ylabel("Nominal directional accuracy  p")
    ax.set_title("The accuracy-latency frontier: more accuracy costs latency")
    ax.legend(loc="lower right", framealpha=0.9)

    paths = plotting._save(fig, cfg["output"]["figures_dir"], "fig_frontier")
    for p in paths:
        print(f"[written] {p}")


if __name__ == "__main__":
    main()
