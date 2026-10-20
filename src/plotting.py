"""Publication-quality matplotlib figures (saved as both PDF and PNG).

Style goals: readable fonts, axis labels with units (latency in milliseconds),
legends, no chartjunk, a consistent colour-blind-friendly palette (the
Okabe-Ito set), and vector PDF output for LaTeX ``\\includegraphics``.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless: no display needed, just write files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito colour-blind-safe palette.
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}


def apply_paper_style() -> None:
    """Set global rcParams for clean, consistent, paper-ready figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def _save(fig, figures_dir: str, name: str) -> list[str]:
    """Save a figure as both PDF (vector, for LaTeX) and PNG (preview)."""
    os.makedirs(figures_dir, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = os.path.join(figures_dir, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Figure 0 -- sample market path (sanity / appendix)
# --------------------------------------------------------------------------- #
def plot_sample_path(market, figures_dir: str, n_show: int = 3000) -> list[str]:
    """Plot a slice of the latent fair value and the observed mid price."""
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = np.arange(n_show)
    ax.plot(x, market.price[:n_show], color=OKABE_ITO["sky"], lw=1.0,
            alpha=0.9, label="Observed mid price  P[t]")
    ax.plot(x, market.latent[:n_show], color=OKABE_ITO["vermillion"], lw=1.6,
            label="Latent fair value  X[t]")
    ax.set_xlabel("Tick  (1 tick = 1 ms)")
    ax.set_ylabel("Price")
    ax.set_title("Synthetic market: mean-reverting fair value + microstructure noise")
    ax.legend(loc="best", frameon=True)
    return _save(fig, figures_dir, "fig0_sample_path")


# --------------------------------------------------------------------------- #
# Figure 1 -- the headline inverted-U
# --------------------------------------------------------------------------- #
def plot_inverted_u(df: pd.DataFrame, figures_dir: str) -> list[str]:
    """Net PnL vs latency (left axis) and accuracy vs latency (right axis).

    If the frame carries multi-seed band columns (``net_pnl_p10/p90`` across
    market realisations), the spread is shaded so the hump's robustness shows.
    """
    df = df.sort_values("latency_ms")
    lat = df["latency_ms"].to_numpy()
    pnl = df["net_pnl"].to_numpy()
    acc = df["accuracy"].to_numpy()
    has_band = {"net_pnl_p10", "net_pnl_p90"}.issubset(df.columns)

    i_peak = int(np.argmax(pnl))

    fig, ax1 = plt.subplots(figsize=(7.2, 4.4))

    # Left axis: net PnL (the result we care about).
    if has_band:
        n_seeds = int(df["n_seeds"].iloc[0]) if "n_seeds" in df.columns else 0
        ax1.fill_between(lat, df["net_pnl_p10"].to_numpy(), df["net_pnl_p90"].to_numpy(),
                         color=OKABE_ITO["blue"], alpha=0.18,
                         label=f"10-90th pct across {n_seeds} markets")
        ax1.plot(lat, pnl, color=OKABE_ITO["blue"], marker="o", ms=4,
                 label="Mean net PnL")
    else:
        ax1.plot(lat, pnl, color=OKABE_ITO["blue"], marker="o", ms=4,
                 label="Net PnL")
    ax1.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax1.set_xlabel("Model latency  L  (ms)")
    ax1.set_ylabel("Net PnL  (sum of per-trade returns)", color=OKABE_ITO["blue"])
    ax1.tick_params(axis="y", labelcolor=OKABE_ITO["blue"])

    # Mark the profit-maximising operating point.
    ax1.scatter([lat[i_peak]], [pnl[i_peak]], color=OKABE_ITO["vermillion"],
                zorder=6, s=90, marker="*",
                label=f"Optimum  (L={lat[i_peak]:.1f} ms)")
    ax1.annotate(
        f"sweet spot\nL = {lat[i_peak]:.1f} ms\np = {acc[i_peak]:.2f}",
        xy=(lat[i_peak], pnl[i_peak]),
        xytext=(lat[i_peak] + 0.18 * (lat.max() - lat.min()), pnl[i_peak]),
        fontsize=9, va="center",
        arrowprops=dict(arrowstyle="->", color="grey", lw=1.0),
    )

    # Right axis: nominal accuracy (rises monotonically with latency).
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(lat, acc, color=OKABE_ITO["green"], ls="--", marker="s", ms=3,
             label="Nominal accuracy  p")
    ax2.set_ylabel("Nominal directional accuracy  p", color=OKABE_ITO["green"])
    ax2.tick_params(axis="y", labelcolor=OKABE_ITO["green"])
    ax2.grid(False)

    ax1.set_title("Latency vs. accuracy trade-off: net profit is hump-shaped in latency")

    # Merge legends from both axes.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower center", frameon=True, ncol=1)
    return _save(fig, figures_dir, "fig1_inverted_u")


# --------------------------------------------------------------------------- #
# Figure 2 -- decoupled heatmap
# --------------------------------------------------------------------------- #
def plot_heatmap(accuracies, latencies, pnl, figures_dir: str) -> list[str]:
    """2-D heatmap of net PnL over (latency, accuracy)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Symmetric diverging scale around zero so profit/loss read clearly.
    vmax = float(np.max(np.abs(pnl)))
    mesh = ax.pcolormesh(
        latencies, accuracies, pnl,
        cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
    )
    # Zero-profit contour: the break-even frontier.
    cs = ax.contour(latencies, accuracies, pnl, levels=[0.0],
                    colors="black", linewidths=1.2, linestyles="--")
    ax.clabel(cs, fmt={0.0: "break-even"}, fontsize=8)

    ax.set_xlabel("Latency  L  (ms)")
    ax.set_ylabel("Nominal accuracy  p")
    ax.set_title("Net PnL over the decoupled (latency, accuracy) grid")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Net PnL")

    # Annotate the message: high latency loses even at top accuracy.
    ax.text(0.97, 0.06,
            "high latency loses\neven at high accuracy",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))
    return _save(fig, figures_dir, "fig2_heatmap")


# --------------------------------------------------------------------------- #
# Figure 3 (realistic mode) -- selectivity + the "right about the past" gap
# --------------------------------------------------------------------------- #
def plot_signal_gating(frontier_df: pd.DataFrame, gap_df: pd.DataFrame,
                       figures_dir: str) -> list[str]:
    """Left: the bot trades selectively (trade rate rises with skill).
    Right: at fixed accuracy, the believed hit rate stays high while the
    realised hit rate decays with latency -- the model is 'right about the
    past'."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # --- Left: selectivity -- trade rate vs nominal accuracy ---------------
    f = frontier_df.sort_values("accuracy")
    axL.plot(f["accuracy"], 100.0 * f["trade_rate"], color=OKABE_ITO["purple"],
             marker="o", ms=3)
    axL.set_xlabel("Nominal directional accuracy  p")
    axL.set_ylabel("Share of decisions actually traded  (%)")
    axL.set_title("The bot waits for a signal: selectivity rises with skill")
    axL.set_ylim(bottom=0)
    axL.annotate("a coin-flip model (p~0.5)\nstays flat -- it never\nfinds a tradeable edge",
                 xy=(f["accuracy"].iloc[0], 100.0 * f["trade_rate"].iloc[0]),
                 xytext=(0.40, 0.62), textcoords="axes fraction", fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="grey", lw=1.0))

    # --- Right: believed vs realised hit rate at fixed accuracy ------------
    g = gap_df.sort_values("latency_ms")
    lat = g["latency_ms"].to_numpy()
    believed = 100.0 * g["believed_hit_rate"].to_numpy()
    realized = 100.0 * g["realized_hit_rate"].to_numpy()
    p_fixed = float(g["accuracy"].iloc[0])

    axR.plot(lat, believed, color=OKABE_ITO["green"], marker="s", ms=3,
             label="Believed hit rate (about the t->t+h move)")
    axR.plot(lat, realized, color=OKABE_ITO["vermillion"], marker="o", ms=3,
             label="Realised hit rate (at execution)")
    axR.fill_between(lat, realized, believed, color=OKABE_ITO["orange"],
                     alpha=0.18, label="damage done by latency")
    axR.axhline(50.0, color="grey", lw=0.9, ls=":")
    axR.text(lat[-1], 50.6, "coin flip (50%)", ha="right", va="bottom",
             fontsize=8.5, color="grey")
    axR.set_xlabel("Model latency  L  (ms)")
    axR.set_ylabel("Hit rate  (%)")
    axR.set_title(f"'Right about the past': accuracy fixed at p = {p_fixed:.2f}")
    axR.legend(loc="lower left", frameon=True, fontsize=8.5)
    return _save(fig, figures_dir, "fig3_signal_gating")


# --------------------------------------------------------------------------- #
# Figure (parked) -- SE-technique overlay + bar chart
# --------------------------------------------------------------------------- #
def plot_se_techniques(frontier_df: pd.DataFrame, tech_df: pd.DataFrame,
                       figures_dir: str) -> list[str]:
    """Left: techniques placed on the PnL-vs-latency frontier.
    Right: net PnL per technique as a bar chart (vs. FP32 baseline)."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- Left: the frontier curve with technique markers -------------------
    f = frontier_df.sort_values("latency_ms")
    axL.plot(f["latency_ms"], f["net_pnl"], color=OKABE_ITO["blue"],
             lw=2.0, label="Frontier (swept)")
    axL.axhline(0.0, color="grey", lw=0.8, ls=":")

    marker_colors = [OKABE_ITO["black"], OKABE_ITO["orange"],
                     OKABE_ITO["purple"], OKABE_ITO["green"]]
    markers = ["o", "D", "^", "*"]
    for i, (_, row) in enumerate(tech_df.iterrows()):
        axL.scatter(row["latency_ms"], row["net_pnl"],
                    color=marker_colors[i % len(marker_colors)],
                    marker=markers[i % len(markers)], s=130, zorder=6,
                    edgecolor="white", linewidth=0.8, label=row["technique"])
    axL.set_xlabel("Average latency  L  (ms)")
    axL.set_ylabel("Net PnL")
    axL.set_title("Where SE techniques land on the profit frontier")
    axL.legend(loc="lower left", frameon=True, fontsize=8.5)

    # --- Right: net PnL bar chart -----------------------------------------
    names = tech_df["technique"].tolist()
    pnls = tech_df["net_pnl"].to_numpy()
    colors = [marker_colors[i % len(marker_colors)] for i in range(len(names))]
    bars = axR.bar(range(len(names)), pnls, color=colors, edgecolor="black",
                   linewidth=0.6)
    axR.axhline(pnls[0], color=OKABE_ITO["black"], ls="--", lw=1.0,
                label="FP32 baseline")
    axR.set_xticks(range(len(names)))
    axR.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8.5)
    axR.set_ylabel("Net PnL")
    axR.set_title("Net PnL by SE technique")
    for b, v in zip(bars, pnls):
        axR.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    axR.legend(loc="best", frameon=True)
    return _save(fig, figures_dir, "fig3_se_techniques")


# --------------------------------------------------------------------------- #
# Figure 4 -- volatility sensitivity
# --------------------------------------------------------------------------- #
def plot_sensitivity(sweeps: dict, figures_dir: str) -> list[str]:
    """Family of PnL-vs-latency curves at different volatilities; mark peaks."""
    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    palette = [OKABE_ITO["sky"], OKABE_ITO["blue"],
               OKABE_ITO["orange"], OKABE_ITO["vermillion"]]
    for i, (mult, df) in enumerate(sorted(sweeps.items())):
        d = df.sort_values("latency_ms")
        lat = d["latency_ms"].to_numpy()
        pnl = d["net_pnl"].to_numpy()
        c = palette[i % len(palette)]
        ax.plot(lat, pnl, color=c, marker="o", ms=3,
                label=f"volatility x{mult:g}")
        ipk = int(np.argmax(pnl))
        ax.scatter([lat[ipk]], [pnl[ipk]], color=c, marker="*", s=120,
                   zorder=6, edgecolor="white", linewidth=0.6)

    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Model latency  L  (ms)")
    ax.set_ylabel("Net PnL")
    ax.set_title("Sensitivity: higher volatility shifts the sweet spot toward faster models")
    ax.legend(loc="best", frameon=True, title="Market regime")
    return _save(fig, figures_dir, "fig4_sensitivity")
