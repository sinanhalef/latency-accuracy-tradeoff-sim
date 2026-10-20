"""The four experiments, each returning tidy data for plotting/reporting.

* Experiment 1 -- main inverted-U: sweep the complexity knob along the frontier
  and record net PnL vs latency (and accuracy vs latency).
* Experiment 2 -- decoupled heatmap: break the accuracy<->latency coupling and
  sweep them independently to show that high latency kills profit even at near
  perfect accuracy.
* Experiment 3 -- SE-technique overlay: map quantization / pruning / early-exit
  onto operating points and compare their net PnL to the FP32 baseline.
* Experiment 4 -- sensitivity: re-run the main sweep at several market
  volatilities and show how the profit-maximising latency (the "sweet spot")
  shifts.

All experiments share one generated market (except sensitivity, which must
regenerate the market per volatility level) and use common random numbers, so
results are deterministic in the global seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest, _decision_ticks
from .market import Market, generate_market, ms_to_ticks
from .model import ModelSpec, frontier_points


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _result_row(spec_complexity: float, res: BacktestResult) -> dict:
    """Flatten a :class:`BacktestResult` into a dict row for a DataFrame."""
    return {
        "complexity": spec_complexity,
        "accuracy": res.nominal_accuracy,
        "latency_ms": res.avg_latency_ms,
        "net_pnl": res.net_pnl,
        "pnl_per_trade": res.pnl_per_trade,
        "hit_rate": res.hit_rate,
        "believed_hit_rate": res.believed_hit_rate,
        "n_decisions": res.n_decisions,
        "n_trades": res.n_trades,
        "trade_rate": res.trade_rate,
        "sharpe": res.sharpe,
        "avg_slippage": res.avg_slippage,
    }


def _shared_grid(market: Market, cfg: dict, max_latency_ms: float):
    """Build a decision grid sized for the worst-case latency in a sweep.

    If latency jitter is enabled, a nominal latency can spike up to
    ``tail_mult x`` its value, so the grid reserves room for that worst case.
    """
    h = int(cfg["backtest"]["forward_horizon"])
    stride = int(cfg["backtest"]["decision_stride"])
    jcfg = cfg["backtest"].get("latency_jitter", {}) or {}
    inflate = float(jcfg.get("tail_mult", 1.0)) if jcfg.get("enabled", False) else 1.0
    max_lat_ticks = ms_to_ticks(max_latency_ms * inflate, market.ms_per_tick)
    dt = _decision_ticks(market, h, stride, max_lat_ticks)
    return dt, max_lat_ticks


# --------------------------------------------------------------------------- #
# Experiment 1 -- inverted-U frontier sweep
# --------------------------------------------------------------------------- #
def run_frontier_sweep(cfg: dict, market: Market, seed: int) -> pd.DataFrame:
    """Sweep complexity 0->1 along the frontier; return one row per model."""
    n_points = int(cfg["experiments"]["frontier_sweep"]["n_points"])
    specs = frontier_points(cfg, n_points)
    dt, max_lat_ticks = _shared_grid(market, cfg, float(cfg["frontier"]["L_max"]))

    rows = []
    for s in specs:
        res = run_backtest(
            market, s.accuracy, s.latency_ms, cfg, seed,
            decision_ticks=dt, max_latency_ticks=max_lat_ticks,
        )
        rows.append(_result_row(s.complexity, res))
    return pd.DataFrame(rows)


def run_frontier_sweep_multiseed(cfg: dict, seed: int, n_seeds: int) -> pd.DataFrame:
    """Run the frontier sweep over ``n_seeds`` independent market realisations.

    Each realisation regenerates the market with a different seed (and uses that
    same seed for its accuracy/jitter draws, so within a realisation the bots
    still share common random numbers -- a fair race). Returns one row per bot
    with the mean and a confidence band of net PnL across realisations, so the
    headline hump can be shown to be robust rather than a single lucky path.
    """
    per_seed = []  # list of net_pnl arrays, one per realisation
    base_df = None
    for k in range(n_seeds):
        s = seed + k
        market = generate_market(cfg, s)
        df = run_frontier_sweep(cfg, market, s).sort_values("complexity").reset_index(drop=True)
        per_seed.append(df["net_pnl"].to_numpy())
        if base_df is None:
            base_df = df  # complexity / accuracy / nominal latency are seed-independent

    pnl = np.vstack(per_seed)  # shape (n_seeds, n_bots)
    out = base_df[["complexity", "accuracy", "latency_ms"]].copy()
    out["net_pnl"] = pnl.mean(axis=0)
    out["net_pnl_std"] = pnl.std(axis=0)
    sem = pnl.std(axis=0, ddof=1) / np.sqrt(n_seeds)
    out["net_pnl_lo"] = out["net_pnl"] - 1.96 * sem   # 95% CI of the mean
    out["net_pnl_hi"] = out["net_pnl"] + 1.96 * sem
    out["net_pnl_p10"] = np.percentile(pnl, 10, axis=0)  # spread ACROSS markets
    out["net_pnl_p90"] = np.percentile(pnl, 90, axis=0)
    out["n_seeds"] = n_seeds
    return out


# --------------------------------------------------------------------------- #
# Experiment 3 (realistic mode) -- "right about the past" gap + selectivity
# --------------------------------------------------------------------------- #
def run_gap_analysis(cfg: dict, market: Market, seed: int,
                     fixed_accuracy: float | None = None) -> pd.DataFrame:
    """Hold accuracy fixed and sweep latency to isolate the latency damage.

    At a *fixed* nominal accuracy the model keeps being right about the coming
    ``t -> t+h`` move at the same rate (``believed_hit_rate``, roughly flat),
    but its *realised* hit rate at execution falls as latency grows -- it is
    increasingly "right about the past". The gap between the two curves is the
    pure cost of latency, with accuracy controlled out. Also records the trade
    rate (selectivity) at each latency.
    """
    if fixed_accuracy is None:
        # A capable model near the top of the frontier, so the gap is dramatic.
        fixed_accuracy = float(cfg["frontier"]["p_max"]) - 0.03

    lat_max = float(cfg["frontier"]["L_max"])
    latencies = np.linspace(0.0, lat_max, 31)
    dt, max_lat_ticks = _shared_grid(market, cfg, lat_max)

    rows = []
    for lat in latencies:
        res = run_backtest(
            market, fixed_accuracy, float(lat), cfg, seed,
            decision_ticks=dt, max_latency_ticks=max_lat_ticks,
        )
        rows.append({
            "latency_ms": float(lat),
            "accuracy": fixed_accuracy,
            "believed_hit_rate": res.believed_hit_rate,
            "realized_hit_rate": res.hit_rate,
            "trade_rate": res.trade_rate,
            "net_pnl": res.net_pnl,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Experiment 2 -- decoupled (accuracy, latency) heatmap
# --------------------------------------------------------------------------- #
def run_heatmap(cfg: dict, market: Market, seed: int):
    """Sweep accuracy and latency *independently* on a grid.

    Returns ``(accuracies, latencies_ms, pnl_matrix)`` where ``pnl_matrix`` has
    shape ``(n_accuracy, n_latency)`` so rows index accuracy and columns index
    latency -- ready for ``imshow``/``pcolormesh``.
    """
    hp = cfg["experiments"]["heatmap"]
    accuracies = np.linspace(float(hp["acc_min"]), float(hp["acc_max"]), int(hp["n_accuracy"]))
    latencies = np.linspace(float(hp["lat_min"]), float(hp["lat_max"]), int(hp["n_latency"]))

    dt, max_lat_ticks = _shared_grid(market, cfg, float(hp["lat_max"]))

    pnl = np.empty((accuracies.shape[0], latencies.shape[0]), dtype=np.float64)
    for i, acc in enumerate(accuracies):
        for j, lat in enumerate(latencies):
            res = run_backtest(
                market, float(acc), float(lat), cfg, seed,
                decision_ticks=dt, max_latency_ticks=max_lat_ticks,
            )
            pnl[i, j] = res.net_pnl
    return accuracies, latencies, pnl


# --------------------------------------------------------------------------- #
# Experiment 3 -- software-engineering technique overlay
# --------------------------------------------------------------------------- #
def _technique_latency(tech: dict, n_decisions: int, rng: np.random.Generator):
    """Return a per-decision latency array (ms) for an SE technique.

    Fixed-latency techniques broadcast a scalar. Adaptive techniques
    (``latency_dist``) sample a *per-decision* latency from a mixture, modelling
    early-exit: most decisions exit a shallow layer (low latency) while a few
    hard ones run the full network (high latency). The realised *average*
    latency is therefore much lower than the worst case.
    """
    if "latency_dist" in tech:
        dist = tech["latency_dist"]
        comps = dist["components"]
        weights = np.array([c["weight"] for c in comps], dtype=np.float64)
        weights = weights / weights.sum()
        lat_vals = np.array([c["latency_ms"] for c in comps], dtype=np.float64)
        choice = rng.choice(len(comps), size=n_decisions, p=weights)
        return lat_vals[choice]
    return float(tech["latency_ms"])


def run_se_techniques(cfg: dict, market: Market, seed: int) -> pd.DataFrame:
    """Evaluate each configured SE technique; return a summary table."""
    techs = cfg["se_techniques"]

    # Worst-case latency across all techniques sizes the shared decision grid.
    def _max_lat(t: dict) -> float:
        if "latency_dist" in t:
            return max(c["latency_ms"] for c in t["latency_dist"]["components"])
        return float(t["latency_ms"])

    max_latency_ms = max(_max_lat(t) for t in techs)
    dt, max_lat_ticks = _shared_grid(market, cfg, max_latency_ms)
    n_dec = dt.shape[0]

    rows = []
    for t in techs:
        # Separate RNG (deterministic) for the per-decision latency draw, kept
        # distinct from the accuracy-draw RNG inside run_backtest.
        lat_rng = np.random.default_rng(seed + 1)
        latency = _technique_latency(t, n_dec, lat_rng)
        res = run_backtest(
            market, float(t["accuracy"]), latency, cfg, seed,
            decision_ticks=dt, max_latency_ticks=max_lat_ticks,
        )
        rows.append({
            "technique": t["name"],
            "accuracy": res.nominal_accuracy,
            "latency_ms": res.avg_latency_ms,
            "net_pnl": res.net_pnl,
            "pnl_per_trade": res.pnl_per_trade,
            "hit_rate": res.hit_rate,
            "n_trades": res.n_trades,
            "sharpe": res.sharpe,
        })
    df = pd.DataFrame(rows)

    # Net PnL relative to the first technique (assumed FP32 baseline).
    baseline_pnl = df.iloc[0]["net_pnl"]
    df["pnl_vs_baseline"] = df["net_pnl"] - baseline_pnl
    return df


# --------------------------------------------------------------------------- #
# Experiment 4 -- volatility sensitivity of the sweet spot
# --------------------------------------------------------------------------- #
def run_sensitivity(cfg: dict, seed: int) -> dict[float, pd.DataFrame]:
    """Re-run the frontier sweep at several volatilities.

    Returns a mapping ``sigma_multiplier -> frontier-sweep DataFrame``. Each
    market is regenerated with the scaled OU volatility (same seed, so it is the
    "same market, more/less volatile") -- this keeps everything reproducible
    while letting the sweet spot move.
    """
    multipliers = cfg["experiments"]["sensitivity"]["sigma_multipliers"]
    base_sigma = float(cfg["market"]["sigma"])

    out: dict[float, pd.DataFrame] = {}
    for mult in multipliers:
        scaled = {**cfg, "market": {**cfg["market"], "sigma": base_sigma * float(mult)}}
        market = generate_market(scaled, seed)
        out[float(mult)] = run_frontier_sweep(scaled, market, seed)
    return out
