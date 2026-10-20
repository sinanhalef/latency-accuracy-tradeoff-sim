"""Event-driven, latency-aware backtest engine -- the trade-off mechanism.

READ THIS to understand the whole experiment.

A signal is *decided* at tick ``t`` using only information available at ``t``
(the model predicts the sign of the latent fair-value move from ``t`` to
``t + h``). But the order is *filled* at tick ``t + L`` -- after the model's
inference latency ``L`` -- at the price ``P[t + L]`` prevailing then, and the
position is closed ``h`` ticks later at ``P[t + L + h]``:

    direction   = predicted_sign(decided at t)          # commitment made at t
    entry_price = P[t + L]                               # we are filled only after latency L
    exit_price  = P[t + L + h]                           # close after holding horizon h
    pnl_return  = direction * (exit_price/entry_price - 1) - costs

The key consequence: a *slow* model (large ``L``) commits to a direction based
on stale information and is filled at a price that has already moved. Because
the market mean-reverts at short horizons, the move it predicted is partly over
(or reversing) by the time it enters, so its *effective* hit rate at execution
is well below its nominal accuracy ``p`` -- even though ``p`` is high. A *fast*
model is filled almost immediately but is less accurate. Net profit is
therefore hump-shaped in latency: both extremes lose, the optimum is interior.

Everything here is vectorised with numpy so a full frontier sweep runs in
seconds. The accuracy random draws use *common random numbers* (the same seed
each call) so that, across a sweep, a more accurate model is always at least as
correct on every individual decision -- this removes Monte-Carlo jitter and
makes the frontier curve smooth and the comparison fair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .market import Market, ms_to_ticks
from .model import generate_signal, predict_directions


@dataclass(frozen=True)
class BacktestResult:
    """Aggregated metrics from one model's run over the whole market."""

    net_pnl: float          # total net PnL summed over EXECUTED trades (return units)
    pnl_per_trade: float    # mean net PnL per executed trade
    hit_rate: float         # fraction of executed trades profitable AT EXECUTION (differs from nominal p)
    n_trades: int           # number of trades ACTUALLY TAKEN (after signal gating)
    sharpe: float           # mean / std of per-trade net PnL (per-trade Sharpe-like ratio)
    avg_slippage: float     # mean |entry - decision price| / decision price (realised slippage)
    nominal_accuracy: float # the input p
    avg_latency_ms: float   # realised average latency in ms (== L for fixed-latency models)
    n_decisions: int = 0    # number of decision opportunities considered
    trade_rate: float = 1.0 # fraction of decisions that became trades (selectivity)
    believed_hit_rate: float = float("nan")  # fraction right about the t->t+h move (pre-latency)
    # Optional per-trade timeline (populated only when return_timeline=True):
    trade_ticks: np.ndarray | None = None  # decision tick of each executed trade
    trade_pnl: np.ndarray | None = None    # net PnL of each executed trade (same order)


def _decision_ticks(market: Market, horizon: int, stride: int, max_latency_ticks: int) -> np.ndarray:
    """Tick indices at which decisions are made.

    Decisions start at tick 0 and are spaced ``stride`` ticks apart. We stop
    early enough that every trade can both be filled (``t + L``) and closed
    (``t + L + h``) within the series, using the *largest* latency we will test
    so all models in a sweep share the same decision grid (fair comparison).
    """
    last_valid = market.n_ticks - max_latency_ticks - horizon - 1
    if last_valid <= 0:
        raise ValueError(
            "Market too short for the requested horizon/latency. "
            "Increase market.n_ticks or reduce forward_horizon / L_max."
        )
    return np.arange(0, last_valid, stride, dtype=np.int64)


def _jittered_latency(nominal_ms: float, n: int, jcfg: dict, seed: int) -> np.ndarray:
    """Per-decision actual latency (ms) around a nominal value.

    Models real latency as a fat-tailed distribution: a lognormal multiplicative
    jitter around ``nominal_ms`` (compute/scheduling variation) plus an occasional
    spike (GC pause, network/queueing). Hard-capped at ``tail_mult * nominal_ms``
    so a fill can never run off the reserved grid. Uses a dedicated RNG stream
    (``seed + 7919``) so jitter is reproducible and common across bots without
    disturbing the accuracy/signal draws.
    """
    rng = np.random.default_rng(seed + 7919)
    rel_sigma = float(jcfg.get("rel_sigma", 0.0))
    tail_prob = float(jcfg.get("tail_prob", 0.0))
    tail_mult = float(jcfg.get("tail_mult", 1.0))

    mult = np.exp(rng.normal(0.0, rel_sigma, n))      # lognormal jitter, mean ~1
    spikes = rng.random(n) < tail_prob                # occasional latency spikes
    mult = np.where(spikes, mult * tail_mult, mult)
    mult = np.clip(mult, 0.0, tail_mult)              # cap at the worst case
    return nominal_ms * mult


def run_backtest(
    market: Market,
    accuracy: float,
    latency_ms: float | np.ndarray,
    cfg: dict,
    seed: int,
    decision_ticks: np.ndarray | None = None,
    max_latency_ticks: int | None = None,
    return_timeline: bool = False,
) -> BacktestResult:
    """Run one abstract model ``(accuracy, latency)`` over the market.

    Parameters
    ----------
    market:
        The generated :class:`~src.market.Market`.
    accuracy:
        Nominal directional accuracy ``p`` in ``[0.5, 1.0]``.
    latency_ms:
        Either a scalar latency in ms (fixed-latency model) or a per-decision
        array of latencies in ms (adaptive models such as early-exit). For the
        array case its length must equal the number of decision ticks.
    cfg:
        Full config dict (we read ``cfg['backtest']``).
    seed:
        Seed for the accuracy draws. Passing the *same* seed across a sweep
        gives common random numbers (smooth, fair frontier).
    decision_ticks, max_latency_ticks:
        Optional precomputed decision grid / worst-case latency, shared across a
        sweep so all models trade at exactly the same instants.
    """
    bt = cfg["backtest"]
    h = int(bt["forward_horizon"])
    stride = int(bt["decision_stride"])
    fee = float(bt["fee_bps"]) * 1e-4
    spread = float(bt.get("spread_bps", 0.0)) * 1e-4
    half_spread = 0.5 * spread          # crossed once on entry, once on exit
    slip_coeff = float(bt["slippage_coeff"]) * 1e-4
    predict_on_latent = bool(bt.get("predict_on_latent", True))
    jcfg = bt.get("latency_jitter", {}) or {}
    jitter_on = bool(jcfg.get("enabled", False))
    tail_mult = float(jcfg.get("tail_mult", 1.0)) if jitter_on else 1.0

    ms_per_tick = market.ms_per_tick

    # --- Resolve latency (scalar or per-decision array) into integer ticks ---
    # With jitter on, a scalar nominal latency L can spike up to `tail_mult * L`,
    # so the decision grid must reserve room for that worst case (otherwise a
    # spiked fill could run off the end of the price series).
    if np.isscalar(latency_ms):
        nominal_max_ticks = ms_to_ticks(float(latency_ms) * tail_mult, ms_per_tick)
        if max_latency_ticks is None:
            max_latency_ticks = nominal_max_ticks
    else:
        latency_ms = np.asarray(latency_ms, dtype=np.float64)
        if max_latency_ticks is None:
            max_latency_ticks = int(
                max(ms_to_ticks(float(x), ms_per_tick) for x in latency_ms)
            )

    if decision_ticks is None:
        decision_ticks = _decision_ticks(market, h, stride, int(max_latency_ticks))

    t = decision_ticks
    n = t.shape[0]

    # Per-decision latency in ticks.
    if np.isscalar(latency_ms):
        if jitter_on:
            # Fat-tailed per-decision latency: a lognormal jitter around L plus an
            # occasional spike, hard-capped at tail_mult * L. Its own RNG stream so
            # it does not perturb the accuracy/signal draws.
            actual_ms = _jittered_latency(float(latency_ms), n, jcfg, seed)
            lat_ticks = np.minimum(
                np.round(actual_ms / ms_per_tick).astype(np.int64), int(max_latency_ticks)
            )
            avg_latency_ms = float(np.mean(actual_ms))
        else:
            lat_ticks = np.full(n, ms_to_ticks(float(latency_ms), ms_per_tick), dtype=np.int64)
            avg_latency_ms = float(latency_ms)
    else:
        if latency_ms.shape[0] != n:
            raise ValueError("per-decision latency array length must match #decisions")
        lat_ticks = np.array(
            [ms_to_ticks(float(x), ms_per_tick) for x in latency_ms], dtype=np.int64
        )
        lat_ticks = np.minimum(lat_ticks, int(max_latency_ticks))
        avg_latency_ms = float(np.mean(latency_ms))

    # --- The true near-future move the model is trying to predict ----------
    # Signal of interest: the latent fair value (or observed mid) move over the
    # forward horizon h, evaluated at DECISION time t. (The bot decides on this;
    # it is then filled L ticks later -- the latency penalty.)
    signal_series = market.latent if predict_on_latent else market.price
    true_move = signal_series[t + h] - signal_series[t]
    true_sign = np.sign(true_move)
    true_sign[true_sign == 0] = 1  # break ties deterministically

    rng = np.random.default_rng(seed)
    sig_cfg = cfg.get("signal", {}) or {}
    gating = bool(sig_cfg.get("enabled", False))

    if gating:
        # --- Realistic mode: a continuous signal the bot decides to act on --
        # The model emits a noisy estimate `g` of the coming move; its sign is
        # the predicted direction and |g| is its confidence.
        g, rho, sd = generate_signal(true_move, accuracy, rng)
        direction = np.sign(g)
        direction[direction == 0] = 1

        # The bot's BELIEVED expected edge for this trade, in return units:
        # E[true_move | g] = rho * g * sd  (price units) / decision price.
        # This is what a real system estimates BEFORE trading -- it cannot see
        # that latency will later erode the realised move.
        decision_price0 = market.price[t]
        believed_edge = np.abs(rho * g * sd) / decision_price0

        # Gate: only trade when the believed edge clears costs (or |signal|
        # clears a z-threshold). The cost the bot anticipates is the round-trip
        # spread plus the exchange fee. Otherwise the bot stays flat.
        round_trip_cost = fee + spread
        mode = str(sig_cfg.get("threshold_mode", "cost_aware"))
        if mode == "zscore":
            trade_mask = np.abs(g) >= float(sig_cfg.get("z_threshold", 0.6))
        else:  # cost_aware
            edge_multiple = float(sig_cfg.get("edge_multiple", 1.0))
            trade_mask = believed_edge >= edge_multiple * round_trip_cost
    else:
        # --- Legacy mode: binary prediction, taken on EVERY decision tick ---
        direction = predict_directions(true_sign, accuracy, rng).astype(np.float64)
        trade_mask = np.ones(n, dtype=bool)

    # --- Restrict to the trades the bot actually takes ----------------------
    idx = np.flatnonzero(trade_mask)
    n_trades = int(idx.shape[0])

    if n_trades == 0:
        # The bot never found a tradeable signal: flat all day, zero PnL.
        return BacktestResult(
            net_pnl=0.0, pnl_per_trade=0.0, hit_rate=float("nan"), n_trades=0,
            sharpe=0.0, avg_slippage=0.0, nominal_accuracy=float(accuracy),
            avg_latency_ms=avg_latency_ms, n_decisions=n, trade_rate=0.0,
            believed_hit_rate=float("nan"),
            trade_ticks=(np.empty(0, dtype=np.int64) if return_timeline else None),
            trade_pnl=(np.empty(0, dtype=np.float64) if return_timeline else None),
        )

    t_g = t[idx]
    dir_g = direction[idx].astype(np.float64)
    lat_g = lat_ticks[idx]

    # --- Latency-aware execution: fill at t+L, close at t+L+h --------------
    price = market.price
    decision_price = price[t_g]        # mid price observed when the decision was made
    entry_idx = t_g + lat_g            # <-- the latency penalty: filled L ticks late
    entry_mid = price[entry_idx]
    exit_mid = price[entry_idx + h]

    # Cross the bid/ask spread: you BUY at the ask (mid + half-spread) and SELL at
    # the bid (mid - half-spread). For either direction you give up half the
    # spread entering and half exiting -> a full spread paid round-trip.
    entry_price = entry_mid * (1.0 + dir_g * half_spread)
    exit_price = exit_mid * (1.0 - dir_g * half_spread)

    # Per-trade return captured between (late) entry and exit, net of the spread
    # (which is already in the fill prices) and the exchange fee.
    trade_return = dir_g * (exit_price / entry_price - 1.0)
    cost = fee + slip_coeff * lat_g
    pnl = trade_return - cost

    # Realised slippage: how far the fill mid drifted from the decision mid purely
    # because of latency (a diagnostic, not charged twice).
    slippage = np.abs(entry_mid - decision_price) / decision_price

    # --- Aggregate metrics over EXECUTED trades -----------------------------
    pnl_per_trade = float(np.mean(pnl))
    std = float(np.std(pnl))
    return BacktestResult(
        net_pnl=float(np.sum(pnl)),
        pnl_per_trade=pnl_per_trade,
        # Realised hit rate: did the market move our way between entry and exit?
        # (Measured on mid prices -- directional correctness, before the spread.)
        hit_rate=float(np.mean((dir_g * (exit_mid - entry_mid)) > 0.0)),
        n_trades=n_trades,
        sharpe=float(pnl_per_trade / std) if std > 0 else 0.0,
        avg_slippage=float(np.mean(slippage)),
        nominal_accuracy=float(accuracy),
        avg_latency_ms=avg_latency_ms,
        n_decisions=n,
        trade_rate=float(n_trades / n),
        # Believed hit rate: was the bot right about the t->t+h move it bet on,
        # IGNORING latency? The gap to hit_rate is the damage latency does.
        believed_hit_rate=float(np.mean(dir_g == true_sign[idx])),
        # Per-trade timeline (for plotting a single bot's PnL over time).
        trade_ticks=(t_g.copy() if return_timeline else None),
        trade_pnl=(pnl.copy() if return_timeline else None),
    )
