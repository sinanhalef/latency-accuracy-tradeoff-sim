"""The abstract AI model: a single point ``(accuracy, latency)``.

The central abstraction of this project is that we do **not** train a neural
network. Instead each "AI model" is summarised by exactly two numbers:

* ``p`` -- directional accuracy in ``[0.5, 1.0]``: the probability that the
  model's up/down call matches the *true* near-future direction of the latent
  fair value. Accuracy is a tunable knob, not an emergent property.
* ``L`` -- inference latency in milliseconds (converted to integer ticks by the
  backtest).

Real systems cannot pick ``(p, L)`` freely: more accurate models are usually
slower. We encode that constraint with a *frontier* mapping a single
"complexity" knob ``c in [0, 1]`` to a ``(p, L)`` pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    """An abstract model as a point on the accuracy/latency plane.

    Parameters
    ----------
    accuracy:
        Nominal directional accuracy ``p`` in ``[0.5, 1.0]``.
    latency_ms:
        Mean inference latency in milliseconds (used for plotting / reporting
        and, for fixed-latency models, for execution).
    complexity:
        The frontier knob ``c`` that produced this point (``nan`` if the point
        was specified directly, e.g. an SE technique).
    label:
        Optional human-readable name (e.g. ``"Quantization (INT8)"``).
    """

    accuracy: float
    latency_ms: float
    complexity: float = float("nan")
    label: str = ""


def accuracy_from_complexity(c: np.ndarray, fr: dict) -> np.ndarray:
    """Accuracy as a saturating function of complexity ``c``.

    ``p(c) = p_min + (p_max - p_min) * (1 - exp(-k c))``. Accuracy rises with
    complexity but saturates: doubling model size yields diminishing accuracy
    gains, which is exactly why the latency cost eventually dominates.
    """
    p_min = float(fr["p_min"])
    p_max = float(fr["p_max"])
    k = float(fr["k"])
    return p_min + (p_max - p_min) * (1.0 - np.exp(-k * c))


def latency_from_complexity(c: np.ndarray, fr: dict) -> np.ndarray:
    """Latency (ms) as a growing function of complexity ``c``.

    ``L(c) = L_min + (L_max - L_min) * c**gamma``. With ``gamma > 1`` latency
    accelerates for the most complex models (deeper nets, more parameters).
    """
    L_min = float(fr["L_min"])
    L_max = float(fr["L_max"])
    gamma = float(fr["gamma"])
    return L_min + (L_max - L_min) * np.power(c, gamma)


def frontier_points(cfg: dict, n_points: int) -> list[ModelSpec]:
    """Sample ``n_points`` models evenly along the complexity frontier.

    Returns a list of :class:`ModelSpec` ordered by increasing complexity
    (hence increasing accuracy *and* increasing latency).
    """
    fr = cfg["frontier"]
    c = np.linspace(0.0, 1.0, n_points)
    p = accuracy_from_complexity(c, fr)
    L = latency_from_complexity(c, fr)
    return [
        ModelSpec(accuracy=float(pi), latency_ms=float(Li), complexity=float(ci))
        for ci, pi, Li in zip(c, p, L)
    ]


def predict_directions(
    true_sign: np.ndarray, accuracy: float, rng: np.random.Generator
) -> np.ndarray:
    """Emit directional predictions at the given nominal accuracy.

    Given the *true* direction of the latent fair-value move (``true_sign`` in
    ``{-1, +1}``), the model returns the correct sign with probability
    ``accuracy`` and the opposite sign otherwise. This is the clean abstraction
    of "a model that is right ``p`` of the time".

    Note: this is the accuracy measured at *decision* time against the latent
    move. The accuracy realised at *execution* time (after latency) is lower --
    that gap is what the backtest measures and what drives the inverted-U.
    """
    n = true_sign.shape[0]
    correct = rng.random(n) < accuracy
    # where correct -> keep true sign; else flip it.
    return np.where(correct, true_sign, -true_sign)


def accuracy_to_skill(accuracy: float) -> float:
    """Convert a nominal directional accuracy ``p`` to a signal-skill ``rho``.

    In the realistic execution mode the model does not emit a binary right/wrong
    call; it emits a *continuous, noisy estimate* of the coming move. We model
    that estimate ``g`` and the (standardised) true move ``u`` as jointly
    Gaussian with correlation ``rho`` -- so ``rho`` is the model's "skill".

    For two zero-mean jointly-Gaussian variables, the probability that they share
    the same sign is ``1/2 + arcsin(rho)/pi``. Setting that equal to the desired
    directional accuracy ``p`` and inverting gives

        rho = sin( pi * (p - 1/2) ).

    So ``p = 0.5`` -> ``rho = 0`` (no skill, pure noise) and ``p = 1.0`` ->
    ``rho = 1`` (the signal *is* the true move). This keeps the frontier
    parametrised by the same interpretable accuracy ``p`` as before, while now
    yielding a graded signal whose *strength* the bot can threshold on.
    """
    p = float(np.clip(accuracy, 0.5, 1.0))
    return float(np.sin(np.pi * (p - 0.5)))


def generate_signal(
    true_move: np.ndarray, accuracy: float, rng: np.random.Generator
) -> tuple[np.ndarray, float, float]:
    """Emit a continuous predictive signal of the coming move.

    Returns ``(g, rho, sd)`` where ``g`` is the model's standardised signal
    (correlated with the true move at level ``rho``), ``rho`` is the skill, and
    ``sd`` is the standard deviation of the true forward move (used to turn the
    standardised signal back into a believed move in price units).

    The sign of ``g`` is the predicted direction; the magnitude ``|g|`` is the
    model's *confidence*. A bot that "waits for a signal" trades only when that
    confidence implies an expected edge larger than its costs.
    """
    sd = float(np.std(true_move))
    if sd <= 0.0:
        sd = 1.0
    u = true_move / sd  # standardised true forward move
    rho = accuracy_to_skill(accuracy)
    e = rng.standard_normal(true_move.shape[0])
    # g has unit variance and correlation rho with the true move u.
    g = rho * u + np.sqrt(max(0.0, 1.0 - rho * rho)) * e
    return g, rho, sd
