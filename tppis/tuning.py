"""Joint grid search over ``d``, ``alpha`` and ``k``.

Invalid combinations violating ``0 <= d < m <= r`` are recorded and skipped
(A-2). An entirely empty grid raises :class:`TPPISError`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from tppis._validation import rank_by_abs
from tppis.criteria import CriterionSweep, default_k_max, incremental_refit
from tppis.exceptions import EmptyGridError, TPPISError
from tppis.factors import ratio_d_from_spec
from tppis.kernel import column_energy, screening_scores
from tppis.spectral import SpectralFactorization, split_blocks

MethodName = Literal["sis", "fpsis", "fpsis_bic", "ppis", "tppis"]

DEFAULT_D_FRACTIONS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_ALPHAS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class FitResult:
    """Outcome of a single ``fit``, including the diagnostic grid."""

    omega: NDArray[np.float64]
    order: NDArray[np.intp]
    k: int
    d: int | None
    alpha: float | None
    bic: float
    beta: NDArray[np.float64]
    grid: list[dict[str, Any]] = field(default_factory=list)
    n_skipped: int = 0


def d_candidates(
    spec: SpectralFactorization,
    d: int | str,
    n: int,
    fractions: Sequence[float] | None = None,
) -> list[int]:
    """Resolve the ``d`` axis of the grid, including the ratio estimator (3)."""
    if isinstance(d, (int, np.integer)):
        return [int(d)]
    if d == "ratio":
        return [ratio_d_from_spec(spec)]
    if d != "grid":
        raise TPPISError(f"d must be an int, 'ratio' or 'grid', got {d!r}.")
    values = {int(np.floor(frac * n)) for frac in (fractions or DEFAULT_D_FRACTIONS)}
    values.add(ratio_d_from_spec(spec))
    return sorted(values)


def alpha_candidates(
    alpha: float | str,
    grid: Sequence[float] | None = None,
) -> list[float]:
    """Resolve the ``alpha`` axis of the grid."""
    if isinstance(alpha, (int, float, np.floating)) and not isinstance(alpha, bool):
        return [float(alpha)]
    if alpha != "grid":
        raise TPPISError(f"alpha must be a float in (0, 1] or 'grid', got {alpha!r}.")
    return [float(a) for a in (grid or DEFAULT_ALPHAS)]


def _method_index(method: MethodName, blocks: Any) -> tuple[slice, int]:
    if method in {"fpsis", "fpsis_bic"}:
        return blocks.tail, 1
    if method == "ppis":
        return blocks.tail, -1
    if method == "tppis":
        return blocks.retained, -1
    raise TPPISError(f"Method {method!r} has no spectral index set.")


def _refit_from_omega(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    omega: NDArray[np.float64],
    order: NDArray[np.intp],
    k_max: int,
    *,
    k: int | str,
    bic_kind: str,
) -> tuple[int, float, NDArray[np.float64], CriterionSweep | None]:
    del omega  # ranking score; the criterion refits on the original columns
    cols = order[:k_max]
    X_sel = X[:, cols]
    # Ordinary least squares on the original columns. The published BIC is this
    # residual. Equation (8)'s transformed coefficient does not produce it.
    F = X_sel.T
    rhs = F @ y
    if isinstance(k, (int, np.integer)):
        n = int(X.shape[0])
        if int(k) >= n:
            raise TPPISError(
                f"k={int(k)} is not allowed for n={n}. Use 1 <= k <= {n - 1}. "
                "k <= p-1 does not keep least squares from interpolating y."
            )
        k_use = min(int(k), k_max)
        if k_use < 1:
            raise TPPISError(f"k must be at least 1, got {k}.")
        gram = F[:k_use] @ F[:k_use].T
        beta, *_ = np.linalg.lstsq(gram, rhs[:k_use], rcond=None)
        resid = y - X_sel[:, :k_use] @ beta
        rss = float(resid @ resid)
        from tppis.criteria import bic_value

        return (
            k_use,
            bic_value(rss, k_use, X.shape[0], X.shape[1], kind=bic_kind),
            beta,
            None,
        )
    sweep = incremental_refit(F, rhs, X_sel, y, X.shape[1], kind=bic_kind)
    return sweep.best_k, sweep.best_bic, sweep.best_beta, sweep


def _search_cap(cap: int, n: int, k: int | str) -> int:
    """Drop the interpolating size from a BIC search.

    After centering, ``k = n - 1`` drives the residual to zero and the
    criterion to ``-inf``. The minimum on ``1 .. n-2`` is the model the
    tables report. A user-chosen integer ``k`` is left alone.
    """
    if isinstance(k, (int, np.integer)) or n <= 2:
        return cap
    return min(cap, n - 2)


def evaluate(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    spec: SpectralFactorization,
    method: MethodName,
    *,
    d: int,
    alpha: float,
    k: int | str,
    k_max: int | None,
    bic_kind: str,
    rounding: str,
) -> tuple[NDArray[np.float64], NDArray[np.intp], int, float, NDArray[np.float64], int]:
    """Score one ``(d, alpha)`` pair and optionally sweep ``k``.

    Returns
    -------
    omega, order, k_star, bic, beta, n_retained
    """
    if method == "sis":
        omega = X.T @ y
        order = rank_by_abs(omega)
        cap = _search_cap(
            default_k_max(spec.p, spec.n, min(spec.n, spec.p), k_max),
            spec.n,
            k,
        )
        k_star, bic, beta, _ = _refit_from_omega(
            X, y, omega, order, cap, k=k, bic_kind=bic_kind
        )
        return omega, order, k_star, bic, beta, min(spec.n, spec.p)

    blocks = split_blocks(spec, d, alpha, rounding=rounding)
    index, exponent = _method_index(method, blocks)
    raw = screening_scores(X, y, spec, index, exponent)
    energy = np.maximum(column_energy(X, spec, index, exponent), 1e-18)
    # Zhao et al. (2020) eq. (15): marginal coefficient, not the raw dot product.
    omega = raw / energy
    order = rank_by_abs(omega)
    cap = _search_cap(
        default_k_max(
            spec.p,
            spec.n,
            blocks.n_retained if method == "tppis" else (spec.r - blocks.d),
            k_max,
        ),
        spec.n,
        k,
    )
    k_star, bic, beta, _ = _refit_from_omega(
        X, y, omega, order, cap, k=k, bic_kind=bic_kind
    )
    return omega, order, k_star, bic, beta, cap


def search(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    spec: SpectralFactorization,
    method: MethodName,
    *,
    d: int | str,
    alpha: float | str,
    k: int | str,
    k_max: int | None,
    bic_kind: str,
    rounding: str,
    d_fractions: Sequence[float] | None = None,
    alpha_grid: Sequence[float] | None = None,
) -> FitResult:
    """Joint search over the axes that the method actually uses."""
    grid: list[dict[str, Any]] = []
    if method == "sis":
        omega, order, k_star, bic, beta, _ = evaluate(
            X,
            y,
            spec,
            method,
            d=0,
            alpha=1.0,
            k=k,
            k_max=k_max,
            bic_kind=bic_kind,
            rounding=rounding,
        )
        grid.append(
            {
                "d": None,
                "alpha": None,
                "k": k_star,
                "bic": bic,
                "skipped": False,
                "reason": None,
            }
        )
        return FitResult(omega, order, k_star, None, None, bic, beta, grid, 0)

    d_vals = d_candidates(spec, d, spec.n, fractions=d_fractions)
    a_vals = [1.0] if method != "tppis" else alpha_candidates(alpha, alpha_grid)
    n_skipped = 0
    best: FitResult | None = None

    for d_i in d_vals:
        for a_i in a_vals:
            try:
                omega, order, k_star, bic, beta, _ = evaluate(
                    X,
                    y,
                    spec,
                    method,
                    d=d_i,
                    alpha=a_i,
                    k=k,
                    k_max=k_max,
                    bic_kind=bic_kind,
                    rounding=rounding,
                )
            except TPPISError as exc:
                grid.append(
                    {
                        "d": d_i,
                        "alpha": a_i,
                        "k": None,
                        "bic": None,
                        "skipped": True,
                        "reason": str(exc),
                    }
                )
                n_skipped += 1
                continue
            grid.append(
                {
                    "d": d_i,
                    "alpha": a_i,
                    "k": k_star,
                    "bic": bic,
                    "skipped": False,
                    "reason": None,
                }
            )
            if best is None or bic < best.bic:
                best = FitResult(
                    omega, order, k_star, d_i, a_i, bic, beta, [], n_skipped
                )

    if best is None:
        raise EmptyGridError(
            "Every (d, alpha) combination was invalid. "
            "Loosen the grid or check that 0 <= d < [n*alpha] <= r."
        )
    best.grid = grid
    best.n_skipped = n_skipped
    return best


def iter_valid_pairs(
    spec: SpectralFactorization,
    d_vals: Iterable[int],
    alpha_vals: Iterable[float],
    rounding: str,
) -> list[tuple[int, float]]:
    """Return the ``(d, alpha)`` pairs that satisfy A-2, for diagnostics."""
    valid: list[tuple[int, float]] = []
    for d_i in d_vals:
        for a_i in alpha_vals:
            try:
                split_blocks(spec, d_i, a_i, rounding=rounding)
            except TPPISError:
                continue
            valid.append((d_i, a_i))
    return valid
