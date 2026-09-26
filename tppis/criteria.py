"""Refit and the BIC-type criterion (10).

The coefficient inside the criterion is ordinary least squares of the original
columns on the centered response. That is the residual whose value matches
Tables 1–4. Equation (8) fits the transformed design instead, and that residual
does not.

The Gram matrices for ``k = 1, 2, ...`` are nested leading blocks, so an
incremental Cholesky sweep produces every ``beta_hat(M_k)`` in ``O(K^3 / 3)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_triangular

from tppis._validation import check_literal
from tppis.exceptions import BoundarySelectionWarning, TPPISError, TppisWarning

# Floor for log(RSS). Combined with A-9 this keeps BIC finite.
_RSS_FLOOR = 1e-300
# Relative pivot tolerance for the incremental Cholesky path.
_CHOL_EPS = 1e-12


@dataclass(frozen=True)
class CriterionSweep:
    """BIC path over ``k = 1 .. K`` for one ``(d, alpha)`` pair."""

    k: NDArray[np.intp]
    bic: NDArray[np.float64]
    rss: NDArray[np.float64]
    best_k: int
    best_bic: float
    best_beta: NDArray[np.float64]
    on_boundary: bool


def bic_value(
    rss: float,
    k: int,
    n: int,
    p: int,
    *,
    kind: str = "paper",
) -> float:
    """BIC-type criterion (10).

    Parameters
    ----------
    rss :
        ``|| y - X(M_k) @ beta_hat(M_k) ||^2`` on the original scale.
    k, n, p :
        Model size, sample size, number of predictors.
    kind :
        ``"paper"`` is ``log(RSS) + (log(p)/n) * k * log(n)`` (A-4).
        ``"rss_mean"`` replaces ``RSS`` with ``RSS / n``.

    Returns
    -------
    bic
        Scalar criterion value. Natural logarithm.
    """
    kind = check_literal("bic", kind, {"paper", "rss_mean"})
    if rss < _RSS_FLOOR:
        warnings.warn(
            f"RSS={rss} was clamped at {_RSS_FLOOR} before taking the log.",
            TppisWarning,
            stacklevel=2,
        )
        rss = _RSS_FLOOR
    scale = rss if kind == "paper" else rss / n
    return float(np.log(scale) + (np.log(p) / n) * k * np.log(n))


def default_k_max(p: int, n: int, n_retained: int, override: int | None) -> int:
    """A-9 cap ``k_max = min(p, n_retained, n - 1)``.

    ``k <= p - 1`` is not the constraint that keeps least squares valid.
    After centering, the columns of ``X`` span at most ``n - 1`` dimensions,
    so ``k >= n`` interpolates ``y`` and ``log(RSS)`` is unbounded below.
    An override at or above ``n`` is rejected. An override above the retained
    rank is still clipped to that rank.
    """
    cap = min(p, n_retained, n - 1)
    if cap < 1:
        raise TPPISError(
            f"k_max collapsed to {cap}; n_retained={n_retained}, n={n}, p={p}."
        )
    if override is None:
        return cap
    if override < 1:
        raise TPPISError(f"k_max must be at least 1, got {override}.")
    if int(override) >= n:
        raise TPPISError(
            f"k_max={override} must be at most n-1={n - 1}. "
            "A column-centered design has rank at most n-1, so k >= n makes "
            "log(RSS) unbounded below. k <= p-1 is not that constraint."
        )
    return min(int(override), cap)


def _chol_solve(
    L: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    z = solve_triangular(L, rhs, lower=True, check_finite=False)
    beta = solve_triangular(L.T, z, lower=False, check_finite=False)
    return np.asarray(beta, dtype=np.float64)


def _min_norm(
    gram: NDArray[np.float64], rhs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Minimum-norm least-squares solve of a possibly singular Gram block (A-9)."""
    beta, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    return np.asarray(beta, dtype=np.float64)


def incremental_refit(
    F: NDArray[np.float64],
    rhs: NDArray[np.float64],
    X_sel: NDArray[np.float64],
    y: NDArray[np.float64],
    p: int,
    *,
    kind: str = "paper",
) -> CriterionSweep:
    """Sweep ``k = 1 .. K`` with nested Cholesky updates of equation (8).

    Parameters
    ----------
    F :
        Factor matrix of shape ``(K, s)`` such that
        ``Gram_k = F[:k] @ F[:k].T``. For TPPIS/PPIS this is ``V_S(M)``;
        for FPSIS it is ``V_S(M) * mu_S``; for SIS it is ``X(M).T``.
    rhs :
        ``X_hat(M_K).T @ y_hat`` restricted to the ranked columns, length ``K``.
        Equals ``omega[order[:K]]``.
    X_sel, y :
        Ranked original (standardized) columns and the centered response, used
        only for the residual of equation (10).
    p :
        Original number of predictors, for the BIC penalty.
    kind :
        Criterion flavour, see :func:`bic_value`.

    Returns
    -------
    CriterionSweep
        Full path and the minimizing ``k``.
    """
    K = int(F.shape[0])
    if K < 1:
        raise TPPISError("Cannot sweep an empty candidate set.")
    n = int(y.shape[0])
    L = np.zeros((K, K), dtype=np.float64)
    bic = np.empty(K, dtype=np.float64)
    rss = np.empty(K, dtype=np.float64)
    best_k = 1
    best_bic = np.inf
    best_beta = np.empty(0, dtype=np.float64)
    use_direct = False

    for k in range(1, K + 1):
        gk = F[:k] @ F[k - 1]
        if not use_direct:
            if k == 1:
                pivot = float(gk[0])
                if pivot <= _CHOL_EPS * max(1.0, float(np.abs(F[0] @ F[0]))):
                    use_direct = True
                else:
                    L[0, 0] = np.sqrt(pivot)
            else:
                try:
                    ell = solve_triangular(
                        L[: k - 1, : k - 1], gk[: k - 1], lower=True, check_finite=False
                    )
                    rem = float(gk[k - 1] - ell @ ell)
                    if rem <= _CHOL_EPS * max(1.0, float(np.abs(gk[k - 1]))):
                        use_direct = True
                    else:
                        L[k - 1, : k - 1] = ell
                        L[k - 1, k - 1] = np.sqrt(rem)
                except np.linalg.LinAlgError:
                    use_direct = True
        gram = F[:k] @ F[:k].T
        if use_direct:
            beta = _min_norm(gram, rhs[:k])
        else:
            beta = _chol_solve(L[:k, :k], rhs[:k])
        resid = y - X_sel[:, :k] @ beta
        rss_k = float(resid @ resid)
        bic_k = bic_value(rss_k, k, n, p, kind=kind)
        rss[k - 1] = rss_k
        bic[k - 1] = bic_k
        if bic_k < best_bic:
            best_bic = bic_k
            best_k = k
            best_beta = beta

    on_boundary = best_k in {1, K}
    if on_boundary:
        warnings.warn(
            f"BIC minimum landed on the k-boundary at k={best_k} (k_max={K}).",
            BoundarySelectionWarning,
            stacklevel=2,
        )
    return CriterionSweep(
        k=np.arange(1, K + 1, dtype=np.intp),
        bic=bic,
        rss=rss,
        best_k=best_k,
        best_bic=best_bic,
        best_beta=best_beta,
        on_boundary=on_boundary,
    )
