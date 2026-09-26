"""Input checking, standardization and tie-stable ranking.

Standardization uses the population standard deviation (``ddof=0``), matching
ambiguity A-5 of the specification and ``sklearn.preprocessing.StandardScaler``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tppis.exceptions import ConstantFeatureWarning, TPPISError

__all__ = [
    "TPPISError",
    "apply_standardize",
    "as_float_arrays",
    "check_literal",
    "rank_by_abs",
    "standardize",
]


def as_float_arrays(
    X: ArrayLike,
    y: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Cast ``X`` and ``y`` to contiguous float64 arrays with matching ``n``.

    Parameters
    ----------
    X :
        Predictor matrix of shape ``(n, p)``.
    y :
        Response vector of shape ``(n,)``.

    Returns
    -------
    X, y
        Contiguous ``float64`` copies.
    """
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    y_arr = np.ascontiguousarray(np.asarray(y, dtype=np.float64).reshape(-1))
    if X_arr.ndim != 2:
        raise TPPISError(f"X must be 2-dimensional, got shape {X_arr.shape}.")
    n, p = X_arr.shape
    if n < 2:
        raise TPPISError(f"n must be at least 2, got {n}.")
    if p < 1:
        raise TPPISError("X must have at least one column.")
    if y_arr.shape[0] != n:
        raise TPPISError(
            f"X and y must share the first dimension, "
            f"got n={n} and y.size={y_arr.size}."
        )
    if not np.isfinite(X_arr).all():
        raise TPPISError("X contains non-finite values.")
    if not np.isfinite(y_arr).all():
        raise TPPISError("y contains non-finite values.")
    return X_arr, y_arr


def standardize(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    enabled: bool = True,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Center ``y`` and optionally column-standardize ``X`` with ``ddof=0``.

    Parameters
    ----------
    X, y :
        Validated arrays from :func:`as_float_arrays`.
    enabled :
        If False, ``X`` is returned unchanged and column scales are ones.

    Returns
    -------
    X_std, y_c, x_mean, x_scale, y_mean
    """
    y_mean = float(y.mean())
    y_c = y - y_mean
    x_mean = X.mean(axis=0)
    if not enabled:
        return X, y_c, x_mean, np.ones(X.shape[1], dtype=np.float64), y_mean
    x_scale = X.std(axis=0, ddof=0)
    zero = x_scale <= np.finfo(np.float64).eps * max(X.shape)
    if np.any(zero):
        warnings.warn(
            f"{int(zero.sum())} constant column(s) left unscaled.",
            ConstantFeatureWarning,
            stacklevel=2,
        )
        x_scale = x_scale.copy()
        x_scale[zero] = 1.0
    X_std = (X - x_mean) / x_scale
    return X_std, y_c, x_mean, x_scale, y_mean


def rank_by_abs(
    omega: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Descending ``|omega|`` order, ties broken by ascending column index (A-7).

    Parameters
    ----------
    omega :
        Marginal importance scores of length ``p``.

    Returns
    -------
    order
        Integer indices of shape ``(p,)``.
    """
    p = omega.shape[0]
    keys = np.empty((p, 2), dtype=np.float64)
    keys[:, 0] = -np.abs(omega)
    keys[:, 1] = np.arange(p, dtype=np.float64)
    return np.lexsort((keys[:, 1], keys[:, 0])).astype(np.intp)


def apply_standardize(
    X: ArrayLike,
    x_mean: NDArray[np.float64],
    x_scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply a previously fitted column standardization to new ``X``."""
    X_arr = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
    if X_arr.ndim != 2:
        raise TPPISError(f"X must be 2-dimensional, got shape {X_arr.shape}.")
    if X_arr.shape[1] != x_mean.shape[0]:
        raise TPPISError(f"X has {X_arr.shape[1]} columns, expected {x_mean.shape[0]}.")
    return (X_arr - x_mean) / x_scale


def check_literal(name: str, value: Any, allowed: set[str]) -> str:
    """Validate a string option against a closed set."""
    if value not in allowed:
        raise TPPISError(f"{name} must be one of {sorted(allowed)}, got {value!r}.")
    return str(value)
