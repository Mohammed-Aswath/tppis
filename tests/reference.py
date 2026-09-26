"""Literal transcription of equations (4), (6), (7), (8) and (10).

These routines form the explicit n-by-n projection matrices printed in the
paper. They are the yardstick for the fast path, not a public API.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def svd_parts(
    X: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    U, mu, Vt = np.linalg.svd(X, full_matrices=False)
    return U, mu, Vt.T


def q_f(U1: NDArray[np.float64]) -> NDArray[np.float64]:
    """Equation (4): ``Q_F = I - U1 (U1.T U1)^{-1} U1.T``."""
    n = U1.shape[0]
    if U1.shape[1] == 0:
        return np.eye(n)
    gram = U1.T @ U1
    return np.eye(n) - U1 @ np.linalg.inv(gram) @ U1.T


def q_p(
    U1: NDArray[np.float64], U2: NDArray[np.float64], mu2: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Equation (6) as printed, without an inverse on ``U1.T U1`` (A-3)."""
    n = U1.shape[0]
    if U2.shape[1] == 0:
        return np.zeros((n, n))
    tail = U2 @ np.diag(1.0 / mu2) @ U2.T
    if U1.shape[1] == 0:
        return tail
    return tail @ (np.eye(n) - U1 @ (U1.T @ U1) @ U1.T)


def q_t(
    U1: NDArray[np.float64], U2a: NDArray[np.float64], mu2a: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Equation (7) as printed."""
    return q_p(U1, U2a, mu2a)


def omega_from_q(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    Q: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``omega = (Q X).T @ (Q y)``."""
    X_hat = Q @ X
    y_hat = Q @ y
    return X_hat.T @ y_hat


def sis_omega(X: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    return X.T @ y


def fpsis_omega(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    d: int,
) -> NDArray[np.float64]:
    U, _mu, _V = svd_parts(X)
    return omega_from_q(X, y, q_f(U[:, :d]))


def ppis_omega(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    d: int,
) -> NDArray[np.float64]:
    U, mu, _V = svd_parts(X)
    return omega_from_q(X, y, q_p(U[:, :d], U[:, d:], mu[d:]))


def tppis_omega(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    d: int,
    m: int,
) -> NDArray[np.float64]:
    U, mu, _V = svd_parts(X)
    return omega_from_q(X, y, q_t(U[:, :d], U[:, d:m], mu[d:m]))


def refit_beta(
    X_hat_m: NDArray[np.float64],
    y_hat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Equation (8): ``beta = (X_hat_M.T X_hat_M)^{-1} X_hat_M.T y_hat``."""
    gram = X_hat_m.T @ X_hat_m
    return np.linalg.solve(gram, X_hat_m.T @ y_hat)


def bic_paper(rss: float, k: int, n: int, p: int) -> float:
    """Equation (10): natural log, unnormalized RSS (A-4)."""
    return float(np.log(rss) + (np.log(p) / n) * k * np.log(n))
