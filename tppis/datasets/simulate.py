"""Simulation designs from Section 4 of Tanaka and Matsui (2023)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SimulatedData:
    """One draw of ``(X, y)`` together with the known active set."""

    X: NDArray[np.float64]
    y: NDArray[np.float64]
    beta: NDArray[np.float64]
    active: NDArray[np.intp]
    sigma: NDArray[np.float64] | None = None


def _as_generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def example1_sigma(p: int, phi: float) -> NDArray[np.float64]:
    """Covariance of Example 1.

    ``Sigma_jj = 1``, ``Sigma_jk = phi`` off-diagonal except row/column 4
    (1-based), which equals ``sqrt(phi)``.
    """
    sigma = np.full((p, p), phi, dtype=np.float64)
    np.fill_diagonal(sigma, 1.0)
    i4 = 3
    others = np.ones(p, dtype=bool)
    others[i4] = False
    sigma[i4, others] = np.sqrt(phi)
    sigma[others, i4] = np.sqrt(phi)
    return sigma


def example2_sigma(p: int, phi: float) -> NDArray[np.float64]:
    """Example 1 covariance with variable 5 uncorrelated with the rest."""
    sigma = example1_sigma(p, phi)
    i5 = 4
    sigma[i5, :] = 0.0
    sigma[:, i5] = 0.0
    sigma[i5, i5] = 1.0
    return sigma


def min_eigenvalue(sigma: NDArray[np.float64]) -> float:
    """Smallest eigenvalue, used as the Section 6.5 PSD gate."""
    return float(np.linalg.eigvalsh(sigma)[0])


def _fan_lv_design(
    rng: np.random.Generator,
    n: int,
    p: int,
    phi: float,
) -> NDArray[np.float64]:
    """Equicorrelation ``phi`` with column 4 (1-based) loading fully on the factor.

    This is the covariance printed in Example 1, drawn from its factor model
    rather than a ``p x p`` factorization. Column 4 equals the latent factor,
    so its correlation with every other column is ``sqrt(phi)``.
    """
    factor = rng.normal(size=n)
    X = np.sqrt(phi) * factor[:, None] + np.sqrt(1.0 - phi) * rng.normal(size=(n, p))
    X[:, 3] = factor
    return X


def make_example1(
    n: int = 100,
    p: int = 1000,
    phi: float = 0.5,
    *,
    seed: int | np.random.Generator | None = None,
) -> SimulatedData:
    """Example 1, with the coefficient that makes ``Cov(y, x_4) = 0``.

    The printed regression writes ``-15 x_4``. Fan and Lv (2008), Example II,
    and Zhao et al. (2020), Example 1, which this design follows, use
    ``-15 sqrt(phi) x_4``. With the printed covariance that is the unique
    coefficient that makes column 4 uncorrelated with ``y``, which is the
    mechanism the paper describes. The typeset ``-15`` leaves a correlation
    of ``15 sqrt(phi) - 15``.

    Parameters
    ----------
    n, p, phi :
        Sample size, dimension, and equicorrelation. The paper uses
        ``n in {100, 300}``, ``p = 1000``, ``phi in {0.5, 0.7, 0.9}``.
    seed :
        ``int`` or :class:`numpy.random.Generator`.
    """
    rng = _as_generator(seed)
    beta = np.zeros(p, dtype=np.float64)
    beta[:4] = (5.0, 5.0, 5.0, -15.0 * np.sqrt(phi))
    sigma = example1_sigma(p, phi)
    X = _fan_lv_design(rng, n, p, phi)
    y = X @ beta + rng.normal(size=n)
    return SimulatedData(X, y, beta, np.array([0, 1, 2, 3], dtype=np.intp), sigma)


def make_example2(
    n: int = 100,
    p: int = 1000,
    phi: float = 0.5,
    *,
    seed: int | np.random.Generator | None = None,
) -> SimulatedData:
    """Example 2: Example 1 plus an uncorrelated fifth active variable."""
    rng = _as_generator(seed)
    beta = np.zeros(p, dtype=np.float64)
    beta[:5] = (5.0, 5.0, 5.0, -15.0 * np.sqrt(phi), 5.0)
    sigma = example2_sigma(p, phi)
    X = _fan_lv_design(rng, n, p, phi)
    X[:, 4] = rng.normal(size=n)
    y = X @ beta + rng.normal(size=n)
    return SimulatedData(X, y, beta, np.array([0, 1, 2, 3, 4], dtype=np.intp), sigma)


def make_example3(
    n: int = 100,
    p: int = 1000,
    phi: float = 0.5,
    *,
    seed: int | np.random.Generator | None = None,
) -> SimulatedData:
    """Example 3: Example 2 plus ``x_6 = 0.8 x_5 + delta``.

    Variable 6 is inactive. ``delta ~ N(0, 0.01)``.
    """
    data = make_example2(n=n, p=p, phi=phi, seed=seed)
    # A caller-supplied Generator was already advanced by make_example2.
    if isinstance(seed, np.random.Generator):
        extra = seed
    elif seed is None:
        extra = np.random.default_rng()
    else:
        extra = np.random.default_rng(seed + 1)
    X = data.X.copy()
    X[:, 5] = 0.8 * X[:, 4] + extra.normal(scale=0.1, size=n)
    return SimulatedData(X, data.y, data.beta, data.active, data.sigma)


def make_example4(
    n: int = 100,
    p: int = 1000,
    *,
    d: int = 3,
    m: int | None = None,
    seed: int | np.random.Generator | None = None,
) -> SimulatedData:
    """Example 4: the spike model of specification Section 6.4.

    Parameters
    ----------
    n, p, d :
        Sample size, dimension, and number of large factors (paper: ``d = 3``).
    m :
        Number of medium factors. The paper uses ``m in {0.2n, 0.4n, 0.6n, 0.8n}``.
        Defaults to ``int(0.2 * n)``.
    seed :
        ``int`` or :class:`numpy.random.Generator`.
    """
    rng = _as_generator(seed)
    if m is None:
        m = int(0.2 * n)
    z = rng.standard_normal((n, d + m))
    B = rng.standard_normal((p, d + m))
    X = z[:, :d] @ B[:, :d].T
    for s in range(m):
        scale = n ** (-(s + 9) / (m + 10))
        X = X + scale * np.outer(z[:, d + s], B[:, d + s])
    X = X + rng.standard_normal((n, p))
    beta = np.zeros(p, dtype=np.float64)
    beta[:4] = (5.0, 4.0, 3.0, 2.0)
    signal = X @ beta
    noise_var = float(np.var(signal, ddof=0) / 5.0)
    y = signal + rng.normal(scale=np.sqrt(max(noise_var, 0.0)), size=n)
    return SimulatedData(X, y, beta, np.array([0, 1, 2, 3], dtype=np.intp), None)
