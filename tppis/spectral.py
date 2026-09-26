"""Thin SVD / Gram decompositions and the three-block spectral split.

Equation (2) of Tanaka and Matsui (2023). The right singular vectors ``V`` are
never stored; see specification Section 4.4.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd as scipy_svd

from tppis._validation import check_literal
from tppis.exceptions import RankDeficiencyWarning, TPPISError


@dataclass(frozen=True)
class SpectralFactorization:
    """Cached left singular structure of a design matrix.

    Parameters
    ----------
    U :
        Orthonormal columns, shape ``(n, r_kept)``.
    mu :
        Singular values in nonincreasing order, length ``r_kept``.
    n, p :
        Original shape of ``X``.
    n_dropped :
        Number of singular values removed by the A-6 tolerance.
    backend :
        ``"svd"`` or ``"gram"``.
    """

    U: NDArray[np.float64]
    mu: NDArray[np.float64]
    n: int
    p: int
    n_dropped: int
    backend: str

    @property
    def r(self) -> int:
        """Number of retained singular values after the A-6 drop."""
        return int(self.mu.shape[0])


@dataclass(frozen=True)
class SpectralBlocks:
    """Index ranges of the three spectral blocks in specification Section 2.1.

    Indices are 0-based and refer to columns of :attr:`SpectralFactorization.U`.
    """

    d: int
    m: int
    r: int
    alpha: float

    @property
    def common(self) -> slice:
        """Block 1: columns ``0 .. d-1``."""
        return slice(0, self.d)

    @property
    def retained(self) -> slice:
        """Block 2a: columns ``d .. m-1``."""
        return slice(self.d, self.m)

    @property
    def truncated(self) -> slice:
        """Block 2b: columns ``m .. r-1``."""
        return slice(self.m, self.r)

    @property
    def tail(self) -> slice:
        """Blocks 2a and 2b together: columns ``d .. r-1`` (PPIS / FPSIS)."""
        return slice(self.d, self.r)

    @property
    def n_retained(self) -> int:
        """Width of the retained block 2a."""
        return self.m - self.d


def default_tol(n: int, p: int) -> float:
    """Default A-6 relative tolerance, matching ``numpy.linalg.pinv``."""
    return float(max(n, p) * np.finfo(np.float64).eps)


def decompose(
    X: NDArray[np.float64],
    *,
    backend: str = "svd",
    tol: float | None = None,
) -> SpectralFactorization:
    """Thin factorization ``X = U @ diag(mu) @ V.T`` without forming ``V``.

    Parameters
    ----------
    X :
        Column-standardized design of shape ``(n, p)``.
    backend :
        ``"svd"`` uses :func:`scipy.linalg.svd`. ``"gram"`` eigendecomposes
        ``X @ X.T`` and is the large-``p`` escape hatch (specification §10).
    tol :
        Relative cutoff ``mu_l <= tol * mu_1``. ``None`` uses :func:`default_tol`.

    Returns
    -------
    SpectralFactorization
        Cached ``U`` and ``mu`` reused across a parameter grid.
    """
    backend = check_literal("backend", backend, {"svd", "gram"})
    n, p = X.shape
    if backend == "svd":
        U, mu, _vt = scipy_svd(
            X, full_matrices=False, overwrite_a=False, check_finite=False
        )
        U = np.ascontiguousarray(U, dtype=np.float64)
        mu = np.ascontiguousarray(mu, dtype=np.float64)
    else:
        gram = X @ X.T
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        U = np.ascontiguousarray(evecs[:, order], dtype=np.float64)
        mu = np.ascontiguousarray(np.sqrt(evals), dtype=np.float64)

    if mu.size == 0:
        raise TPPISError("Decomposition produced no singular values.")
    cutoff = (default_tol(n, p) if tol is None else float(tol)) * mu[0]
    keep = mu > cutoff
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(
            f"Dropped {n_dropped} singular value(s) at or below {cutoff:.3e} (A-6).",
            RankDeficiencyWarning,
            stacklevel=2,
        )
    if not np.any(keep):
        raise TPPISError("All singular values were dropped by the A-6 tolerance.")
    return SpectralFactorization(
        U=U[:, keep],
        mu=mu[keep],
        n=n,
        p=p,
        n_dropped=n_dropped,
        backend=backend,
    )


def alpha_to_m(n: int, alpha: float, *, rounding: str = "floor") -> int:
    """Convert ``alpha`` to the paper's ``[n * alpha]`` cut (A-1)."""
    rounding = check_literal("alpha_rounding", rounding, {"floor", "round"})
    if not (0.0 < alpha <= 1.0):
        raise TPPISError(f"alpha must lie in (0, 1], got {alpha}.")
    raw = n * alpha
    return int(np.floor(raw) if rounding == "floor" else np.round(raw))


def split_blocks(
    spec: SpectralFactorization,
    d: int,
    alpha: float,
    *,
    rounding: str = "floor",
) -> SpectralBlocks:
    """Build the three-block split of specification Section 2.1.

    Enforces ``0 <= d < m <= r`` (A-2, A-8). ``m`` is also capped at ``r``.
    """
    if d < 0:
        raise TPPISError(f"d must be nonnegative, got {d}.")
    m = min(alpha_to_m(spec.n, alpha, rounding=rounding), spec.r)
    if not (0 <= d < m <= spec.r):
        raise TPPISError(
            f"Invalid split: need 0 <= d < m <= r, got d={d}, m={m}, r={spec.r}."
        )
    return SpectralBlocks(d=int(d), m=int(m), r=spec.r, alpha=float(alpha))
