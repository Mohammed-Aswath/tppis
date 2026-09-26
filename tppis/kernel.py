"""Shared screening kernel ``omega(S, g)`` of specification Section 4.3.

Equation-level identities
-------------------------
The four methods of Tanaka and Matsui (2023) are the same product

    omega(S, g) = V_S @ (D_S ** g) @ U_S.T @ y

implemented without forming ``V`` (specification Section 4.4):

    w     = U_S @ (mu_S ** (g - 1)) @ U_S.T @ y
    omega = X.T @ w
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tppis._validation import TPPISError
from tppis.spectral import SpectralFactorization


def spectral_slice(
    spec: SpectralFactorization,
    index: slice,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(U_S, mu_S)`` for a column slice of the cached factorization."""
    U_s = spec.U[:, index]
    mu_s = spec.mu[index]
    if U_s.shape[1] == 0:
        raise TPPISError("Spectral index set S is empty.")
    return U_s, mu_s


def preconditioner_weights(
    spec: SpectralFactorization,
    index: slice,
    exponent: int,
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """The n-vector ``w`` of specification Section 4.4.

    Parameters
    ----------
    spec :
        Cached factorization from one ``fit``.
    index :
        Spectral index set ``S``.
    exponent :
        ``g`` in ``omega(S, g)``. ``+1`` for SIS/FPSIS, ``-1`` for PPIS/TPPIS.
    y :
        Centered response.

    Returns
    -------
    w
        Vector of length ``n`` such that ``omega = X.T @ w``.
    """
    U_s, mu_s = spectral_slice(spec, index)
    # Divide, never invert a diagonal matrix. The tail inversion is the method.
    scale = mu_s ** (exponent - 1)
    return U_s @ (scale * (U_s.T @ y))


def omega_from_weights(
    X: NDArray[np.float64],
    w: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``omega = X.T @ w``."""
    return X.T @ w


def column_energy(
    X: NDArray[np.float64],
    spec: SpectralFactorization,
    index: slice,
    exponent: int,
) -> NDArray[np.float64]:
    """Squared Euclidean norm of each column of the transformed design.

    Zhao et al. (2020), equation (15), rank by the marginal coefficient
    ``(xhat_j · yhat) / ||xhat_j||^2``. That denominator is the same for every
    column only before a factor transform. After FPSIS or PPIS it is not, and
    dividing by it is what makes the profiled ``x4`` visible.

    ``exponent == +1`` is FPSIS (``Xhat = U_S D_S V_S^T``).
    ``exponent == -1`` is PPIS/TPPIS (``Xhat = U_S V_S^T``).
    """
    U_s, mu_s = spectral_slice(spec, index)
    proj = U_s.T @ X
    if exponent == 1:
        return np.sum(proj * proj, axis=0)
    return np.sum((proj / mu_s[:, None]) ** 2, axis=0)


def screening_scores(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    spec: SpectralFactorization,
    index: slice,
    exponent: int,
) -> NDArray[np.float64]:
    """Full kernel evaluation ``omega(S, g)``.

    Parameters
    ----------
    X, y :
        Standardized design and centered response.
    spec :
        Cached factorization.
    index, exponent :
        Method-specific ``(S, g)``.

    Returns
    -------
    omega
        Marginal importance of each of the ``p`` columns.
    """
    return omega_from_weights(X, preconditioner_weights(spec, index, exponent, y))


def transformed_response(
    spec: SpectralFactorization,
    index: slice,
    exponent: int,
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``y_hat = U_S @ (mu_S ** exponent) @ U_S.T @ y`` when ``U`` is used as Q.

    For TPPIS/PPIS (``g = -1``) this is ``Q_T y`` / ``Q_P y``.
    For FPSIS (``g = +1``) the paper uses ``Q_F y = y - U_1 U_1.T y``, which is
    *not* this expression unless ``U`` spans ``R^n``. Callers that need ``Q_F y``
    should form it from the common-factor block instead.
    """
    U_s, mu_s = spectral_slice(spec, index)
    return U_s @ ((mu_s**exponent) * (U_s.T @ y))


def right_vectors(
    X: NDArray[np.float64],
    spec: SpectralFactorization,
    index: slice,
    columns: NDArray[np.intp] | None = None,
) -> NDArray[np.float64]:
    """``V_S`` (or the rows of ``V_S`` indexed by ``columns``) via ``X.T @ U_S / mu``.

    Parameters
    ----------
    X :
        Standardized design.
    spec, index :
        Cached factorization and spectral block.
    columns :
        If given, only those feature rows of ``V_S`` are formed (``K x |S|``).
    """
    U_s, mu_s = spectral_slice(spec, index)
    X_use = X if columns is None else X[:, columns]
    return (X_use.T @ U_s) / mu_s
