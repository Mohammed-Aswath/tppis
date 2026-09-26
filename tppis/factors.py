"""Factor-count selection by the singular-value ratio, equation (3).

d = argmax_{1 <= l <= r-1}  mu_l^2 / mu_{l+1}^2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tppis._validation import TPPISError
from tppis.spectral import SpectralFactorization


def ratio_d(mu: NDArray[np.float64]) -> int:
    """Return ``d`` from equation (3) of Tanaka and Matsui (2023).

    Parameters
    ----------
    mu :
        Positive singular values in nonincreasing order.

    Returns
    -------
    d
        1-based count of leading common factors, in ``1 .. r-1``.
    """
    if mu.size < 2:
        raise TPPISError("Equation (3) needs at least two singular values.")
    # mu is already sorted nonincreasing and A-6-positive.
    ratios = (mu[:-1] ** 2) / (mu[1:] ** 2)
    return int(np.argmax(ratios) + 1)


def ratio_d_from_spec(spec: SpectralFactorization) -> int:
    """Equation (3) applied to a cached factorization."""
    return ratio_d(spec.mu)
