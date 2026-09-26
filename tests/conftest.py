"""Shared fixtures.

Every random draw here is seeded explicitly, so that a failure is reproducible from the
test name alone.
"""

from __future__ import annotations

import numpy as np
import pytest


def multicollinear_design(n_samples, n_features, n_factors=3, seed=0):
    """A design with a few strong common factors above a noise floor."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_samples, n_factors))
    B = rng.standard_normal((n_factors, n_features))
    return Z @ B + rng.standard_normal((n_samples, n_features))


@pytest.fixture
def small_problem():
    """A wide, strongly multicollinear problem with a known active set."""
    rng = np.random.default_rng(11)
    X = multicollinear_design(40, 120, seed=11)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 5.0 * X[:, 0] - 4.0 * X[:, 1] + 3.0 * X[:, 2] + 0.1 * rng.standard_normal(40)
    return X, y - y.mean()


@pytest.fixture(params=[(20, 60), (30, 30), (40, 15)], ids=["wide", "square", "tall"])
def shaped_problem(request):
    """The same problem across the three shape regimes n < p, n == p and n > p."""
    n_samples, n_features = request.param
    rng = np.random.default_rng(hash(request.param) % 2**32)
    X = multicollinear_design(n_samples, n_features, n_factors=2, seed=3)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = 4.0 * X[:, 0] - 3.0 * X[:, 1] + 0.2 * rng.standard_normal(n_samples)
    return X, y - y.mean()
