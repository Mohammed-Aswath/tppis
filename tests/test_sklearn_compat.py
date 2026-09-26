"""The sklearn 1.3 / 1.6 input-check adapter."""

from __future__ import annotations

import numpy as np
from tppis import TPPIS
from tppis._sklearn_compat import check_X, check_xy, sklearn_at_least


def test_check_xy_sets_n_features() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 6))
    y = X[:, 0] + rng.standard_normal(20)
    est = TPPIS(d=1, alpha=1.0, k=1)
    X_out, y_out = check_xy(est, X, y, reset=True)
    assert X_out.shape == (20, 6)
    assert y_out.shape == (20,)
    assert est.n_features_in_ == 6


def test_check_X_after_fit() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 6))
    y = X[:, 0]
    est = TPPIS(d=1, alpha=1.0, k=1).fit(X, y)
    X_out = check_X(est, X, reset=False)
    assert X_out.shape == X.shape


def test_sklearn_at_least_is_boolean() -> None:
    assert sklearn_at_least(1, 0) is True
    assert sklearn_at_least(99, 0) is False
