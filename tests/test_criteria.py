"""Refit (8), BIC (10), and the analytic check of specification Section 5.1."""

from __future__ import annotations

import numpy as np
from tppis.criteria import bic_value, incremental_refit
from tppis.datasets import make_example1

from tests.reference import bic_paper, refit_beta


def test_bic_matches_reference_transcription() -> None:
    got = bic_value(100.0, 4, 100, 1000, kind="paper")
    assert got == bic_paper(100.0, 4, 100, 1000)


def test_section_51_analytic_check() -> None:
    # RSS ~ n * sigma^2 = 100 at the true model; penalty ~ 1.272; BIC ~ 5.877.
    value = bic_value(100.0, 4, 100, 1000, kind="paper")
    assert 5.85 < value < 5.90


def test_incremental_matches_direct_solve() -> None:
    rng = np.random.default_rng(0)
    n, K, s = 30, 6, 8
    F = rng.standard_normal((K, s))
    X_sel = rng.standard_normal((n, K))
    y = rng.standard_normal(n)
    rhs = rng.standard_normal(K)
    sweep = incremental_refit(F, rhs, X_sel, y, p=20, kind="paper")
    for k in range(1, K + 1):
        gram = F[:k] @ F[:k].T
        beta_k = np.linalg.solve(gram, rhs[:k])
        resid = y - X_sel[:, :k] @ beta_k
        rss = float(resid @ resid)
        np.testing.assert_allclose(sweep.rss[k - 1], rss, rtol=1e-10, atol=1e-10)
        if k == sweep.best_k:
            np.testing.assert_allclose(sweep.best_beta, beta_k, rtol=1e-10, atol=1e-10)


def test_direct_refit_matches_equation_8() -> None:
    rng = np.random.default_rng(1)
    n, k = 25, 4
    X_hat = rng.standard_normal((n, k))
    y_hat = rng.standard_normal(n)
    beta = refit_beta(X_hat, y_hat)
    gram = X_hat.T @ X_hat
    np.testing.assert_allclose(gram @ beta, X_hat.T @ y_hat, rtol=1e-10, atol=1e-10)


def test_example1_true_model_bic_is_near_588() -> None:
    data = make_example1(n=100, p=80, phi=0.5, seed=0)
    # Residual of the true four-variable model on this one draw.
    X4 = data.X[:, :4]
    beta, *_ = np.linalg.lstsq(X4, data.y, rcond=None)
    rss = float(np.sum((data.y - X4 @ beta) ** 2))
    value = bic_value(rss, 4, 100, 80, kind="paper")
    # One draw, not the expectation: just check the criterion is O(1) not O(10).
    assert 3.0 < value < 9.0
