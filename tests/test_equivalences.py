"""Exact identities: SIS = X.T @ y, TPPIS(alpha=1) = PPIS."""

from __future__ import annotations

import numpy as np
import pytest
from tppis import PPIS, SIS, TPPIS
from tppis._validation import standardize
from tppis.kernel import screening_scores
from tppis.spectral import decompose, split_blocks


def _xy(n: int = 30, p: int = 18, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X[:, 0] + 0.7 * X[:, 1] + 0.1 * rng.standard_normal(n)
    return X, y


def test_sis_equals_xty() -> None:
    X, y = _xy()
    Xs, ys, *_ = standardize(X, y)
    est = SIS(k=2, standardize=False).fit(Xs, ys)
    np.testing.assert_allclose(est.omega_, Xs.T @ ys, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("d", [1, 2, 4, 6])
@pytest.mark.parametrize("seed", [0, 3, 7])
def test_tppis_alpha_1_equals_ppis(d: int, seed: int) -> None:
    X, y = _xy(seed=seed)
    a = PPIS(d=d, k=4).fit(X, y)
    b = TPPIS(d=d, alpha=1.0, k=4).fit(X, y)
    np.testing.assert_allclose(a.omega_, b.omega_, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(a.ranking_, b.ranking_)


def test_tppis_alpha_1_kernel_equals_ppis_kernel() -> None:
    X, y = _xy()
    Xs, ys, *_ = standardize(X, y)
    spec = decompose(Xs)
    d = 3
    blocks = split_blocks(spec, d, 1.0)
    assert blocks.m == spec.r
    np.testing.assert_allclose(
        screening_scores(Xs, ys, spec, blocks.retained, -1),
        screening_scores(Xs, ys, spec, blocks.tail, -1),
        rtol=0,
        atol=0,
    )
