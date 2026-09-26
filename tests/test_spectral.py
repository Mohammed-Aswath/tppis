"""Spectral factorization, backends, and block boundaries."""

from __future__ import annotations

import numpy as np
import pytest
from tppis._validation import TPPISError
from tppis.spectral import alpha_to_m, decompose, split_blocks


def _full_rank(n: int = 12, p: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, p))


def test_svd_and_gram_agree_on_well_conditioned_data() -> None:
    X = _full_rank()
    a = decompose(X, backend="svd")
    b = decompose(X, backend="gram")
    # Singular values must match; U is unique up to column sign.
    np.testing.assert_allclose(a.mu, b.mu, rtol=1e-8, atol=1e-8)
    dots = np.abs(np.sum(a.U * b.U, axis=0))
    np.testing.assert_allclose(dots, np.ones_like(dots), rtol=1e-6, atol=1e-6)


def test_rank_deficient_columns_are_dropped() -> None:
    X = _full_rank(n=10, p=8)
    X[:, 7] = X[:, 0]
    spec = decompose(X, backend="svd")
    assert spec.n_dropped >= 1
    assert spec.r < min(X.shape)


def test_alpha_rounding_floor_and_round() -> None:
    assert alpha_to_m(10, 0.25, rounding="floor") == 2
    assert alpha_to_m(10, 0.25, rounding="round") == 2
    assert alpha_to_m(10, 0.35, rounding="floor") == 3
    assert alpha_to_m(10, 0.35, rounding="round") == 4


def test_block_boundaries() -> None:
    spec = decompose(_full_rank(n=10, p=15))
    blocks = split_blocks(spec, d=2, alpha=0.6, rounding="floor")
    assert blocks.d == 2
    assert blocks.m == 6
    assert blocks.common == slice(0, 2)
    assert blocks.retained == slice(2, 6)
    assert blocks.truncated == slice(6, spec.r)
    assert blocks.n_retained == 4


@pytest.mark.parametrize("d,alpha", [(0, 0.4), (3, 0.5), (5, 1.0)])
def test_valid_splits_accepted(d: int, alpha: float) -> None:
    spec = decompose(_full_rank(n=12, p=20))
    blocks = split_blocks(spec, d=d, alpha=alpha)
    assert 0 <= blocks.d < blocks.m <= blocks.r


def test_invalid_split_rejected() -> None:
    spec = decompose(_full_rank(n=10, p=12))
    with pytest.raises(TPPISError, match="Invalid split"):
        split_blocks(spec, d=6, alpha=0.4)  # m = 4, d = 6
    with pytest.raises(TPPISError, match="Invalid split"):
        split_blocks(spec, d=10, alpha=1.0)  # d = m = r
