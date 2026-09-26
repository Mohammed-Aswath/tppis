"""Fast kernel against the literal transcription and algebraic identities."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tppis.kernel import screening_scores
from tppis.spectral import decompose, split_blocks

from tests.reference import omega_from_q, q_f, q_p, q_t, sis_omega

RTOL = 1e-10
ATOL = 1e-10


def _xy(n: int, p: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # Uncentered full-rank draws so A-6 does not drop the centering null
    # vector; the identities are then exact against the paper's matrices.
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize(
    "n,p,d,alpha,seed",
    [
        (12, 20, 2, 0.6, 0),
        (15, 10, 1, 0.8, 1),  # n > p
        (20, 20, 3, 1.0, 2),
        (18, 25, 0, 0.5, 3),
    ],
)
def test_fast_kernel_matches_reference(
    n: int, p: int, d: int, alpha: float, seed: int
) -> None:
    X, y = _xy(n, p, seed)
    spec = decompose(X, tol=0.0)
    blocks = split_blocks(spec, d, alpha)
    U, mu = spec.U, spec.mu
    m = blocks.m

    if spec.r == spec.n:
        np.testing.assert_allclose(
            screening_scores(X, y, spec, slice(0, spec.r), 1),
            sis_omega(X, y),
            rtol=RTOL,
            atol=ATOL,
        )
    np.testing.assert_allclose(
        screening_scores(X, y, spec, blocks.tail, 1),
        omega_from_q(X, y, q_f(U[:, :d])),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        screening_scores(X, y, spec, blocks.tail, -1),
        omega_from_q(X, y, q_p(U[:, :d], U[:, d:], mu[d:])),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        screening_scores(X, y, spec, blocks.retained, -1),
        omega_from_q(X, y, q_t(U[:, :d], U[:, d:m], mu[d:m])),
        rtol=RTOL,
        atol=ATOL,
    )


def test_transformed_design_has_unit_singular_values() -> None:
    X, y = _xy(16, 22, seed=4)
    spec = decompose(X)
    blocks = split_blocks(spec, d=3, alpha=0.7)
    U_s = spec.U[:, blocks.retained]
    from tppis.kernel import right_vectors

    V_s = right_vectors(X, spec, blocks.retained)
    X_hat = U_s @ V_s.T
    svals = np.linalg.svd(X_hat, compute_uv=False)
    nonzero = svals[svals > 1e-12]
    np.testing.assert_allclose(nonzero, np.ones_like(nonzero), rtol=1e-10, atol=1e-10)


def test_common_and_retained_blocks_are_orthogonal() -> None:
    spec = decompose(_xy(14, 18, seed=5)[0])
    blocks = split_blocks(spec, d=4, alpha=0.8)
    U1 = spec.U[:, blocks.common]
    U2a = spec.U[:, blocks.retained]
    gram = U2a.T @ U1
    np.testing.assert_allclose(gram, np.zeros_like(gram), atol=1e-12)


@given(
    n=st.integers(8, 16),
    p=st.integers(10, 24),
    seed=st.integers(0, 50),
)
@settings(max_examples=15, deadline=None)
def test_permuting_columns_permutes_omega(n: int, p: int, seed: int) -> None:
    X, y = _xy(n, p, seed)
    spec = decompose(X)
    d = 2
    blocks = split_blocks(spec, d, 0.7)
    omega = screening_scores(X, y, spec, blocks.retained, -1)
    perm = np.random.default_rng(seed + 99).permutation(p)
    spec_p = decompose(X[:, perm])
    # d/m stay the same because the spectrum of X is permutation-invariant.
    omega_p = screening_scores(X[:, perm], y, spec_p, blocks.retained, -1)
    np.testing.assert_allclose(omega_p, omega[perm], rtol=1e-8, atol=1e-8)


def test_rescaling_y_scales_omega() -> None:
    X, y = _xy(12, 16, seed=6)
    spec = decompose(X)
    blocks = split_blocks(spec, 2, 0.6)
    omega = screening_scores(X, y, spec, blocks.retained, -1)
    omega2 = screening_scores(X, 3.0 * y, spec, blocks.retained, -1)
    np.testing.assert_allclose(omega2, 3.0 * omega, rtol=1e-12, atol=1e-12)
