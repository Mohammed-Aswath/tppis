"""Simulation generators, PSD gate, and seed reproducibility."""

from __future__ import annotations

import numpy as np
import pytest
from tppis.datasets import (
    example1_sigma,
    example2_sigma,
    make_example1,
    make_example3,
    make_example4,
    min_eigenvalue,
)


@pytest.mark.parametrize("phi", [0.5, 0.7, 0.9])
def test_example1_sigma_is_psd(phi: float) -> None:
    lam = min_eigenvalue(example1_sigma(1000, phi))
    assert lam > -1e-10, f"Example 1 Sigma is indefinite at phi={phi}: λ_min={lam}"


@pytest.mark.parametrize("phi", [0.5, 0.7, 0.9])
def test_example2_sigma_is_psd(phi: float) -> None:
    lam = min_eigenvalue(example2_sigma(200, phi))
    assert lam > -1e-10, f"Example 2 Sigma is indefinite at phi={phi}: λ_min={lam}"


def test_generators_are_reproducible() -> None:
    a = make_example1(n=30, p=20, phi=0.5, seed=7)
    b = make_example1(n=30, p=20, phi=0.5, seed=7)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    c = make_example4(n=30, p=20, m=6, seed=3)
    d = make_example4(n=30, p=20, m=6, seed=3)
    np.testing.assert_array_equal(c.X, d.X)


def test_example1_active_set() -> None:
    data = make_example1(n=80, p=12, phi=0.5, seed=0)
    assert data.active.tolist() == [0, 1, 2, 3]
    assert data.beta[3] == pytest.approx(-15.0 * np.sqrt(0.5))
    cov = data.sigma @ data.beta
    assert cov[3] == pytest.approx(0.0, abs=1e-8)
    assert cov[0] == pytest.approx(5.0 * (1.0 - 0.5))


def test_example3_overwrites_column_six() -> None:
    data = make_example3(n=200, p=15, phi=0.5, seed=1)
    residual = data.X[:, 5] - 0.8 * data.X[:, 4]
    assert residual.std() < 0.2
    assert 5 not in data.active.tolist()
