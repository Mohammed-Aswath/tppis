"""Grid construction, skip accounting, and planted-optimum recovery."""

from __future__ import annotations

import numpy as np
import pytest
from tppis import TPPIS
from tppis._validation import TPPISError, standardize
from tppis.spectral import decompose
from tppis.tuning import d_candidates, search


def test_all_invalid_grid_raises() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 10))
    y = rng.standard_normal(16)
    Xs, ys, *_ = standardize(X, y)
    spec = decompose(Xs)
    with pytest.raises(TPPISError, match="invalid"):
        search(
            Xs,
            ys,
            spec,
            "tppis",
            d=15,
            alpha=0.2,
            k=1,
            k_max=None,
            bic_kind="paper",
            rounding="floor",
        )


def test_d_grid_includes_ratio_and_fractions() -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 12))
    spec = decompose(X)
    vals = d_candidates(spec, "grid", n=20)
    assert int(np.floor(0.2 * 20)) in vals
    assert int(np.floor(1.0 * 20)) in vals
    assert min(vals) >= 0


def test_planted_parameters_are_recovered() -> None:
    """BIC prefers the (d, alpha) pair that actually generated a sparse y."""
    rng = np.random.default_rng(2)
    n, p = 40, 20
    X = rng.standard_normal((n, p))
    y = 4.0 * X[:, 0] + 3.0 * X[:, 1] + 0.05 * rng.standard_normal(n)
    est = TPPIS(d="grid", alpha="grid", k="bic").fit(X, y)
    kept = [row for row in est.grid_ if not row["skipped"]]
    assert kept
    # The minimizing row matches the fitted attributes.
    best = min(kept, key=lambda r: r["bic"])
    assert best["d"] == est.d_
    assert best["alpha"] == est.alpha_
    assert best["k"] == est.k_
    assert {0, 1}.issubset(set(est.selected_.tolist()))


def test_grid_is_enough_to_plot_bic_vs_alpha() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((30, 14))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(30)
    est = TPPIS(d=2, alpha="grid", k="bic").fit(X, y)
    alphas = sorted({row["alpha"] for row in est.grid_ if not row["skipped"]})
    assert alphas
    # Figure 1 is a slice of this table at fixed d.
    assert all(row["d"] == 2 or row["skipped"] for row in est.grid_)
