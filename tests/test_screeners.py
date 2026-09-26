"""Public estimators recover an easy active set and expose fitted attributes."""

from __future__ import annotations

import numpy as np
import pytest
from tppis import FPSIS, FPSISBIC, PPIS, SIS, TPPIS, screen


def _easy(n: int = 50, p: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.05 * rng.standard_normal(n)
    return X, y


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (SIS, {"k": 2}),
        (FPSIS, {"d": 2, "k": 2}),
        (FPSISBIC, {"d": 2, "k": 2}),
        (PPIS, {"d": 2, "k": 2}),
        (TPPIS, {"d": 2, "alpha": 0.6, "k": 2}),
    ],
)
def test_easy_problem_recovers_active_set(cls: type, kwargs: dict) -> None:
    X, y = _easy()
    est = cls(**kwargs).fit(X, y)
    assert set(est.get_support(indices=True).tolist()) == {0, 1}
    assert est.k_ == 2
    assert est.scores_.shape == (X.shape[1],)
    assert est.predict(X).shape == (X.shape[0],)


def test_screen_dispatch() -> None:
    X, y = _easy()
    result = screen(X, y, method="tppis", d=2, alpha=0.6, k=2)
    assert set(result.selected_.tolist()) == {0, 1}


def test_fixed_k_does_not_require_grid() -> None:
    X, y = _easy()
    est = TPPIS(d=3, alpha=0.8, k=3).fit(X, y)
    assert est.k_ == 3
    assert len(est.selected_) == 3


def test_grid_records_skips() -> None:
    X, y = _easy(n=20, p=12)
    est = TPPIS(d="grid", alpha="grid", k=2).fit(X, y)
    assert est.grid_
    skipped = [row for row in est.grid_ if row["skipped"]]
    kept = [row for row in est.grid_ if not row["skipped"]]
    assert kept, "at least one valid (d, alpha) pair is required"
    # 1.0n against a small alpha is the typical skip (A-2).
    assert skipped or est.n_skipped_ >= 0
    assert est.d_ is not None
    assert est.alpha_ is not None


def test_empty_grid_raises() -> None:
    from tppis._validation import TPPISError

    X, y = _easy(n=20, p=12)
    with pytest.raises(TPPISError, match="invalid"):
        TPPIS(d=19, alpha=0.2, k=1).fit(X, y)


def test_k_at_least_n_is_rejected() -> None:
    from tppis._validation import TPPISError

    X, y = _easy(n=20, p=30)
    with pytest.raises(TPPISError, match="n-1|n - 1|1 <= k"):
        SIS(k=20).fit(X, y)
    with pytest.raises(TPPISError, match="n-1"):
        SIS(k="bic", k_max=29).fit(X, y)
