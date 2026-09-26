"""scikit-learn compatibility: clone, Pipeline, GridSearchCV, estimator checks."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks
from tppis import SIS, TPPIS
from tppis._sklearn_compat import sklearn_at_least


def _easy() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 12))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.standard_normal(40)
    return X, y


def test_clone_and_params() -> None:
    est = TPPIS(d=2, alpha=0.6, k=2)
    cloned = clone(est)
    assert cloned.get_params()["d"] == 2
    cloned.set_params(k=3)
    assert cloned.k == 3


def test_pipeline_and_transform() -> None:
    X, y = _easy()
    pipe = Pipeline(
        [
            ("select", TPPIS(d=2, alpha=0.6, k=2)),
            ("lm", LinearRegression()),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (X.shape[0],)
    Xt = pipe.named_steps["select"].transform(X)
    assert Xt.shape == (X.shape[0], 2)


def test_gridsearchcv_over_k() -> None:
    X, y = _easy()
    search = GridSearchCV(
        TPPIS(d=2, alpha=0.8),
        param_grid={"k": [1, 2, 3]},
        cv=3,
        scoring="r2",
    )
    search.fit(X, y)
    assert search.best_params_["k"] in {1, 2, 3}


@pytest.mark.skipif(
    not sklearn_at_least(1, 6),
    reason="official estimator checks use the sklearn 1.6 tags API",
)
@parametrize_with_checks(
    [
        SIS(k=1),
        TPPIS(d=0, alpha=1.0, k=1),
    ]
)
def test_sklearn_estimator_checks(estimator: object, check: object) -> None:
    check(estimator)
