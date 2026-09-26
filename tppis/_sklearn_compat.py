"""scikit-learn 1.3 and 1.6 input checks behind one function.

``validate_data`` and ``__sklearn_tags__`` exist from sklearn 1.6, which
needs Python 3.9+. Python 3.8 is limited to sklearn 1.3, which uses
``_validate_data`` and ``_more_tags``. Callers of TPPIS do not see this.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

try:
    from sklearn.utils.validation import validate_data as _validate_data_fn
except ImportError:  # sklearn < 1.6
    _validate_data_fn = None


def sklearn_at_least(major: int, minor: int) -> bool:
    """True when the installed scikit-learn is at least ``major.minor``."""
    import sklearn

    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


def check_xy(
    estimator: Any,
    X: ArrayLike,
    y: ArrayLike,
    *,
    reset: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate ``X`` and ``y`` and set sklearn's ``n_features_in_``."""
    if _validate_data_fn is not None:
        X_arr, y_arr = _validate_data_fn(
            estimator,
            X,
            y,
            dtype=np.float64,
            accept_sparse=False,
            ensure_all_finite=True,
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=True,
            reset=reset,
        )
    else:
        X_arr, y_arr = estimator._validate_data(
            X,
            y,
            dtype=np.float64,
            accept_sparse=False,
            force_all_finite=True,
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=True,
            reset=reset,
        )
    X_out = np.ascontiguousarray(X_arr, dtype=np.float64)
    y_out = np.ascontiguousarray(np.asarray(y_arr, dtype=np.float64).reshape(-1))
    return X_out, y_out


def check_X(
    estimator: Any,
    X: ArrayLike,
    *,
    reset: bool,
) -> NDArray[np.float64]:
    """Validate ``X`` for ``predict`` / ``transform`` (``y`` is not required)."""
    if _validate_data_fn is not None:
        X_arr = _validate_data_fn(
            estimator,
            X,
            dtype=np.float64,
            accept_sparse=False,
            ensure_all_finite=True,
            reset=reset,
        )
    else:
        X_arr = estimator._validate_data(
            X,
            dtype=np.float64,
            accept_sparse=False,
            force_all_finite=True,
            reset=reset,
        )
    return np.ascontiguousarray(X_arr, dtype=np.float64)
