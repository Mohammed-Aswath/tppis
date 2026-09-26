"""Public SIS / FPSIS / PPIS / TPPIS estimators.

One kernel, five estimators. They differ only in the spectral index set, the
exponent ``g``, and how ``d`` is chosen. See specification Section 4.3.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.metrics import r2_score
from sklearn.utils.validation import check_is_fitted

from tppis._sklearn_compat import check_X, check_xy
from tppis._validation import TPPISError, check_literal, standardize
from tppis.spectral import decompose
from tppis.tuning import FitResult, MethodName, search

MethodLiteral = Literal["sis", "fpsis", "fpsis_bic", "ppis", "tppis"]


class _BaseScreener(SelectorMixin, BaseEstimator):  # type: ignore[misc]
    """Shared scikit-learn estimator for every method in the paper.

    Parameters
    ----------
    d :
        Factor count, ``"ratio"`` for equation (3), or ``"grid"`` to search.
    alpha :
        Truncation parameter in ``(0, 1]``, or ``"grid"``. Used only by TPPIS.
    k :
        Number of selected variables, or ``"bic"`` to minimize equation (10).
        An integer ``k`` must satisfy ``1 <= k <= n - 1``. ``k <= p - 1`` is
        not enough: after centering, ``k >= n`` interpolates ``y``.
    alpha_rounding :
        How ``[n * alpha]`` is read (A-1). ``"floor"`` is the paper default.
    bic :
        ``"paper"`` is unnormalized ``log(RSS)``; ``"rss_mean"`` uses ``RSS / n``.
    backend :
        ``"svd"`` (default) or ``"gram"`` (large-``p`` escape hatch).
    tol :
        Relative singular-value cutoff (A-6). ``None`` uses ``max(n, p) * eps``.
    standardize :
        Column-standardize ``X`` with ``ddof=0`` and center ``y``.
    k_max :
        Override for the A-9 cap ``min(p, rank(X_hat), n - 1)``. Values
        ``>= n`` are rejected. The BIC search never goes past ``n - 1``.
    d_grid, alpha_grid :
        Optional custom grids replacing the paper's defaults.
    random_state :
        Accepted for scikit-learn compatibility. The method is deterministic.
    """

    _method: MethodName = "tppis"

    def __init__(
        self,
        d: int | str = "grid",
        alpha: float | str = "grid",
        k: int | str = "bic",
        *,
        alpha_rounding: str = "floor",
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        d_grid: Sequence[float] | None = None,
        alpha_grid: Sequence[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        self.d = d
        self.alpha = alpha
        self.k = k
        self.alpha_rounding = alpha_rounding
        self.bic = bic
        self.backend = backend
        self.tol = tol
        self.standardize = standardize
        self.k_max = k_max
        self.d_grid = d_grid
        self.alpha_grid = alpha_grid
        self.random_state = random_state

    def _more_tags(self) -> dict[str, Any]:
        return {"requires_y": True, "poor_score": True, "allow_nan": False}

    def __sklearn_tags__(self) -> Any:
        parent = getattr(super(), "__sklearn_tags__", None)
        if parent is None:
            return {"requires_y": True, "poor_score": True, "allow_nan": False}
        tags = parent()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        tags.input_tags.sparse = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike) -> _BaseScreener:
        """Fit the screener.

        Parameters
        ----------
        X :
            Predictor matrix of shape ``(n, p)``.
        y :
            Response of shape ``(n,)``.

        Returns
        -------
        self
        """
        check_literal("alpha_rounding", self.alpha_rounding, {"floor", "round"})
        check_literal("bic", self.bic, {"paper", "rss_mean"})
        check_literal("backend", self.backend, {"svd", "gram"})
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")
        X_arr, y_arr = check_xy(self, X, y, reset=True)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise TPPISError(
                f"X and y must share the first dimension, got n={X_arr.shape[0]} "
                f"and y.size={y_arr.size}."
            )
        X_std, y_c, x_mean, x_scale, y_mean = standardize(
            X_arr, y_arr, enabled=self.standardize
        )
        self.x_mean_ = x_mean
        self.x_scale_ = x_scale
        self.y_mean_ = y_mean
        self.n_samples_ = int(X_std.shape[0])
        spec = decompose(X_std, backend=self.backend, tol=self.tol)
        self.n_dropped_ = spec.n_dropped
        result: FitResult = search(
            X_std,
            y_c,
            spec,
            self._method,
            d=self.d,
            alpha=self.alpha,
            k=self.k,
            k_max=self.k_max,
            bic_kind=self.bic,
            rounding=self.alpha_rounding,
            d_fractions=self.d_grid,
            alpha_grid=self.alpha_grid,
        )
        self._set_from_result(result)
        return self

    def _set_from_result(self, result: FitResult) -> None:
        self.omega_ = result.omega
        self.scores_ = np.abs(result.omega)
        self.ranking_ = result.order
        self.k_ = int(result.k)
        self.selected_ = result.order[: result.k].copy()
        self.d_ = result.d
        self.alpha_ = result.alpha
        self.bic_ = float(result.bic)
        self.coef_ = result.beta
        self.grid_ = result.grid
        self.n_skipped_ = int(result.n_skipped)

    def _get_support_mask(self) -> NDArray[np.bool_]:
        mask = np.zeros(self.n_features_in_, dtype=bool)
        mask[self.selected_] = True
        return mask

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict on the original scale from the ordinary-least-squares refit.

        New rows are standardized with the training column means and scales,
        the selected columns are multiplied by ``coef_``, and the training
        mean of ``y`` is added back.

        Parameters
        ----------
        X :
            Design of shape ``(n_new, p)``.

        Returns
        -------
        y_hat
            Predictions of shape ``(n_new,)``.
        """
        check_is_fitted(self)
        X_arr = check_X(self, X, reset=False)
        if self.standardize:
            X_arr = (X_arr - self.x_mean_) / self.x_scale_
        pred = X_arr[:, self.selected_] @ self.coef_ + self.y_mean_
        return np.asarray(pred, dtype=np.float64)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """R² of :meth:`predict` against ``y``."""
        return float(r2_score(y, self.predict(X)))


class SIS(_BaseScreener):
    """Sure Independence Screening (Fan and Lv, 2008).

    ``omega = X.T @ y``. No factor-analysis transform.

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import SIS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 12))
    >>> y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=40)
    >>> SIS(k=2).fit(X, y).get_support(indices=True).tolist()
    [0, 1]
    """

    _method: MethodName = "sis"

    def __init__(
        self,
        k: int | str = "bic",
        *,
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            d=0,
            alpha=1.0,
            k=k,
            bic=bic,
            backend=backend,
            tol=tol,
            standardize=standardize,
            k_max=k_max,
            random_state=random_state,
        )


class FPSIS(_BaseScreener):
    """Factor Profiled Sure Independence Screening (Wang, 2012).

    Projects out the leading ``d`` left singular vectors (equation (4)) and
    chooses ``d`` by the singular-value ratio (3) unless a value is given.

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import FPSIS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 12))
    >>> y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=40)
    >>> FPSIS(d=2, k=2).fit(X, y).k_
    2
    """

    _method: MethodName = "fpsis"

    def __init__(
        self,
        d: int | str = "ratio",
        k: int | str = "bic",
        *,
        alpha_rounding: str = "floor",
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        d_grid: Sequence[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            d=d,
            alpha=1.0,
            k=k,
            alpha_rounding=alpha_rounding,
            bic=bic,
            backend=backend,
            tol=tol,
            standardize=standardize,
            k_max=k_max,
            d_grid=d_grid,
            random_state=random_state,
        )


class FPSISBIC(_BaseScreener):
    """FPSIS with ``d`` chosen by the BIC-type criterion (10).

    The paper's ``FPSIS_BIC`` benchmark, used to isolate the gain from choosing
    ``d`` by BIC versus the gain from truncation.

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import FPSISBIC
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(30, 10))
    >>> y = X[:, 0] + rng.normal(scale=0.1, size=30)
    >>> est = FPSISBIC(k=1).fit(X, y)
    >>> int(est.selected_[0]) in range(10)
    True
    """

    _method: MethodName = "fpsis_bic"

    def __init__(
        self,
        d: int | str = "grid",
        k: int | str = "bic",
        *,
        alpha_rounding: str = "floor",
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        d_grid: Sequence[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            d=d,
            alpha=1.0,
            k=k,
            alpha_rounding=alpha_rounding,
            bic=bic,
            backend=backend,
            tol=tol,
            standardize=standardize,
            k_max=k_max,
            d_grid=d_grid,
            random_state=random_state,
        )


class PPIS(_BaseScreener):
    """Preconditioned Profiled Independence Screening (Zhao et al., 2020).

    Puffer-style whitening of the tail spectrum (equation (6)). Equivalent to
    :class:`TPPIS` at ``alpha = 1`` with matching ``d``.

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import PPIS, TPPIS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 15))
    >>> y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=40)
    >>> a = PPIS(d=3, k=2).fit(X, y).omega_
    >>> b = TPPIS(d=3, alpha=1.0, k=2).fit(X, y).omega_
    >>> np.allclose(a, b)
    True
    """

    _method: MethodName = "ppis"

    def __init__(
        self,
        d: int | str = "ratio",
        k: int | str = "bic",
        *,
        alpha_rounding: str = "floor",
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        d_grid: Sequence[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            d=d,
            alpha=1.0,
            k=k,
            alpha_rounding=alpha_rounding,
            bic=bic,
            backend=backend,
            tol=tol,
            standardize=standardize,
            k_max=k_max,
            d_grid=d_grid,
            random_state=random_state,
        )


class TPPIS(_BaseScreener):
    """Truncated Preconditioned Profiled Independence Screening.

    Equation (7) of Tanaka and Matsui (2023). Default ``d`` and ``alpha`` are
    chosen jointly by the BIC-type criterion (10).

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import TPPIS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 15))
    >>> y = X[:, 0] + X[:, 1] + rng.normal(scale=0.1, size=40)
    >>> est = TPPIS(d=2, alpha=0.6, k=2).fit(X, y)
    >>> set(est.get_support(indices=True).tolist()) == {0, 1}
    True
    """

    _method: MethodName = "tppis"

    def __init__(
        self,
        d: int | str = "grid",
        alpha: float | str = "grid",
        k: int | str = "bic",
        *,
        alpha_rounding: str = "floor",
        bic: str = "paper",
        backend: str = "svd",
        tol: float | None = None,
        standardize: bool = True,
        k_max: int | None = None,
        d_grid: Sequence[float] | None = None,
        alpha_grid: Sequence[float] | None = None,
        random_state: int | None = None,
    ) -> None:
        super().__init__(
            d=d,
            alpha=alpha,
            k=k,
            alpha_rounding=alpha_rounding,
            bic=bic,
            backend=backend,
            tol=tol,
            standardize=standardize,
            k_max=k_max,
            d_grid=d_grid,
            alpha_grid=alpha_grid,
            random_state=random_state,
        )


_DISPATCH: dict[MethodLiteral, type[_BaseScreener]] = {
    "sis": SIS,
    "fpsis": FPSIS,
    "fpsis_bic": FPSISBIC,
    "ppis": PPIS,
    "tppis": TPPIS,
}


def screen(
    X: ArrayLike,
    y: ArrayLike,
    method: MethodLiteral = "tppis",
    **kwargs: Any,
) -> _BaseScreener:
    """Fit a screener in one call.

    Parameters
    ----------
    X, y :
        Design and response.
    method :
        One of ``"sis"``, ``"fpsis"``, ``"fpsis_bic"``, ``"ppis"``, ``"tppis"``.
    **kwargs :
        Forwarded to the corresponding estimator.

    Returns
    -------
    estimator
        A fitted instance. All fitted attributes (``selected_``, ``grid_``,
        ``bic_``, ...) are available on the return value.

    Examples
    --------
    >>> import numpy as np
    >>> from tppis import screen
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 12))
    >>> y = X[:, 0] + rng.normal(scale=0.1, size=40)
    >>> result = screen(X, y, method="sis", k=1)
    >>> int(result.selected_[0])
    0
    """
    if method not in _DISPATCH:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(_DISPATCH)}.")
    return _DISPATCH[method](**kwargs).fit(X, y)
