"""Validation helpers, fetchers, and remaining edge paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from tppis import SIS, screen
from tppis._validation import (
    apply_standardize,
    as_float_arrays,
    check_literal,
    standardize,
)
from tppis.criteria import bic_value, default_k_max
from tppis.datasets import fetch_hydraulic, fetch_sp500_info, make_example4
from tppis.datasets.real import verify_checksum
from tppis.exceptions import TPPISError
from tppis.factors import ratio_d
from tppis.kernel import spectral_slice, transformed_response
from tppis.metrics import screening_scores, selection_indicators
from tppis.spectral import alpha_to_m, decompose
from tppis.tuning import alpha_candidates, d_candidates, iter_valid_pairs


def test_as_float_arrays_rejects_bad_shapes() -> None:
    with pytest.raises(TPPISError, match="2-dimensional"):
        as_float_arrays(np.arange(3.0), np.arange(3.0))
    with pytest.raises(TPPISError, match="at least 2"):
        as_float_arrays(np.ones((1, 2)), [1.0])
    with pytest.raises(TPPISError, match="at least one column"):
        as_float_arrays(np.ones((3, 0)), [1.0, 2.0, 3.0])
    with pytest.raises(TPPISError, match="first dimension"):
        as_float_arrays(np.ones((3, 2)), [1.0, 2.0])
    with pytest.raises(TPPISError, match="non-finite"):
        as_float_arrays(np.array([[1.0, np.nan], [0.0, 1.0]]), [0.0, 1.0])
    with pytest.raises(TPPISError, match="non-finite"):
        as_float_arrays(np.ones((3, 2)), [1.0, np.inf, 0.0])


def test_apply_standardize_and_literal() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = np.array([1.0, 2.0, 3.0])
    Xs, yc, mean, scale, ymean = standardize(X, y, enabled=True)
    out = apply_standardize(X, mean, scale)
    np.testing.assert_allclose(out, Xs)
    with pytest.raises(TPPISError, match="2-dimensional"):
        apply_standardize(np.arange(2.0), mean, scale)
    with pytest.raises(TPPISError, match="columns"):
        apply_standardize(np.ones((2, 3)), mean, scale)
    assert check_literal("backend", "svd", {"svd", "gram"}) == "svd"
    with pytest.raises(TPPISError, match="must be one of"):
        check_literal("backend", "qr", {"svd", "gram"})


def test_standardize_disabled_and_constant_column() -> None:
    X = np.array([[1.0, 5.0], [1.0, 6.0], [1.0, 7.0]])
    y = np.array([0.0, 1.0, 2.0])
    Xs, yc, *_ = standardize(X, y, enabled=False)
    np.testing.assert_array_equal(Xs, X)
    standardize(X, y, enabled=True)  # constant first column warns


def test_ratio_d_and_k_max_errors() -> None:
    with pytest.raises(TPPISError, match="at least two"):
        ratio_d(np.array([1.0]))
    with pytest.raises(TPPISError, match="collapsed"):
        default_k_max(p=3, n=2, n_retained=0, override=None)
    with pytest.raises(TPPISError, match="at least 1"):
        default_k_max(p=3, n=10, n_retained=4, override=0)
    with pytest.raises(TPPISError, match="n-1"):
        default_k_max(p=1000, n=100, n_retained=99, override=999)
    assert default_k_max(p=1000, n=100, n_retained=99, override=99) == 99


def test_bic_rss_mean_and_floor() -> None:
    paper = bic_value(100.0, 4, 100, 1000, kind="paper")
    mean = bic_value(100.0, 4, 100, 1000, kind="rss_mean")
    assert mean < paper
    floor = bic_value(0.0, 1, 10, 5, kind="paper")
    assert np.isfinite(floor)


def test_transformed_response_and_indicators() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 10))
    y = rng.standard_normal(16)
    spec = decompose(X, tol=0.0)
    y_hat = transformed_response(spec, slice(2, 8), -1, y)
    assert y_hat.shape == (16,)
    np.testing.assert_array_equal(selection_indicators([0, 2], [0, 1, 2]), [1, 0, 1])
    assert screening_scores([0], [0], p=2).as_dict()["tp"] == 1
    with pytest.raises(TPPISError, match="empty"):
        spectral_slice(spec, slice(5, 5))


def test_tuning_helpers() -> None:
    spec = decompose(np.random.default_rng(0).standard_normal((20, 12)), tol=0.0)
    assert d_candidates(spec, 4, n=20) == [4]
    assert d_candidates(spec, "ratio", n=20)[0] >= 1
    with pytest.raises(TPPISError, match="d must be"):
        d_candidates(spec, "nope", n=20)
    assert alpha_candidates(0.4) == [0.4]
    with pytest.raises(TPPISError, match="alpha must be"):
        alpha_candidates("nope")  # type: ignore[arg-type]
    pairs = iter_valid_pairs(spec, [2, 19], [0.4, 1.0], "floor")
    assert (2, 1.0) in pairs
    assert (19, 0.4) not in pairs


def test_fetchers(tmp_path: Path) -> None:
    info = fetch_sp500_info()
    assert "fred_url" in info

    def fake(url: str, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"abc")
        return dest

    path = fetch_hydraulic(tmp_path, downloader=fake)
    assert path.exists()
    digest = __import__("hashlib").sha256(b"abc").hexdigest()
    assert verify_checksum(path, digest)
    assert not verify_checksum(path, "0" * 64)


def test_spectral_and_alpha_errors() -> None:
    with pytest.raises(TPPISError, match="alpha must lie"):
        alpha_to_m(10, 0.0)
    with pytest.raises(TPPISError, match="All singular"):
        decompose(np.zeros((6, 8)), tol=1.0)


def test_screen_unknown_method_and_k_zero() -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8))
    y = rng.standard_normal(20)
    with pytest.raises(ValueError, match="Unknown method"):
        screen(X, y, method="holp")  # type: ignore[arg-type]
    with pytest.raises(TPPISError, match="at least 1"):
        SIS(k=0).fit(X, y)
    with pytest.raises(ValueError, match="requires y"):
        SIS(k=1).fit(X, None)


def test_example4_default_m() -> None:
    data = make_example4(n=20, p=15, seed=0)
    assert data.X.shape == (20, 15)
    assert data.active.tolist() == [0, 1, 2, 3]
