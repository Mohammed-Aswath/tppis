"""Screening metrics: confusion counts, precision, recall, and F-beta.

The paper reports the F2-score, weighting recall, because a screening step
is allowed to over-select but must not drop a truly active variable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ScreeningScores:
    """Confusion counts and derived scores for one selected set."""

    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    fbeta: float
    beta: float

    def as_dict(self) -> dict[str, float]:
        """Flat mapping, convenient for tables."""
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "fbeta": self.fbeta,
            "beta": self.beta,
        }


def confusion(
    selected: ArrayLike,
    active: ArrayLike,
    p: int,
) -> tuple[int, int, int, int]:
    """True/false positive and negative counts.

    Parameters
    ----------
    selected :
        Indices chosen by the screener.
    active :
        Indices of variables with nonzero true coefficients.
    p :
        Total number of predictors.
    """
    sel = np.unique(np.asarray(selected, dtype=np.intp))
    act = np.unique(np.asarray(active, dtype=np.intp))
    tp = int(np.intersect1d(sel, act, assume_unique=True).size)
    fp = int(sel.size - tp)
    fn = int(act.size - tp)
    tn = int(p - tp - fp - fn)
    return tp, fp, tn, fn


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Precision and recall, defined as 0 when the denominator is 0."""
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return prec, rec


def fbeta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Weighted F-score of specification Section 7.

    Parameters
    ----------
    precision, recall :
        In ``[0, 1]``.
    beta :
        Recall weight. The paper uses ``beta = 2``.
    """
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta**2
    return float((1.0 + b2) * precision * recall / (recall + b2 * precision))


def screening_scores(
    selected: ArrayLike,
    active: ArrayLike,
    p: int,
    *,
    beta: float = 2.0,
) -> ScreeningScores:
    """End-to-end metric bundle used in the paper's tables.

    Examples
    --------
    >>> from tppis.metrics import screening_scores
    >>> s = screening_scores([0, 1, 4], [0, 1, 2], p=10, beta=2)
    >>> round(s.fbeta, 3)
    0.714
    """
    tp, fp, tn, fn = confusion(selected, active, p)
    prec, rec = precision_recall(tp, fp, fn)
    return ScreeningScores(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=prec,
        recall=rec,
        fbeta=fbeta_score(prec, rec, beta),
        beta=float(beta),
    )


def selection_indicators(
    selected: ArrayLike,
    variables: ArrayLike,
) -> NDArray[np.intp]:
    """0/1 indicators for whether each listed variable was selected."""
    sel = set(np.asarray(selected, dtype=np.intp).tolist())
    vars_ = np.asarray(variables, dtype=np.intp)
    return np.asarray([int(v in sel) for v in vars_], dtype=np.intp)
