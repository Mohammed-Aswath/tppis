"""Hand-checked F-beta and confusion counts."""

from __future__ import annotations

from tppis.metrics import fbeta_score, screening_scores


def test_perfect_selection() -> None:
    s = screening_scores([0, 1], [0, 1], p=10, beta=2)
    assert s.tp == 2 and s.fp == 0 and s.fn == 0 and s.tn == 8
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.fbeta == 1.0


def test_hand_computed_f2() -> None:
    # selected {0, 1, 4}, active {0, 1, 2}: TP=2, FP=1, FN=1
    # precision = 2/3, recall = 2/3
    # F2 = 5*(2/3)*(2/3) / ((2/3)+4*(2/3)) = 2/3
    s = screening_scores([0, 1, 4], [0, 1, 2], p=10, beta=2)
    assert s.tp == 2 and s.fp == 1 and s.fn == 1
    assert abs(s.precision - 2 / 3) < 1e-12
    assert abs(s.recall - 2 / 3) < 1e-12
    assert abs(s.fbeta - 2 / 3) < 1e-12


def test_zero_precision_and_recall() -> None:
    assert fbeta_score(0.0, 0.0, beta=2) == 0.0
    s = screening_scores([5], [0, 1], p=6, beta=2)
    assert s.tp == 0 and s.fbeta == 0.0
