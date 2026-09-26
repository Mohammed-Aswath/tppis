"""TPPIS: Truncated Preconditioned Profiled Independence Screening."""

from tppis.datasets import (
    make_example1,
    make_example2,
    make_example3,
    make_example4,
)
from tppis.exceptions import TPPISError
from tppis.metrics import screening_scores
from tppis.screeners import FPSIS, FPSISBIC, PPIS, SIS, TPPIS, screen

__version__ = "0.1.0"

__all__ = [
    "FPSIS",
    "FPSISBIC",
    "PPIS",
    "SIS",
    "TPPIS",
    "TPPISError",
    "make_example1",
    "make_example2",
    "make_example3",
    "make_example4",
    "screen",
    "screening_scores",
    "__version__",
]
