"""Warning and error types raised by :mod:`tppis`.

Every degenerate situation the package can meet has a dedicated type here, so that
callers can silence or escalate them individually. Requirement NFR-10 of the plan
forbids handling any of these silently.
"""

from __future__ import annotations

__all__ = [
    "BoundarySelectionWarning",
    "ConstantFeatureWarning",
    "EmptyGridError",
    "RankDeficiencyWarning",
    "TPPISError",
    "TppisWarning",
]


class TppisWarning(UserWarning):
    """Base class for every warning raised by :mod:`tppis`."""


class RankDeficiencyWarning(TppisWarning):
    """Singular values were discarded, or a Cholesky factorization broke down.

    Raised when the design is numerically rank deficient, which happens with
    duplicated or exactly collinear columns. See decision A-6 of the specification.
    """


class ConstantFeatureWarning(TppisWarning):
    """One or more columns of ``X`` have zero variance.

    Such columns carry no information, cannot be standardized, and are held at zero
    importance so that they are never selected.
    """


class BoundarySelectionWarning(TppisWarning):
    """The criterion was minimized at the edge of the search range.

    The chosen value is then an artefact of the range rather than a true optimum, and
    the range should be widened. See decision A-9 of the specification.
    """


class TPPISError(ValueError):
    """Invalid input or an empty parameter grid."""


class EmptyGridError(TPPISError):
    """No point of the parameter grid satisfies the constraint ``0 <= d < m <= rank``.

    See decision A-2 of the specification.
    """
