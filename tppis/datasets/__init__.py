"""Simulation generators and real-data fetchers."""

from tppis.datasets.real import fetch_hydraulic, fetch_sp500_info
from tppis.datasets.simulate import (
    SimulatedData,
    example1_sigma,
    example2_sigma,
    make_example1,
    make_example2,
    make_example3,
    make_example4,
    min_eigenvalue,
)

__all__ = [
    "SimulatedData",
    "example1_sigma",
    "example2_sigma",
    "fetch_hydraulic",
    "fetch_sp500_info",
    "make_example1",
    "make_example2",
    "make_example3",
    "make_example4",
    "min_eigenvalue",
]
