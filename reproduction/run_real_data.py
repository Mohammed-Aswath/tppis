"""Reproduce Tables 5 and 6 once the real datasets have been fetched.

    python reproduction/run_real_data.py --hydraulic path/to/array.npy --response path/to/y.npy
    python reproduction/run_real_data.py --sp500-x path/to/X.npy --sp500-y path/to/y.npy

The paper's sources are documented by ``tppis.datasets.fetch_hydraulic`` and
``tppis.datasets.fetch_sp500_info``. This script does not download Kaggle data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tppis import FPSIS, FPSISBIC, PPIS, SIS, TPPIS
from tppis.datasets import fetch_hydraulic, fetch_sp500_info


def _run(X: np.ndarray, y: np.ndarray, label: str) -> None:
    methods = {
        "SIS": SIS(k="bic"),
        "FPSIS": FPSIS(d="ratio", k="bic"),
        "FPSISBIC": FPSISBIC(d="grid", k="bic"),
        "PPIS": PPIS(d="ratio", k="bic"),
        "TPPIS": TPPIS(d="grid", alpha="grid", k="bic"),
    }
    print(label, X.shape)
    for name, est in methods.items():
        est.fit(X, y)
        print(f"  {name:10s}  alpha={est.alpha_}  k={est.k_}  BIC={est.bic_:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hydraulic", type=Path, default=None)
    parser.add_argument("--response", type=Path, default=None)
    parser.add_argument("--sp500-x", type=Path, default=None)
    parser.add_argument("--sp500-y", type=Path, default=None)
    parser.add_argument("--fetch-hydraulic", action="store_true")
    args = parser.parse_args()

    if args.fetch_hydraulic:
        path = fetch_hydraulic()
        print("downloaded", path)
        print(fetch_sp500_info()["notes"])
        return
    if args.hydraulic is None and args.sp500_x is None:
        print(fetch_sp500_info())
        raise SystemExit(
            "Pass --hydraulic/--response or --sp500-x/--sp500-y arrays, "
            "or --fetch-hydraulic to cache the UCI zip."
        )
    if args.hydraulic is not None:
        X = np.load(args.hydraulic)
        y = np.load(args.response) if args.response else np.load(args.hydraulic.with_name("y.npy"))
        _run(X, y, "hydraulic")
    if args.sp500_x is not None:
        _run(np.load(args.sp500_x), np.load(args.sp500_y), "sp500")


if __name__ == "__main__":
    main()
