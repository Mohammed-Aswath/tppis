"""Reproduce Figure 1: BIC and F2 versus alpha at a fixed d.

    python reproduction/make_figure1.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tppis import TPPIS, make_example1
from tppis.metrics import screening_scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=80)
    parser.add_argument("--p", type=int, default=40)
    parser.add_argument("--phi", type=float, default=0.7)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reproduction/results/figure1.csv"))
    args = parser.parse_args()

    data = make_example1(n=args.n, p=args.p, phi=args.phi, seed=args.seed)
    est = TPPIS(d=args.d, alpha="grid", k="bic").fit(data.X, data.y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("d,alpha,k,bic,f2,skipped\n")
        for row in est.grid_:
            f2 = ""
            if not row["skipped"]:
                # Re-fit this alpha to recover the selected set for F2.
                one = TPPIS(d=args.d, alpha=row["alpha"], k="bic").fit(data.X, data.y)
                f2 = screening_scores(one.selected_, data.active, args.p).fbeta
            handle.write(
                f"{row['d']},{row['alpha']},{row['k']},{row['bic']},{f2},{row['skipped']}\n"
            )
    print(f"wrote {args.out}")
    print("fitted alpha", est.alpha_, "k", est.k_, "BIC", est.bic_)


if __name__ == "__main__":
    main()
