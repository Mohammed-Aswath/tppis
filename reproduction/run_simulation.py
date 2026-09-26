"""Reproduce Tables 1 to 4 of Tanaka and Matsui (2023).

The draw uses the Fan–Lv coefficient ``-15√φ`` on ``x4``, so that column is
uncorrelated with ``y``. Each method ranks by the marginal coefficient on its
transformed columns and chooses ``k`` by BIC on the original least-squares fit,
stopping before the interpolating size ``k = n-1``.

Default is a reduced smoke run. Pass ``--full`` for the paper's 100 replications.

    python reproduction/run_simulation.py
    python reproduction/run_simulation.py --full --example 1 --n 100 --phi 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tppis import FPSIS, FPSISBIC, PPIS, SIS, TPPIS
from tppis.datasets import make_example1, make_example2, make_example3, make_example4
from tppis.metrics import screening_scores

GENERATORS = {
    1: make_example1,
    2: make_example2,
    3: make_example3,
    4: make_example4,
}

METHODS = {
    "SIS": lambda: SIS(k="bic"),
    "FPSIS": lambda: FPSIS(d="ratio", k="bic"),
    "FPSISBIC": lambda: FPSISBIC(d="grid", k="bic"),
    "PPIS": lambda: PPIS(d="ratio", k="bic"),
    "TPPIS": lambda: TPPIS(d="grid", alpha="grid", k="bic"),
}


def one_rep(example: int, n: int, p: int, phi: float, seed: int, m: int | None) -> dict[str, object]:
    if example == 4:
        data = make_example4(n=n, p=p, m=m, seed=seed)
    else:
        data = GENERATORS[example](n=n, p=p, phi=phi, seed=seed)
    row: dict[str, object] = {"example": example, "n": n, "p": p, "phi": phi, "seed": seed}
    for name, factory in METHODS.items():
        est = factory().fit(data.X, data.y)
        scores = screening_scores(est.selected_, data.active, p)
        row[f"{name}_bic"] = est.bic_
        row[f"{name}_k"] = est.k_
        row[f"{name}_f2"] = scores.fbeta
        row[f"{name}_alpha"] = est.alpha_
        for j in data.active:
            row[f"{name}_x{j + 1}"] = int(j in set(est.selected_.tolist()))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="100 replications, paper grids.")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--p", type=int, default=30)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--m", type=int, default=None, help="Example 4 medium-factor count.")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reproduction/results/simulation.csv"))
    args = parser.parse_args()

    n = 100 if args.full and args.n == 40 else args.n
    p = 1000 if args.full and args.p == 30 else args.p
    reps = 100 if args.full else args.reps

    rows = [
        one_rep(args.example, n, p, args.phi, seed=i, m=args.m)
        for i in range(reps)
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row[k]) for k in keys) + "\n")

    print(f"wrote {args.out} ({reps} replications)")
    for name in METHODS:
        f2 = float(np.mean([row[f"{name}_f2"] for row in rows]))
        print(f"  {name:10s}  mean F2={f2:.3f}")


if __name__ == "__main__":
    main()
