"""Draw the README comparison figure for the Example 1 cell of Table 1.

    python reproduction/make_readme_figure.py

Needs the ``repro`` extra (``pip install -e ".[repro]"``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from tppis import FPSIS, FPSISBIC, PPIS, SIS, TPPIS, make_example1
from tppis.metrics import screening_scores

METHODS = {
    "SIS": lambda: SIS(k="bic"),
    "FPSIS": lambda: FPSIS(d="ratio", k="bic"),
    "FPSIS-BIC": lambda: FPSISBIC(d="grid", k="bic"),
    "PPIS": lambda: PPIS(d="ratio", k="bic"),
    "TPPIS": lambda: TPPIS(d="grid", alpha="grid", k="bic"),
}
COLORS = {
    "SIS": "#9ca3af",
    "FPSIS": "#60a5fa",
    "FPSIS-BIC": "#818cf8",
    "PPIS": "#fbbf24",
    "TPPIS": "#059669",
}


def _style() -> None:
    dejavu = (
        Path(matplotlib.__file__).resolve().parent
        / "mpl-data"
        / "fonts"
        / "ttf"
        / "DejaVuSans.ttf"
    )
    if dejavu.is_file():
        font_manager.fontManager.addfont(str(dejavu))
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )


def run(n: int, p: int, phi: float, reps: int) -> dict[str, dict[str, float]]:
    f2: dict[str, list[float]] = {name: [] for name in METHODS}
    x4: dict[str, int] = {name: 0 for name in METHODS}
    for seed in range(reps):
        data = make_example1(n=n, p=p, phi=phi, seed=seed)
        for name, factory in METHODS.items():
            est = factory().fit(data.X, data.y)
            f2[name].append(screening_scores(est.selected_, data.active, p).fbeta)
            x4[name] += int(3 in set(est.selected_.tolist()))
    return {
        name: {"f2": float(np.mean(f2[name])), "x4": 100.0 * x4[name] / reps}
        for name in METHODS
    }


def draw(
    summary: dict[str, dict[str, float]],
    n: int,
    p: int,
    phi: float,
    reps: int,
    out: Path,
) -> None:
    _style()
    names = list(METHODS)
    colors = [COLORS[name] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    panels = [
        (axes[0], [summary[m]["f2"] for m in names], "Mean F2 score", 1.0, "{:.2f}"),
        (
            axes[1],
            [summary[m]["x4"] for m in names],
            "Runs that keep x4 (%)",
            100.0,
            "{:.0f}",
        ),
    ]
    for ax, values, title, top, fmt in panels:
        bars = ax.bar(names, values, color=colors, width=0.62)
        ax.set_ylim(0, top * 1.12)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + top * 0.02,
                fmt.format(value),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        f"Example 1 of Tanaka and Matsui (2023): n={n}, p={p}, φ={phi}, {reps} runs",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--p", type=int, default=1000)
    parser.add_argument("--phi", type=float, default=0.5)
    parser.add_argument("--reps", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("docs/assets/example1_f2.png"))
    args = parser.parse_args()

    summary = run(args.n, args.p, args.phi, args.reps)
    draw(summary, args.n, args.p, args.phi, args.reps, args.out)
    print(f"wrote {args.out}")
    for name, row in summary.items():
        print(f"  {name:10s} mean F2={row['f2']:.3f}  x4 kept={row['x4']:.0f}%")


if __name__ == "__main__":
    main()
