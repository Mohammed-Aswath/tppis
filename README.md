# tppis

[![PyPI](https://img.shields.io/pypi/v/tppis.svg)](https://pypi.org/project/tppis/)
[![Python versions](https://img.shields.io/pypi/pyversions/tppis.svg)](https://pypi.org/project/tppis/)
[![CI](https://github.com/Mohammed-Aswath/tppis/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Mohammed-Aswath/tppis/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/Mohammed-Aswath/tppis/branch/main/graph/badge.svg)](https://codecov.io/gh/Mohammed-Aswath/tppis)
[![Docs](https://github.com/Mohammed-Aswath/tppis/actions/workflows/docs.yml/badge.svg?branch=main)](https://mohammed-aswath.github.io/tppis/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Mohammed-Aswath/tppis/blob/main/LICENSE)

Open-source Python implementation of **Truncated Preconditioned Profiled Independence Screening** (TPPIS) from Tanaka and Matsui (2023), plus the baselines SIS, FPSIS (and FPSIS-BIC), and PPIS.

The method screens variables in high-dimensional regression when predictors are strongly multicollinear. A factor-analysis transform removes the common factors; TPPIS then *truncates* the tail of that transform so the unique-factor signal is not washed out.

**Maintainer:** [Mohammed Aswath M](https://github.com/Mohammed-Aswath) · [mohammed.aswath07@gmail.com](mailto:mohammed.aswath07@gmail.com) · issues: [github.com/Mohammed-Aswath/tppis](https://github.com/Mohammed-Aswath/tppis/issues)

**Documentation:** [mohammed-aswath.github.io/tppis](https://mohammed-aswath.github.io/tppis/)

## Results

![Mean F2 score and how often x4 is kept, for the five methods on Example 1](https://raw.githubusercontent.com/Mohammed-Aswath/tppis/main/docs/assets/example1_f2.png)

Example 1 of the paper, at the Table 1 setting n = 100, p = 1000, φ = 0.5, averaged
over 50 simulated datasets. All methods choose their parameters by BIC. F2 weights
recall twice as much as precision, which is the paper's metric.

| Method | Mean F2 | Runs that keep x4 |
| --- | --- | --- |
| SIS | 0.276 | 0% |
| FPSIS | 0.971 | 96% |
| FPSIS-BIC | 1.000 | 100% |
| PPIS | 0.900 | 86% |
| TPPIS | 1.000 | 100% |

The true predictor x4 has zero marginal correlation with y. SIS ranks columns by
marginal correlation, so it never finds x4. The factor-based methods remove the
shared factors first, which recovers it.

This is one cell of Table 1. The other examples can be rerun with the scripts in
[`reproduction/`](https://github.com/Mohammed-Aswath/tppis/tree/main/reproduction),
and they do not all match the paper this closely. To redraw the figure:

```bash
pip install -e ".[repro]"
python reproduction/make_readme_figure.py
```

## Install

```bash
pip install tppis
```

The package requires Python 3.8+. On 3.9+ it uses scikit-learn 1.6+; on 3.8 it
uses scikit-learn 1.3. The TPPIS API is the same.

From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.8+, NumPy, SciPy, and scikit-learn.

## Quickstart

```python
from tppis import TPPIS, make_example1

data = make_example1(n=100, p=80, phi=0.7, seed=0)
est = TPPIS().fit(data.X, data.y)

est.get_support(indices=True)   # selected columns, ranking order
est.d_, est.alpha_, est.k_      # BIC-chosen parameters
est.bic_
est.grid_                       # every (d, alpha, k, bic) evaluated
```

Fixed parameters, no search:

```python
TPPIS(d=20, alpha=0.4, k=10).fit(data.X, data.y)
```

Inside a scikit-learn pipeline:

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from tppis import TPPIS

pipe = Pipeline([
    ("select", TPPIS(d=8, alpha=0.6, k=10)),
    ("lm", LinearRegression()),
])
pipe.fit(data.X, data.y)
```

The functional entry point `screen(X, y, method="tppis")` returns the same fitted estimator.

## When to use which method

| Method | Transform | How `d` is chosen |
| --- | --- | --- |
| SIS | none (`X.T @ y`) | — |
| FPSIS | project out leading `d` factors | singular-value ratio (3) |
| FPSIS-BIC | same | BIC grid |
| PPIS | Puffer whitening of the tail | singular-value ratio (3) |
| TPPIS | PPIS with the tail truncated at `[n α]` | joint BIC over `d`, `α`, `k` |

TPPIS at `alpha=1` is exactly PPIS. That identity is tested.

## Citation

If you use the method, cite the paper. If you use this software, cite the package as well.

```
Tanaka, S. and Matsui, H. (2023). Variable screening using factor analysis
for high-dimensional data with multicollinearity. arXiv:2306.05702.
```

See [`CITATION.cff`](https://github.com/Mohammed-Aswath/tppis/blob/main/CITATION.cff).

## License

MIT. See [`LICENSE`](https://github.com/Mohammed-Aswath/tppis/blob/main/LICENSE).
