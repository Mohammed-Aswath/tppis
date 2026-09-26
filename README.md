# tppis

Open-source Python implementation of **Truncated Preconditioned Profiled Independence Screening** (TPPIS) from Tanaka and Matsui (2023), plus the baselines SIS, FPSIS (and FPSIS-BIC), and PPIS.

The method screens variables in high-dimensional regression when predictors are strongly multicollinear. A factor-analysis transform removes the common factors; TPPIS then *truncates* the tail of that transform so the unique-factor signal is not washed out.

**Maintainer:** [Mohammed Aswath M](https://github.com/Mohammed-Aswath) · [mohammed.aswath07@gmail.com](mailto:mohammed.aswath07@gmail.com) · issues: [github.com/Mohammed-Aswath/tppis](https://github.com/Mohammed-Aswath/tppis/issues)

## Install

```bash
pip install tppis
```

The package requires Python 3.10+, NumPy, SciPy, and **scikit-learn 1.6+**.

From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python 3.10+, NumPy, SciPy, and scikit-learn 1.6+.

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

See [`CITATION.cff`](CITATION.cff).

## License

MIT. See [`LICENSE`](LICENSE).
