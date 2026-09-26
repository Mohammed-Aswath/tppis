# Quickstart

## Install

```bash
pip install tppis
```

## Fit TPPIS

```python
from tppis import TPPIS, screen, make_example1

data = make_example1(n=100, p=60, phi=0.7, seed=1)

# Grid search over d, alpha and k (paper defaults).
est = TPPIS().fit(data.X, data.y)
est.selected_, est.d_, est.alpha_, est.k_, est.bic_

# Or fix the parameters.
TPPIS(d=10, alpha=0.4, k=8).fit(data.X, data.y)

# Functional form, same object.
result = screen(data.X, data.y, method="tppis", d=10, alpha=0.4, k=8)
```

## scikit-learn pipeline

```python
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from tppis import TPPIS

pipe = Pipeline([
    ("select", TPPIS(d=8, alpha=0.6, k=10)),
    ("lm", LinearRegression()),
])
pipe.fit(data.X, data.y)
pipe.predict(data.X)
```

## Options

```python
TPPIS(
    alpha_rounding="floor",  # A-1
    bic="paper",             # A-4, natural log of unnormalized RSS
    backend="svd",           # "gram" for very large p
    standardize=True,        # A-5, ddof=0
)
```
