# Example 1 of the paper

Four active variables. The coefficient on `x₄` is `−15√φ`, so
`Cov(y, x₄) = 0` under the Example 1 covariance. SIS is therefore blind to
that column; the factor methods are not.

```python
from tppis import SIS, TPPIS, make_example1, screening_scores

data = make_example1(n=100, p=80, phi=0.7, seed=0)

sis = SIS(k="bic").fit(data.X, data.y)
tppis = TPPIS(d=8, alpha=0.8, k="bic").fit(data.X, data.y)

print("SIS   ", sis.selected_, screening_scores(sis.selected_, data.active, data.X.shape[1]).fbeta)
print("TPPIS ", tppis.selected_, screening_scores(tppis.selected_, data.active, data.X.shape[1]).fbeta)
```
