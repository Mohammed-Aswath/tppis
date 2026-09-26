# Theory

A short restatement of Tanaka and Matsui (2023) as implemented here.
Equation numbers refer to the paper.

## Model

`y = X β + ε` with `n < p`, columns of `X` standardized, `y` centered.
The thin SVD is `X = U D Vᵀ` (equation (2)).

## One kernel

Every method is

```
ω(S, g) = V_S D_S^g U_Sᵀ y
```

implemented as `ω = Xᵀ w` with `w = U_S μ_S^{g-1} U_Sᵀ y`, so `V` is never formed.

| Method | Index set S | Exponent g |
| --- | --- | --- |
| SIS | `1 .. r` | `+1` |
| FPSIS / FPSIS-BIC | `d+1 .. r` | `+1` |
| PPIS | `d+1 .. r` | `-1` |
| TPPIS | `d+1 .. ⌊nα⌋` | `-1` |

TPPIS at `α = 1` is PPIS.

## Ranking, refit, and BIC

Importance is the marginal coefficient on the transformed columns,
`(x̂ⱼ · ŷ) / ‖x̂ⱼ‖²`. The selected set is then refit by ordinary least squares
on the original columns. The criterion is

```
BIC(M_k) = log(‖ y − X(M_k) β̂(M_k) ‖²) + (log p / n) · k · log n
```

Natural logarithm, unnormalized RSS. Search is over `k = 1 … n−2`. `d`, `α`
and `k` are chosen by a grid that skips any triple violating
`0 ≤ d < ⌊nα⌋ ≤ r`.
