# polars-sdist

Statistical distributions as native Polars expressions. Built with Rust and PyO3 for zero-copy, vectorized computation over Polars DataFrames.

## Install

```bash
pip install polars-sdist
```

## Quick start

```python
import polars as pl
from polars_sdist import SdistNamespace as sdist

df = pl.DataFrame({"x": [-1.96, 0.0, 1.96]})

# CDF, PDF, PPF, SF, ln_pdf — all as expressions
df.with_columns(
    cdf=sdist(pl.col("x")).normal_cdf(mu=0, sigma=1),
    pdf=sdist(pl.col("x")).normal_pdf(mu=0, sigma=1),
    sf=sdist(pl.col("x")).normal_sf(mu=0, sigma=1),
)
```

### Inverse CDF (PPF)

```python
df = pl.DataFrame({"p": [0.025, 0.5, 0.975]})
df.with_columns(
    z=sdist(pl.col("p")).normal_ppf(mu=0, sigma=1),
)
```

### Discrete distributions

```python
df = pl.DataFrame({"k": [0, 1, 2, 3, 4, 5]}).cast({"k": pl.Float64})
df.with_columns(
    pmf=sdist(pl.col("k")).poisson_pmf(lambda_=3.0),
    cdf=sdist(pl.col("k")).poisson_cdf(lambda_=3.0),
)
```

### Random sampling (fixed parameters)

```python
import polars_sdist

s = polars_sdist.sample_normal(n=10_000, mu=0, sigma=1, seed=42)
s = polars_sdist.sample_beta(n=10_000, alpha=2, beta=5, seed=42)
```

### Random sampling (column parameters)

```python
df = pl.DataFrame({
    "mu": [0.0, 10.0, -10.0],
    "sigma": [1.0, 0.5, 2.0],
})
df.with_columns(
    sample=sdist(pl.col("mu")).sample_normal(sigma=pl.col("sigma"), seed=42),
)
```

## Supported distributions

### Continuous

| Distribution | Parameters | PDF | CDF | PPF | SF | ln_pdf |
|-|-|-|-|-|-|-|
| Normal | `mu`, `sigma` | x | x | x | x | x |
| Log-Normal | `mu`, `sigma` | x | x | x | x | x |
| Beta | `alpha`, `beta` | x | x | x | x | x |
| Gamma | `shape`, `rate` | x | x | x | x | x |
| Exponential | `lambda_` | x | x | x | x | x |
| Cauchy | `location`, `scale` | x | x | x | x | x |
| Chi-Squared | `df` | x | x | x | x | x |
| Student's t | `df` | x | x | x | x | x |
| Fisher-Snedecor (F) | `d1`, `d2` | x | x | x | x | x |
| Gumbel | `location`, `scale` | x | x | x | x | x |
| Inverse Gamma | `shape`, `scale` | x | x | x | x | x |
| Laplace | `location`, `scale` | x | x | x | x | x |
| Pareto | `shape`, `scale` | x | x | x | x | x |
| Triangular | `min`, `max`, `mode` | x | x | x | x | x |
| Uniform | `a`, `b` | x | x | x | x | x |
| Weibull | `shape`, `scale` | x | x | x | x | x |

### Discrete

| Distribution | Parameters | PMF | CDF | PPF | SF |
|-|-|-|-|-|-|
| Bernoulli | `p` | x | x | x | x |
| Binomial | `n`, `p` | x | x | x | x |
| Geometric | `p` | x | x | x | x |
| Hypergeometric | `pop_size`, `success_states`, `draws` | x | x | x | x |
| Negative Binomial | `r`, `p` | x | x | x | x |
| Poisson | `lambda_` | x | x | x | x |
| Discrete Uniform | `a`, `b` | x | x | x | x |

### Sampling-only

These distributions support random sampling but not PDF/CDF:

PERT, Skew-Normal, Inverse Gaussian, Frechet, Zeta, Zipf

## License

MIT
