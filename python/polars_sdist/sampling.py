from __future__ import annotations

import polars as pl

from polars_sdist._polars_sdist import sample_direct as _sample_rs


def _sample(
    dist: str,
    n: int,
    param1: float,
    param2: float | None = None,
    param3: float | None = None,
    seed: int | None = None,
) -> pl.Series:
    """Generate n samples from the given distribution with fixed parameters."""
    return _sample_rs(dist, n, param1, param2, param3, seed)


def sample_normal(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("normal", n, mu, sigma, seed=seed)


def sample_lognormal(n: int, mu: float = 0.0, sigma: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("lognormal", n, mu, sigma, seed=seed)


def sample_beta(n: int, alpha: float, beta: float, seed: int | None = None) -> pl.Series:
    return _sample("beta", n, alpha, beta, seed=seed)


def sample_gamma(n: int, shape: float, rate: float, seed: int | None = None) -> pl.Series:
    return _sample("gamma", n, shape, rate, seed=seed)


def sample_cauchy(n: int, location: float = 0.0, scale: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("cauchy", n, location, scale, seed=seed)


def sample_chi_squared(n: int, df: float, seed: int | None = None) -> pl.Series:
    return _sample("chi_squared", n, df, seed=seed)


def sample_exponential(n: int, lambda_: float, seed: int | None = None) -> pl.Series:
    return _sample("exponential", n, lambda_, seed=seed)


def sample_fisher_snedecor(n: int, d1: float, d2: float, seed: int | None = None) -> pl.Series:
    return _sample("fisher_snedecor", n, d1, d2, seed=seed)


def sample_gumbel(n: int, location: float = 0.0, scale: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("gumbel", n, location, scale, seed=seed)


def sample_inverse_gamma(n: int, shape: float, scale: float, seed: int | None = None) -> pl.Series:
    return _sample("inverse_gamma", n, shape, scale, seed=seed)


def sample_laplace(n: int, location: float = 0.0, scale: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("laplace", n, location, scale, seed=seed)


def sample_pareto(n: int, shape: float, scale: float, seed: int | None = None) -> pl.Series:
    return _sample("pareto", n, shape, scale, seed=seed)


def sample_students_t(n: int, df: float, seed: int | None = None) -> pl.Series:
    return _sample("students_t", n, df, seed=seed)


def sample_triangular(n: int, min: float, max: float, mode: float, seed: int | None = None) -> pl.Series:
    return _sample("triangular", n, min, max, mode, seed=seed)


def sample_uniform(n: int, a: float = 0.0, b: float = 1.0, seed: int | None = None) -> pl.Series:
    return _sample("uniform", n, a, b, seed=seed)


def sample_weibull(n: int, shape: float, scale: float, seed: int | None = None) -> pl.Series:
    return _sample("weibull", n, shape, scale, seed=seed)


def sample_pert(n: int, min: float, max: float, mode: float, seed: int | None = None) -> pl.Series:
    return _sample("pert", n, min, max, mode, seed=seed)


def sample_skew_normal(n: int, location: float, scale: float, shape: float, seed: int | None = None) -> pl.Series:
    return _sample("skew_normal", n, location, scale, shape, seed=seed)


def sample_inverse_gaussian(n: int, mean: float, shape: float, seed: int | None = None) -> pl.Series:
    return _sample("inverse_gaussian", n, mean, shape, seed=seed)


def sample_frechet(n: int, location: float, scale: float, shape: float, seed: int | None = None) -> pl.Series:
    return _sample("frechet", n, location, scale, shape, seed=seed)


def sample_zeta(n: int, s: float, seed: int | None = None) -> pl.Series:
    return _sample("zeta", n, s, seed=seed)


def sample_zipf(n: int, n_elements: int, s: float, seed: int | None = None) -> pl.Series:
    return _sample("zipf", n, float(n_elements), s, seed=seed)


def sample_bernoulli(n: int, p: float, seed: int | None = None) -> pl.Series:
    return _sample("bernoulli", n, p, seed=seed)


def sample_binomial(n: int, trials: int, p: float, seed: int | None = None) -> pl.Series:
    return _sample("binomial", n, float(trials), p, seed=seed)


def sample_geometric(n: int, p: float, seed: int | None = None) -> pl.Series:
    return _sample("geometric", n, p, seed=seed)


def sample_poisson(n: int, lambda_: float, seed: int | None = None) -> pl.Series:
    return _sample("poisson", n, lambda_, seed=seed)
