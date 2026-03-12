from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from polars_sdist._utils import pl_plugin

if TYPE_CHECKING:
    pass


def _dist_kwargs(
    dist: str,
    param1: float,
    param2: float | None = None,
    param3: float | None = None,
) -> dict:
    return {"dist": dist, "param1": param1, "param2": param2, "param3": param3}


def _discrete_kwargs(
    dist: str,
    param1: float,
    param2: float | None = None,
    param3: float | None = None,
) -> dict:
    return {"dist": dist, "param1": param1, "param2": param2, "param3": param3}


def _sample_col_kwargs(
    dist: str,
    seed: int | None = None,
) -> dict:
    return {"dist": dist, "param1": 0.0, "param2": None, "param3": None, "seed": seed}


@pl.api.register_expr_namespace("sdist")
class SdistNamespace:
    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    # ── Continuous: PDF ──

    def normal_pdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("normal", mu, sigma))

    def lognormal_pdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("lognormal", mu, sigma))

    def beta_pdf(self, alpha: float, beta: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("beta", alpha, beta))

    def gamma_pdf(self, shape: float, rate: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("gamma", shape, rate))

    def cauchy_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("cauchy", location, scale))

    def chi_squared_pdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("chi_squared", df))

    def exponential_pdf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("exponential", lambda_))

    def fisher_snedecor_pdf(self, d1: float, d2: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("fisher_snedecor", d1, d2))

    def gumbel_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("gumbel", location, scale))

    def inverse_gamma_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("inverse_gamma", shape, scale))

    def laplace_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("laplace", location, scale))

    def pareto_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("pareto", shape, scale))

    def students_t_pdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("students_t", df))

    def triangular_pdf(self, min: float, max: float, mode: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("triangular", min, max, mode))

    def uniform_pdf(self, a: float = 0.0, b: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("uniform", a, b))

    def weibull_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_pdf", args=[self._expr], kwargs=_dist_kwargs("weibull", shape, scale))

    # ── Continuous: CDF ──

    def normal_cdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("normal", mu, sigma))

    def lognormal_cdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("lognormal", mu, sigma))

    def beta_cdf(self, alpha: float, beta: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("beta", alpha, beta))

    def gamma_cdf(self, shape: float, rate: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("gamma", shape, rate))

    def cauchy_cdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("cauchy", location, scale))

    def chi_squared_cdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("chi_squared", df))

    def exponential_cdf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("exponential", lambda_))

    def fisher_snedecor_cdf(self, d1: float, d2: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("fisher_snedecor", d1, d2))

    def gumbel_cdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("gumbel", location, scale))

    def inverse_gamma_cdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("inverse_gamma", shape, scale))

    def laplace_cdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("laplace", location, scale))

    def pareto_cdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("pareto", shape, scale))

    def students_t_cdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("students_t", df))

    def triangular_cdf(self, min: float, max: float, mode: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("triangular", min, max, mode))

    def uniform_cdf(self, a: float = 0.0, b: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("uniform", a, b))

    def weibull_cdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_cdf", args=[self._expr], kwargs=_dist_kwargs("weibull", shape, scale))

    # ── Continuous: PPF (inverse CDF) ──

    def normal_ppf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("normal", mu, sigma))

    def lognormal_ppf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("lognormal", mu, sigma))

    def beta_ppf(self, alpha: float, beta: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("beta", alpha, beta))

    def gamma_ppf(self, shape: float, rate: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("gamma", shape, rate))

    def cauchy_ppf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("cauchy", location, scale))

    def chi_squared_ppf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("chi_squared", df))

    def exponential_ppf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("exponential", lambda_))

    def fisher_snedecor_ppf(self, d1: float, d2: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("fisher_snedecor", d1, d2))

    def gumbel_ppf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("gumbel", location, scale))

    def inverse_gamma_ppf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("inverse_gamma", shape, scale))

    def laplace_ppf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("laplace", location, scale))

    def pareto_ppf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("pareto", shape, scale))

    def students_t_ppf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("students_t", df))

    def triangular_ppf(self, min: float, max: float, mode: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("triangular", min, max, mode))

    def uniform_ppf(self, a: float = 0.0, b: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("uniform", a, b))

    def weibull_ppf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ppf", args=[self._expr], kwargs=_dist_kwargs("weibull", shape, scale))

    # ── Continuous: SF (survival function = 1 - CDF) ──

    def normal_sf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("normal", mu, sigma))

    def lognormal_sf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("lognormal", mu, sigma))

    def beta_sf(self, alpha: float, beta: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("beta", alpha, beta))

    def gamma_sf(self, shape: float, rate: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("gamma", shape, rate))

    def cauchy_sf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("cauchy", location, scale))

    def chi_squared_sf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("chi_squared", df))

    def exponential_sf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("exponential", lambda_))

    def fisher_snedecor_sf(self, d1: float, d2: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("fisher_snedecor", d1, d2))

    def gumbel_sf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("gumbel", location, scale))

    def inverse_gamma_sf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("inverse_gamma", shape, scale))

    def laplace_sf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("laplace", location, scale))

    def pareto_sf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("pareto", shape, scale))

    def students_t_sf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("students_t", df))

    def triangular_sf(self, min: float, max: float, mode: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("triangular", min, max, mode))

    def uniform_sf(self, a: float = 0.0, b: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("uniform", a, b))

    def weibull_sf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_sf", args=[self._expr], kwargs=_dist_kwargs("weibull", shape, scale))

    # ── Continuous: ln_pdf ──

    def normal_ln_pdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("normal", mu, sigma))

    def lognormal_ln_pdf(self, mu: float = 0.0, sigma: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("lognormal", mu, sigma))

    def beta_ln_pdf(self, alpha: float, beta: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("beta", alpha, beta))

    def gamma_ln_pdf(self, shape: float, rate: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("gamma", shape, rate))

    def cauchy_ln_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("cauchy", location, scale))

    def chi_squared_ln_pdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("chi_squared", df))

    def exponential_ln_pdf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("exponential", lambda_))

    def fisher_snedecor_ln_pdf(self, d1: float, d2: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("fisher_snedecor", d1, d2))

    def gumbel_ln_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("gumbel", location, scale))

    def inverse_gamma_ln_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("inverse_gamma", shape, scale))

    def laplace_ln_pdf(self, location: float = 0.0, scale: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("laplace", location, scale))

    def pareto_ln_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("pareto", shape, scale))

    def students_t_ln_pdf(self, df: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("students_t", df))

    def triangular_ln_pdf(self, min: float, max: float, mode: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("triangular", min, max, mode))

    def uniform_ln_pdf(self, a: float = 0.0, b: float = 1.0) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("uniform", a, b))

    def weibull_ln_pdf(self, shape: float, scale: float) -> pl.Expr:
        return pl_plugin(symbol="dist_ln_pdf", args=[self._expr], kwargs=_dist_kwargs("weibull", shape, scale))

    # ── Discrete: PMF ──

    def bernoulli_pmf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("bernoulli", p))

    def binomial_pmf(self, n: int, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("binomial", float(n), p))

    def geometric_pmf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("geometric", p))

    def hypergeometric_pmf(self, pop_size: int, success_states: int, draws: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("hypergeometric", float(pop_size), float(success_states), float(draws)))

    def negative_binomial_pmf(self, r: float, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("negative_binomial", r, p))

    def poisson_pmf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("poisson", lambda_))

    def discrete_uniform_pmf(self, a: int, b: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_pmf", args=[self._expr], kwargs=_discrete_kwargs("discrete_uniform", float(a), float(b)))

    # ── Discrete: CDF ──

    def bernoulli_cdf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("bernoulli", p))

    def binomial_cdf(self, n: int, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("binomial", float(n), p))

    def geometric_cdf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("geometric", p))

    def hypergeometric_cdf(self, pop_size: int, success_states: int, draws: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("hypergeometric", float(pop_size), float(success_states), float(draws)))

    def negative_binomial_cdf(self, r: float, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("negative_binomial", r, p))

    def poisson_cdf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("poisson", lambda_))

    def discrete_uniform_cdf(self, a: int, b: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_cdf", args=[self._expr], kwargs=_discrete_kwargs("discrete_uniform", float(a), float(b)))

    # ── Discrete: PPF ──

    def bernoulli_ppf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("bernoulli", p))

    def binomial_ppf(self, n: int, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("binomial", float(n), p))

    def geometric_ppf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("geometric", p))

    def hypergeometric_ppf(self, pop_size: int, success_states: int, draws: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("hypergeometric", float(pop_size), float(success_states), float(draws)))

    def negative_binomial_ppf(self, r: float, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("negative_binomial", r, p))

    def poisson_ppf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("poisson", lambda_))

    def discrete_uniform_ppf(self, a: int, b: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_ppf", args=[self._expr], kwargs=_discrete_kwargs("discrete_uniform", float(a), float(b)))

    # ── Discrete: SF ──

    def bernoulli_sf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("bernoulli", p))

    def binomial_sf(self, n: int, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("binomial", float(n), p))

    def geometric_sf(self, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("geometric", p))

    def hypergeometric_sf(self, pop_size: int, success_states: int, draws: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("hypergeometric", float(pop_size), float(success_states), float(draws)))

    def negative_binomial_sf(self, r: float, p: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("negative_binomial", r, p))

    def poisson_sf(self, lambda_: float) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("poisson", lambda_))

    def discrete_uniform_sf(self, a: int, b: int) -> pl.Expr:
        return pl_plugin(symbol="discrete_sf", args=[self._expr], kwargs=_discrete_kwargs("discrete_uniform", float(a), float(b)))

    # ── Column-parameterized sampling ──

    def sample_normal(self, sigma: pl.Expr, seed: int | None = None) -> pl.Expr:
        return pl_plugin(symbol="dist_sample_col", args=[self._expr, sigma], kwargs=_sample_col_kwargs("normal", seed))

    def sample_lognormal(self, sigma: pl.Expr, seed: int | None = None) -> pl.Expr:
        return pl_plugin(symbol="dist_sample_col", args=[self._expr, sigma], kwargs=_sample_col_kwargs("lognormal", seed))

    def sample_beta(self, beta: pl.Expr, seed: int | None = None) -> pl.Expr:
        return pl_plugin(symbol="dist_sample_col", args=[self._expr, beta], kwargs=_sample_col_kwargs("beta", seed))

    def sample_gamma(self, rate: pl.Expr, seed: int | None = None) -> pl.Expr:
        return pl_plugin(symbol="dist_sample_col", args=[self._expr, rate], kwargs=_sample_col_kwargs("gamma", seed))

    def sample_uniform(self, b: pl.Expr, seed: int | None = None) -> pl.Expr:
        return pl_plugin(symbol="dist_sample_col", args=[self._expr, b], kwargs=_sample_col_kwargs("uniform", seed))
