from __future__ import annotations

import polars as pl
import pytest

from polars_sdist import SdistNamespace as sdist
import polars_sdist


class TestFixedParamSampling:
    def test_normal_deterministic(self):
        s1 = polars_sdist.sample_normal(n=100, mu=0, sigma=1, seed=42)
        s2 = polars_sdist.sample_normal(n=100, mu=0, sigma=1, seed=42)
        assert s1.to_list() == s2.to_list()

    @pytest.mark.parametrize("sample_fn, kwargs", [
        (polars_sdist.sample_normal, {"mu": 0, "sigma": 1}),
        (polars_sdist.sample_lognormal, {"mu": 0, "sigma": 1}),
        (polars_sdist.sample_beta, {"alpha": 2, "beta": 5}),
        (polars_sdist.sample_chi_squared, {"df": 5}),
        (polars_sdist.sample_binomial, {"trials": 10, "p": 0.5}),
        (polars_sdist.sample_poisson, {"lambda_": 3.0}),
        (polars_sdist.sample_exponential, {"lambda_": 1.0}),
        (polars_sdist.sample_gamma, {"shape": 2, "rate": 1}),
        (polars_sdist.sample_uniform, {"a": 0, "b": 1}),
        (polars_sdist.sample_students_t, {"df": 10}),
        (polars_sdist.sample_cauchy, {"location": 0, "scale": 1}),
        (polars_sdist.sample_laplace, {"location": 0, "scale": 1}),
        (polars_sdist.sample_bernoulli, {"p": 0.5}),
        (polars_sdist.sample_geometric, {"p": 0.3}),
        (polars_sdist.sample_weibull, {"shape": 2, "scale": 1}),
    ])
    def test_seeded_determinism(self, sample_fn, kwargs):
        s1 = sample_fn(n=500, **kwargs, seed=99)
        s2 = sample_fn(n=500, **kwargs, seed=99)
        assert s1.to_list() == s2.to_list()
        # Different seed produces different output
        s3 = sample_fn(n=500, **kwargs, seed=100)
        assert s1.to_list() != s3.to_list()

    def test_normal_stats(self):
        s = polars_sdist.sample_normal(n=50_000, mu=5.0, sigma=2.0, seed=123)
        assert abs(s.mean() - 5.0) < 0.1
        assert abs(s.std() - 2.0) < 0.1

    def test_uniform_range(self):
        s = polars_sdist.sample_uniform(n=10_000, a=3.0, b=7.0, seed=1)
        assert s.min() >= 3.0
        assert s.max() <= 7.0

    def test_exponential_positive(self):
        s = polars_sdist.sample_exponential(n=1000, lambda_=1.0, seed=1)
        assert s.min() > 0.0

    def test_bernoulli_values(self):
        s = polars_sdist.sample_bernoulli(n=10_000, p=0.5, seed=42)
        vals = set(s.to_list())
        assert vals <= {0.0, 1.0}
        mean = s.mean()
        assert 0.45 < mean < 0.55

    def test_poisson_mean(self):
        s = polars_sdist.sample_poisson(n=50_000, lambda_=3.0, seed=99)
        assert abs(s.mean() - 3.0) < 0.1

    def test_beta_range(self):
        s = polars_sdist.sample_beta(n=10_000, alpha=2.0, beta=5.0, seed=7)
        assert s.min() >= 0.0
        assert s.max() <= 1.0

    def test_students_t_symmetric(self):
        s = polars_sdist.sample_students_t(n=50_000, df=10.0, seed=55)
        assert abs(s.mean()) < 0.1

    def test_length(self):
        s = polars_sdist.sample_normal(n=42, seed=1)
        assert len(s) == 42

    # --- lognormal ---

    def test_lognormal_positive(self):
        s = polars_sdist.sample_lognormal(n=10_000, mu=0.0, sigma=0.5, seed=1)
        assert s.min() > 0.0

    def test_lognormal_mean(self):
        import math
        s = polars_sdist.sample_lognormal(n=50_000, mu=0.0, sigma=0.5, seed=2)
        expected_mean = math.exp(0.0 + 0.25 / 2)  # exp(mu + sigma^2/2)
        assert abs(s.mean() - expected_mean) < 0.05

    # --- gamma ---

    def test_gamma_positive(self):
        s = polars_sdist.sample_gamma(n=10_000, shape=2.0, rate=1.0, seed=1)
        assert s.min() > 0.0

    def test_gamma_mean(self):
        s = polars_sdist.sample_gamma(n=50_000, shape=2.0, rate=1.0, seed=2)
        assert abs(s.mean() - 2.0) < 0.1

    # --- cauchy ---

    def test_cauchy_median(self):
        s = polars_sdist.sample_cauchy(n=50_000, location=3.0, scale=1.0, seed=1)
        median = s.median()
        assert abs(median - 3.0) < 0.1

    # --- chi_squared ---

    def test_chi_squared_positive(self):
        s = polars_sdist.sample_chi_squared(n=10_000, df=5.0, seed=1)
        assert s.min() > 0.0

    def test_chi_squared_mean(self):
        s = polars_sdist.sample_chi_squared(n=50_000, df=5.0, seed=2)
        assert abs(s.mean() - 5.0) < 0.15

    # --- fisher_snedecor ---

    def test_fisher_snedecor_positive(self):
        s = polars_sdist.sample_fisher_snedecor(n=10_000, d1=10.0, d2=20.0, seed=1)
        assert s.min() > 0.0

    def test_fisher_snedecor_mean(self):
        s = polars_sdist.sample_fisher_snedecor(n=50_000, d1=10.0, d2=20.0, seed=2)
        expected_mean = 20.0 / (20.0 - 2.0)  # d2 / (d2 - 2)
        assert abs(s.mean() - expected_mean) < 0.05

    # --- gumbel ---

    def test_gumbel_mean(self):
        s = polars_sdist.sample_gumbel(n=50_000, location=0.0, scale=1.0, seed=1)
        euler_mascheroni = 0.5772156649
        assert abs(s.mean() - euler_mascheroni) < 0.05

    # --- inverse_gamma ---

    def test_inverse_gamma_positive(self):
        s = polars_sdist.sample_inverse_gamma(n=10_000, shape=5.0, scale=1.0, seed=1)
        assert s.min() > 0.0

    def test_inverse_gamma_mean(self):
        s = polars_sdist.sample_inverse_gamma(n=50_000, shape=5.0, scale=1.0, seed=2)
        expected_mean = 1.0 / (5.0 - 1.0)  # scale / (shape - 1)
        assert abs(s.mean() - expected_mean) < 0.02

    # --- laplace ---

    def test_laplace_mean(self):
        s = polars_sdist.sample_laplace(n=50_000, location=0.0, scale=1.0, seed=1)
        assert abs(s.mean() - 0.0) < 0.05

    def test_laplace_symmetric(self):
        s = polars_sdist.sample_laplace(n=50_000, location=0.0, scale=1.0, seed=2)
        pos_count = (s > 0).sum()
        neg_count = (s < 0).sum()
        ratio = pos_count / neg_count
        assert 0.95 < ratio < 1.05

    # --- pareto ---

    def test_pareto_range(self):
        s = polars_sdist.sample_pareto(n=10_000, shape=3.0, scale=1.0, seed=1)
        assert s.min() >= 1.0

    def test_pareto_positive(self):
        s = polars_sdist.sample_pareto(n=10_000, shape=3.0, scale=1.0, seed=2)
        assert s.min() > 0.0

    # --- triangular ---

    def test_triangular_range(self):
        s = polars_sdist.sample_triangular(n=10_000, min=0.0, max=10.0, mode=5.0, seed=1)
        assert s.min() >= 0.0
        assert s.max() <= 10.0

    def test_triangular_mean(self):
        s = polars_sdist.sample_triangular(n=50_000, min=0.0, max=10.0, mode=5.0, seed=2)
        expected_mean = (0.0 + 10.0 + 5.0) / 3.0
        assert abs(s.mean() - expected_mean) < 0.1

    # --- weibull ---

    def test_weibull_positive(self):
        s = polars_sdist.sample_weibull(n=10_000, shape=2.0, scale=1.0, seed=1)
        assert s.min() > 0.0

    def test_weibull_mean(self):
        import math
        s = polars_sdist.sample_weibull(n=50_000, shape=2.0, scale=1.0, seed=2)
        expected_mean = math.sqrt(math.pi) / 2.0  # scale * Gamma(1 + 1/shape) = Gamma(1.5) = sqrt(pi)/2
        assert abs(s.mean() - expected_mean) < 0.05

    # --- pert ---

    def test_pert_range(self):
        s = polars_sdist.sample_pert(n=10_000, min=0.0, max=10.0, mode=5.0, seed=1)
        assert s.min() >= 0.0
        assert s.max() <= 10.0

    def test_pert_mean(self):
        s = polars_sdist.sample_pert(n=50_000, min=0.0, max=10.0, mode=5.0, seed=2)
        expected_mean = (0.0 + 4.0 * 5.0 + 10.0) / 6.0
        assert abs(s.mean() - expected_mean) < 0.1

    # --- skew_normal ---

    def test_skew_normal_mean_zero_shape(self):
        s = polars_sdist.sample_skew_normal(n=50_000, location=0.0, scale=1.0, shape=0.0, seed=1)
        assert abs(s.mean() - 0.0) < 0.05

    # --- inverse_gaussian ---

    def test_inverse_gaussian_positive(self):
        s = polars_sdist.sample_inverse_gaussian(n=10_000, mean=1.0, shape=1.0, seed=1)
        assert s.min() > 0.0

    def test_inverse_gaussian_mean(self):
        s = polars_sdist.sample_inverse_gaussian(n=50_000, mean=1.0, shape=1.0, seed=2)
        assert abs(s.mean() - 1.0) < 0.05

    # --- frechet ---

    def test_frechet_positive(self):
        s = polars_sdist.sample_frechet(n=10_000, location=0.0, scale=1.0, shape=2.0, seed=1)
        assert s.min() >= 0.0

    # --- zeta ---

    def test_zeta_positive_integers(self):
        s = polars_sdist.sample_zeta(n=10_000, s=2.5, seed=1)
        assert s.min() >= 1.0
        assert (s == s.cast(pl.Int64).cast(s.dtype)).all()

    # --- zipf ---

    def test_zipf_range(self):
        s = polars_sdist.sample_zipf(n=10_000, n_elements=100, s=1.5, seed=1)
        assert s.min() >= 1.0
        assert s.max() <= 100.0

    # --- binomial ---

    def test_binomial_range(self):
        s = polars_sdist.sample_binomial(n=10_000, trials=10, p=0.5, seed=1)
        assert s.min() >= 0.0
        assert s.max() <= 10.0

    def test_binomial_mean(self):
        s = polars_sdist.sample_binomial(n=50_000, trials=10, p=0.5, seed=2)
        assert abs(s.mean() - 5.0) < 0.1

    # --- geometric ---

    def test_geometric_non_negative(self):
        # statrs geometric is 0-based (number of failures before first success)
        s = polars_sdist.sample_geometric(n=10_000, p=0.3, seed=1)
        assert s.min() >= 0.0

    def test_geometric_mean(self):
        s = polars_sdist.sample_geometric(n=50_000, p=0.3, seed=2)
        # 0-based geometric: mean = (1-p)/p
        expected_mean = (1.0 - 0.3) / 0.3
        assert abs(s.mean() - expected_mean) < 0.15


class TestColumnParamSampling:
    def test_normal_col(self):
        df = pl.DataFrame({"mu": [0.0, 10.0, -10.0], "sigma": [0.001, 0.001, 0.001]})
        result = df.with_columns(
            sdist(pl.col("mu")).sample_normal(sigma=pl.col("sigma"), seed=42)
        )
        samples = result["mu"].to_list()
        assert abs(samples[0] - 0.0) < 0.1
        assert abs(samples[1] - 10.0) < 0.1
        assert abs(samples[2] - (-10.0)) < 0.1

    def test_lognormal_col_sigma(self):
        df = pl.DataFrame({"mu": [0.0, 0.0, 0.0], "sigma": [0.1, 0.5, 1.0]})
        result = df.with_columns(
            sdist(pl.col("mu")).sample_lognormal(sigma=pl.col("sigma"), seed=7)
        )
        samples = result["mu"].to_list()
        assert all(v > 0.0 for v in samples)

    def test_beta_col_beta(self):
        df = pl.DataFrame({"alpha": [2.0, 5.0, 1.0], "beta": [5.0, 2.0, 1.0]})
        result = df.with_columns(
            sdist(pl.col("alpha")).sample_beta(beta=pl.col("beta"), seed=13)
        )
        samples = result["alpha"].to_list()
        assert all(0.0 <= v <= 1.0 for v in samples)

    def test_gamma_col_rate(self):
        df = pl.DataFrame({"shape": [2.0, 3.0, 5.0], "rate": [1.0, 2.0, 0.5]})
        result = df.with_columns(
            sdist(pl.col("shape")).sample_gamma(rate=pl.col("rate"), seed=17)
        )
        samples = result["shape"].to_list()
        assert all(v > 0.0 for v in samples)

    def test_uniform_col_b(self):
        df = pl.DataFrame({"a": [0.0, 5.0, -1.0], "b": [1.0, 10.0, 1.0]})
        result = df.with_columns(
            sdist(pl.col("a")).sample_uniform(b=pl.col("b"), seed=23)
        )
        samples = result["a"].to_list()
        assert samples[0] >= 0.0 and samples[0] <= 1.0
        assert samples[1] >= 5.0 and samples[1] <= 10.0
        assert samples[2] >= -1.0 and samples[2] <= 1.0
