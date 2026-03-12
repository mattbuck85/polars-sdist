from __future__ import annotations

import polars as pl

from polars_sdist import SdistNamespace as sdist
import polars_sdist


class TestFixedParamSampling:
    def test_normal_deterministic(self):
        s1 = polars_sdist.sample_normal(n=100, mu=0, sigma=1, seed=42)
        s2 = polars_sdist.sample_normal(n=100, mu=0, sigma=1, seed=42)
        assert s1.to_list() == s2.to_list()

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
