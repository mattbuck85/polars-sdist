from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_series_equal

from polars_sdist import SdistNamespace as sdist


# ── Continuous: known values (scipy reference) ──


class TestNormal:
    def test_cdf_known(self):
        df = pl.DataFrame({"x": [0.0, 1.96, -1.96]})
        result = df.select(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)).to_series()
        expected = pl.Series("x", [0.5, 0.975002, 0.024998])
        assert_series_equal(result, expected, abs_tol=1e-4)

    def test_ppf_known(self):
        df = pl.DataFrame({"p": [0.025, 0.5, 0.975]})
        result = df.select(sdist(pl.col("p")).normal_ppf(mu=0, sigma=1)).to_series()
        expected = pl.Series("p", [-1.96, 0.0, 1.96])
        assert_series_equal(result, expected, abs_tol=1e-2)

    def test_roundtrip(self):
        """ppf(cdf(x)) ≈ x"""
        df = pl.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)).normal_ppf(
                mu=0, sigma=1
            )
        ).to_series()
        expected = df["x"]
        assert_series_equal(result, expected, abs_tol=1e-10)

    def test_pdf_at_zero(self):
        df = pl.DataFrame({"x": [0.0]})
        result = df.select(sdist(pl.col("x")).normal_pdf(mu=0, sigma=1)).to_series()
        expected_val = 1.0 / math.sqrt(2 * math.pi)
        assert abs(result[0] - expected_val) < 1e-10

    def test_sf_complement(self):
        """sf(x) = 1 - cdf(x)"""
        df = pl.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0]})
        cdf = df.select(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)).to_series()
        sf = df.select(sdist(pl.col("x")).normal_sf(mu=0, sigma=1)).to_series()
        complement = (cdf + sf).to_list()
        for v in complement:
            assert abs(v - 1.0) < 1e-12

    def test_ln_pdf(self):
        df = pl.DataFrame({"x": [0.0, 1.0]})
        pdf = df.select(sdist(pl.col("x")).normal_pdf(mu=0, sigma=1)).to_series()
        ln_pdf = df.select(sdist(pl.col("x")).normal_ln_pdf(mu=0, sigma=1)).to_series()
        for i in range(len(pdf)):
            assert abs(math.log(pdf[i]) - ln_pdf[i]) < 1e-12


class TestExponential:
    def test_cdf_known(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
        result = df.select(sdist(pl.col("x")).exponential_cdf(lambda_=1.0)).to_series()
        expected = pl.Series("x", [0.0, 1 - math.exp(-1), 1 - math.exp(-2)])
        assert_series_equal(result, expected, abs_tol=1e-10)

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.1, 0.5, 1.0, 5.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).exponential_cdf(lambda_=2.0)).exponential_ppf(
                lambda_=2.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestBeta:
    def test_cdf_endpoints(self):
        df = pl.DataFrame({"x": [0.0, 1.0]})
        result = df.select(sdist(pl.col("x")).beta_cdf(alpha=2.0, beta=5.0)).to_series()
        assert abs(result[0] - 0.0) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.1, 0.3, 0.5, 0.7, 0.9]})
        result = df.select(
            sdist(sdist(pl.col("x")).beta_cdf(alpha=2, beta=5)).beta_ppf(
                alpha=2, beta=5
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestStudentsT:
    def test_cdf_symmetry(self):
        df = pl.DataFrame({"x": [-2.0, 2.0]})
        result = df.select(sdist(pl.col("x")).students_t_cdf(df=10)).to_series()
        assert abs(result[0] + result[1] - 1.0) < 1e-10

    def test_sf(self):
        df = pl.DataFrame({"x": [2.0]})
        sf = df.select(sdist(pl.col("x")).students_t_sf(df=30)).to_series()
        # For df=30, t=2.0 => p-value ≈ 0.0273 (one-tailed)
        assert 0.02 < sf[0] < 0.04


class TestUniform:
    def test_cdf(self):
        df = pl.DataFrame({"x": [0.0, 0.5, 1.0]})
        result = df.select(sdist(pl.col("x")).uniform_cdf(a=0, b=1)).to_series()
        expected = pl.Series("x", [0.0, 0.5, 1.0])
        assert_series_equal(result, expected, abs_tol=1e-12)


# ── Discrete distributions ──


class TestPoisson:
    def test_pmf_known(self):
        df = pl.DataFrame({"k": [0.0, 1.0, 2.0, 3.0]})
        result = df.select(sdist(pl.col("k")).poisson_pmf(lambda_=2.0)).to_series()
        # P(X=0) = e^{-2}, P(X=1) = 2e^{-2}, P(X=2) = 2e^{-2}, P(X=3) = 4/3 e^{-2}
        e2 = math.exp(-2)
        expected = pl.Series("k", [e2, 2 * e2, 2 * e2, 4.0 / 3.0 * e2])
        assert_series_equal(result, expected, abs_tol=1e-10)


class TestBinomial:
    def test_pmf_known(self):
        df = pl.DataFrame({"k": [0.0, 5.0, 10.0]})
        result = df.select(sdist(pl.col("k")).binomial_pmf(n=10, p=0.5)).to_series()
        # P(X=0) = 0.5^10, P(X=5) = C(10,5)*0.5^10, P(X=10) = 0.5^10
        assert abs(result[0] - 0.5**10) < 1e-10
        assert abs(result[2] - 0.5**10) < 1e-10

    def test_cdf_at_n(self):
        df = pl.DataFrame({"k": [10.0]})
        result = df.select(sdist(pl.col("k")).binomial_cdf(n=10, p=0.3)).to_series()
        assert abs(result[0] - 1.0) < 1e-10


class TestBernoulli:
    def test_pmf(self):
        df = pl.DataFrame({"k": [0.0, 1.0]})
        result = df.select(sdist(pl.col("k")).bernoulli_pmf(p=0.7)).to_series()
        assert abs(result[0] - 0.3) < 1e-10
        assert abs(result[1] - 0.7) < 1e-10


# ── Null propagation ──


class TestNullPropagation:
    def test_null_in_null_out(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0]})
        result = df.select(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)).to_series()
        assert result[1] is None
        assert result[0] is not None
        assert result[2] is not None
