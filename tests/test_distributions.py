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


class TestLogNormal:
    def test_cdf_known(self):
        # LogNormal CDF(1.0, mu=0, sigma=1) = Normal CDF(0) = 0.5
        df = pl.DataFrame({"x": [1.0]})
        result = df.select(sdist(pl.col("x")).lognormal_cdf(mu=0, sigma=1)).to_series()
        assert abs(result[0] - 0.5) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.5, 1.0, 2.0, 5.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).lognormal_cdf(mu=0, sigma=1)).lognormal_ppf(
                mu=0, sigma=1
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestGamma:
    def test_cdf_known(self):
        # Gamma(shape=1, rate=1) is Exponential(1): CDF(x) = 1 - exp(-x)
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0]})
        result = df.select(sdist(pl.col("x")).gamma_cdf(shape=1.0, rate=1.0)).to_series()
        expected = pl.Series("x", [0.0, 1 - math.exp(-1), 1 - math.exp(-2)])
        assert_series_equal(result, expected, abs_tol=1e-10)

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.5, 1.0, 2.0, 4.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).gamma_cdf(shape=2.0, rate=1.5)).gamma_ppf(
                shape=2.0, rate=1.5
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestCauchy:
    def test_cdf_known(self):
        # Cauchy CDF(0, loc=0, scale=1) = 0.5
        df = pl.DataFrame({"x": [0.0]})
        result = df.select(sdist(pl.col("x")).cauchy_cdf(location=0.0, scale=1.0)).to_series()
        assert abs(result[0] - 0.5) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).cauchy_cdf(location=0.0, scale=1.0)).cauchy_ppf(
                location=0.0, scale=1.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestChiSquared:
    def test_cdf_known(self):
        # ChiSquared(df=2) is Exponential(1/2): CDF(x) = 1 - exp(-x/2)
        df = pl.DataFrame({"x": [2.0]})
        result = df.select(sdist(pl.col("x")).chi_squared_cdf(df=2)).to_series()
        expected = 1 - math.exp(-1)
        assert abs(result[0] - expected) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [1.0, 3.0, 5.0, 10.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).chi_squared_cdf(df=4)).chi_squared_ppf(df=4)
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestFisherSnedecor:
    def test_cdf_known(self):
        # F(d1, d2) median ≈ 1 when d1 = d2 (symmetric case)
        # CDF(1.0) for F(10, 10) ≈ 0.5
        df = pl.DataFrame({"x": [1.0]})
        result = df.select(sdist(pl.col("x")).fisher_snedecor_cdf(d1=10, d2=10)).to_series()
        assert abs(result[0] - 0.5) < 1e-6

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.5, 1.0, 2.0, 3.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).fisher_snedecor_cdf(d1=5, d2=10)).fisher_snedecor_ppf(
                d1=5, d2=10
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestGumbel:
    def test_cdf_known(self):
        # Gumbel CDF(loc) = exp(-exp(0)) = exp(-1) ≈ 0.3679
        location = 2.0
        df = pl.DataFrame({"x": [location]})
        result = df.select(sdist(pl.col("x")).gumbel_cdf(location=location, scale=1.0)).to_series()
        expected = math.exp(-1)
        assert abs(result[0] - expected) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.0, 1.0, 3.0, 5.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).gumbel_cdf(location=1.0, scale=2.0)).gumbel_ppf(
                location=1.0, scale=2.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestInverseGamma:
    def test_cdf_known(self):
        # InverseGamma CDF(x; shape, scale) = Gamma_upper(shape, scale/x) / Gamma(shape)
        # For shape=1, scale=1: CDF(x) = exp(-1/x), so CDF(1) = exp(-1)
        df = pl.DataFrame({"x": [1.0]})
        result = df.select(sdist(pl.col("x")).inverse_gamma_cdf(shape=1.0, scale=1.0)).to_series()
        expected = math.exp(-1)
        assert abs(result[0] - expected) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.5, 1.0, 2.0, 4.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).inverse_gamma_cdf(shape=2.0, scale=1.0)).inverse_gamma_ppf(
                shape=2.0, scale=1.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-4)


class TestLaplace:
    def test_cdf_known(self):
        # Laplace CDF(loc) = 0.5
        df = pl.DataFrame({"x": [0.0]})
        result = df.select(sdist(pl.col("x")).laplace_cdf(location=0.0, scale=1.0)).to_series()
        assert abs(result[0] - 0.5) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [-2.0, -1.0, 0.0, 1.0, 2.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).laplace_cdf(location=0.0, scale=1.0)).laplace_ppf(
                location=0.0, scale=1.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestPareto:
    def test_cdf_known(self):
        # Pareto: shape=x_m (minimum), scale=alpha (tail index)
        # CDF(x) = 1 - (shape/x)^scale for x >= shape
        shape = 2.0  # x_m
        scale = 1.0  # alpha
        x = 5.0
        expected = 1 - (shape / x) ** scale  # 1 - (2/5)^1 = 0.6
        df = pl.DataFrame({"x": [x]})
        result = df.select(sdist(pl.col("x")).pareto_cdf(shape=shape, scale=scale)).to_series()
        assert abs(result[0] - expected) < 1e-10

    def test_roundtrip(self):
        # x values must be >= shape (x_m)
        df = pl.DataFrame({"x": [3.0, 4.0, 6.0, 10.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).pareto_cdf(shape=3.0, scale=2.0)).pareto_ppf(
                shape=3.0, scale=2.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-4)


class TestTriangular:
    def test_cdf_known(self):
        # Symmetric triangular on [0, 2] with mode=1:
        # CDF(1) = 0.5 by symmetry
        df = pl.DataFrame({"x": [1.0]})
        result = df.select(sdist(pl.col("x")).triangular_cdf(min=0.0, max=2.0, mode=1.0)).to_series()
        assert abs(result[0] - 0.5) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.2, 0.5, 1.0, 1.5, 1.8]})
        result = df.select(
            sdist(sdist(pl.col("x")).triangular_cdf(min=0.0, max=2.0, mode=1.0)).triangular_ppf(
                min=0.0, max=2.0, mode=1.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


class TestWeibull:
    def test_cdf_known(self):
        # Weibull CDF(x; shape, scale) = 1 - exp(-(x/scale)^shape)
        shape = 2.0
        scale = 1.0
        x = 1.0
        expected = 1 - math.exp(-(x / scale) ** shape)
        df = pl.DataFrame({"x": [x]})
        result = df.select(sdist(pl.col("x")).weibull_cdf(shape=shape, scale=scale)).to_series()
        assert abs(result[0] - expected) < 1e-10

    def test_roundtrip(self):
        df = pl.DataFrame({"x": [0.5, 1.0, 2.0, 3.0]})
        result = df.select(
            sdist(sdist(pl.col("x")).weibull_cdf(shape=1.5, scale=2.0)).weibull_ppf(
                shape=1.5, scale=2.0
            )
        ).to_series()
        assert_series_equal(result, df["x"], abs_tol=1e-10)


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


class TestGeometric:
    def test_pmf_known(self):
        # Geometric PMF(k=1, p) = p (first trial succeeds)
        p = 0.4
        df = pl.DataFrame({"k": [1.0]})
        result = df.select(sdist(pl.col("k")).geometric_pmf(p=p)).to_series()
        assert abs(result[0] - p) < 1e-10

    def test_cdf_known(self):
        # Geometric CDF(k, p) = 1 - (1-p)^k
        p = 0.3
        k = 3.0
        expected = 1 - (1 - p) ** int(k)
        df = pl.DataFrame({"k": [k]})
        result = df.select(sdist(pl.col("k")).geometric_cdf(p=p)).to_series()
        assert abs(result[0] - expected) < 1e-10


class TestHypergeometric:
    def test_pmf_known(self):
        # Hypergeometric: pop_size=10, success_states=5, draws=4
        # P(X=2) = C(5,2)*C(5,2) / C(10,4) = 10*10/210 = 100/210
        pop_size = 10
        success_states = 5
        draws = 4
        k = 2.0
        expected = (
            math.comb(success_states, 2)
            * math.comb(pop_size - success_states, draws - 2)
            / math.comb(pop_size, draws)
        )
        df = pl.DataFrame({"k": [k]})
        result = df.select(
            sdist(pl.col("k")).hypergeometric_pmf(
                pop_size=pop_size, success_states=success_states, draws=draws
            )
        ).to_series()
        assert abs(result[0] - expected) < 1e-10

    def test_cdf_at_draws(self):
        # CDF(draws) = 1.0 (all draws from successes is the maximum possible)
        pop_size = 20
        success_states = 7
        draws = 5
        df = pl.DataFrame({"k": [float(draws)]})
        result = df.select(
            sdist(pl.col("k")).hypergeometric_cdf(
                pop_size=pop_size, success_states=success_states, draws=draws
            )
        ).to_series()
        assert abs(result[0] - 1.0) < 1e-10


class TestNegativeBinomial:
    def test_pmf_known(self):
        # NegativeBinomial PMF(k=0; r, p) = p^r
        r = 3
        p = 0.6
        expected = p**r
        df = pl.DataFrame({"k": [0.0]})
        result = df.select(sdist(pl.col("k")).negative_binomial_pmf(r=r, p=p)).to_series()
        assert abs(result[0] - expected) < 1e-10

    def test_cdf_known(self):
        # NegativeBinomial CDF(0; r, p) = p^r (only 0 failures possible at k=0)
        r = 2
        p = 0.5
        expected = p**r
        df = pl.DataFrame({"k": [0.0]})
        result = df.select(sdist(pl.col("k")).negative_binomial_cdf(r=r, p=p)).to_series()
        assert abs(result[0] - expected) < 1e-10


class TestDiscreteUniform:
    def test_pmf_known(self):
        # DiscreteUniform PMF = 1/(b - a + 1)
        a = 1
        b = 5
        expected = 1.0 / (b - a + 1)
        df = pl.DataFrame({"k": [1.0, 3.0, 5.0]})
        result = df.select(sdist(pl.col("k")).discrete_uniform_pmf(a=a, b=b)).to_series()
        for v in result.to_list():
            assert abs(v - expected) < 1e-10

    def test_cdf_known(self):
        # DiscreteUniform CDF(b) = 1.0 and CDF(a) = 1/(b-a+1)
        a = 1
        b = 5
        df = pl.DataFrame({"k": [float(a), float(b)]})
        result = df.select(sdist(pl.col("k")).discrete_uniform_cdf(a=a, b=b)).to_series()
        assert abs(result[0] - 1.0 / (b - a + 1)) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10


# ── Null propagation ──


class TestNullPropagation:
    def test_null_in_null_out(self):
        df = pl.DataFrame({"x": [1.0, None, 3.0]})
        result = df.select(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)).to_series()
        assert result[1] is None
        assert result[0] is not None
        assert result[2] is not None
