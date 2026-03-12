use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use rand::prelude::*;

use super::DistKwargs;
use crate::utils::make_rng;

/// Macro to construct a rand_distr distribution and sample N times into Vec<f64>.
macro_rules! dispatch_sample {
    ($name:expr, $p1:expr, $p2:expr, $p3:expr, $n:expr, $rng:expr) => {{
        macro_rules! err {
            ($dist_name:expr, $e:expr) => {
                PolarsError::ComputeError(format!("{}: {}", $dist_name, $e).into())
            };
        }
        match $name {
            "normal" => {
                let d = rand_distr::Normal::new($p1, $p2.unwrap()).map_err(|e| err!("normal", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "lognormal" => {
                let d = rand_distr::LogNormal::new($p1, $p2.unwrap()).map_err(|e| err!("lognormal", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "beta" => {
                let d = rand_distr::Beta::new($p1, $p2.unwrap()).map_err(|e| err!("beta", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "gamma" => {
                // statrs Gamma(shape, rate) but rand_distr Gamma(shape, scale=1/rate)
                let d = rand_distr::Gamma::new($p1, 1.0 / $p2.unwrap()).map_err(|e| err!("gamma", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "cauchy" => {
                let d = rand_distr::Cauchy::new($p1, $p2.unwrap()).map_err(|e| err!("cauchy", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "chi_squared" => {
                let d = rand_distr::ChiSquared::new($p1).map_err(|e| err!("chi_squared", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "exponential" => {
                let d = rand_distr::Exp::new($p1).map_err(|e| err!("exponential", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "fisher_snedecor" => {
                let d = rand_distr::FisherF::new($p1, $p2.unwrap()).map_err(|e| err!("fisher_snedecor", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "gumbel" => {
                let d = rand_distr::Gumbel::new($p1, $p2.unwrap()).map_err(|e| err!("gumbel", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "inverse_gamma" => {
                // Sample as 1/Gamma(alpha, 1/beta)
                let d = rand_distr::Gamma::new($p1, 1.0 / $p2.unwrap()).map_err(|e| err!("inverse_gamma", e))?;
                Ok((0..$n).map(|_| 1.0 / $rng.sample(d)).collect::<Vec<f64>>())
            }
            "laplace" => {
                // Laplace via inverse CDF
                let uniform = rand_distr::Uniform::new(0.0f64, 1.0).unwrap();
                let loc = $p1;
                let scale = $p2.unwrap();
                Ok((0..$n).map(|_| {
                    let u: f64 = $rng.sample(uniform);
                    let v = u - 0.5;
                    loc - scale * v.signum() * (1.0 - 2.0 * v.abs()).ln()
                }).collect::<Vec<f64>>())
            }
            "pareto" => {
                // rand_distr::Pareto::new(scale, shape)
                let d = rand_distr::Pareto::new($p2.unwrap(), $p1).map_err(|e| err!("pareto", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "students_t" => {
                let d = rand_distr::StudentT::new($p1).map_err(|e| err!("students_t", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "triangular" => {
                let d = rand_distr::Triangular::new($p1, $p2.unwrap(), $p3.unwrap()).map_err(|e| err!("triangular", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "uniform" => {
                let d = rand_distr::Uniform::new($p1, $p2.unwrap()).map_err(|e| err!("uniform", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "weibull" => {
                // rand_distr::Weibull::new(scale, shape)
                let d = rand_distr::Weibull::new($p2.unwrap(), $p1).map_err(|e| err!("weibull", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "pert" => {
                // Pert::new(min, max).with_mode(mode)
                let d = rand_distr::Pert::new($p1, $p2.unwrap())
                    .with_mode($p3.unwrap())
                    .map_err(|e| err!("pert", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "skew_normal" => {
                let d = rand_distr::SkewNormal::new($p1, $p2.unwrap(), $p3.unwrap()).map_err(|e| err!("skew_normal", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "inverse_gaussian" => {
                let d = rand_distr::InverseGaussian::new($p1, $p2.unwrap()).map_err(|e| err!("inverse_gaussian", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "frechet" => {
                // Frechet::new(location, scale, shape)
                let d = rand_distr::Frechet::new($p1, $p2.unwrap(), $p3.unwrap()).map_err(|e| err!("frechet", e))?;
                Ok((0..$n).map(|_| $rng.sample(d)).collect::<Vec<f64>>())
            }
            "zeta" => {
                let d = rand_distr::Zeta::new($p1).map_err(|e| err!("zeta", e))?;
                Ok((0..$n).map(|_| $rng.sample::<f64, _>(&d)).collect::<Vec<f64>>())
            }
            "zipf" => {
                // Zipf::new(n, s) where n is f64
                let d = rand_distr::Zipf::new($p1, $p2.unwrap()).map_err(|e| err!("zipf", e))?;
                Ok((0..$n).map(|_| $rng.sample::<f64, _>(&d)).collect::<Vec<f64>>())
            }
            "bernoulli" => {
                let d = rand_distr::Bernoulli::new($p1).map_err(|e| err!("bernoulli", e))?;
                Ok((0..$n).map(|_| if $rng.sample(&d) { 1.0 } else { 0.0 }).collect::<Vec<f64>>())
            }
            "binomial" => {
                let d = rand_distr::Binomial::new($p1 as u64, $p2.unwrap()).map_err(|e| err!("binomial", e))?;
                Ok((0..$n).map(|_| $rng.sample(&d) as f64).collect::<Vec<f64>>())
            }
            "geometric" => {
                let d = rand_distr::Geometric::new($p1).map_err(|e| err!("geometric", e))?;
                Ok((0..$n).map(|_| $rng.sample(&d) as f64).collect::<Vec<f64>>())
            }
            "poisson" => {
                let d = rand_distr::Poisson::new($p1).map_err(|e| err!("poisson", e))?;
                Ok((0..$n).map(|_| $rng.sample::<f64, _>(&d)).collect::<Vec<f64>>())
            }
            other => {
                Err(PolarsError::ComputeError(format!("unknown distribution for sampling: {other}").into()))
            }
        }
    }};
}

/// Core sampling logic: returns a Vec<f64> of N draws from the given distribution.
pub(crate) fn sample_vec(
    dist: &str,
    n: usize,
    param1: f64,
    param2: Option<f64>,
    param3: Option<f64>,
    seed: Option<u64>,
) -> PolarsResult<Vec<f64>> {
    let mut rng = make_rng(seed);
    dispatch_sample!(dist, param1, param2, param3, n, rng)
}

/// Fixed-parameter sampling: generates a series of N draws.
#[polars_expr(output_type=Float64)]
fn dist_sample(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let n = inputs[0].len();
    let values = sample_vec(
        kwargs.dist.as_str(),
        n,
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        kwargs.seed,
    )?;
    let ca = Float64Chunked::from_vec("sample".into(), values);
    Ok(ca.into_series())
}

/// Column-parameterized sampling: per-row distribution params from columns.
#[polars_expr(output_type=Float64)]
fn dist_sample_col(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let p1_ca = inputs[0].f64()?;
    let p2_ca = if inputs.len() > 1 {
        Some(inputs[1].f64()?)
    } else {
        None
    };
    let p3_ca = if inputs.len() > 2 {
        Some(inputs[2].f64()?)
    } else {
        None
    };
    let n = p1_ca.len();
    let mut rng = make_rng(kwargs.seed);
    let dist_name = kwargs.dist.as_str();

    let mut values: Vec<Option<f64>> = Vec::with_capacity(n);

    for i in 0..n {
        let p1 = p1_ca.get(i);
        let p2 = p2_ca.as_ref().and_then(|c| c.get(i));
        let p3 = p3_ca.as_ref().and_then(|c| c.get(i));

        if p1.is_none() {
            values.push(None);
            continue;
        }
        let p1 = p1.unwrap();

        let sampled: Vec<f64> = dispatch_sample!(dist_name, p1, p2, p3, 1usize, rng)?;
        values.push(Some(sampled[0]));
    }

    let ca = Float64Chunked::from_iter_options("sample".into(), values.into_iter());
    Ok(ca.into_series())
}
