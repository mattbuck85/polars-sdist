use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use statrs::distribution::{Discrete as DiscreteTrait, DiscreteCDF};

use super::DistKwargs;

/// Helper macro for discrete distributions where the method takes u64.
macro_rules! dispatch_discrete_u64 {
    ($name:expr, $p1:expr, $p2:expr, $p3:expr, $method:ident, $ca:expr) => {{
        let name = $ca.name().clone();
        macro_rules! err {
            ($dist_name:expr, $e:expr) => {
                PolarsError::ComputeError(format!("{}: {}", $dist_name, $e).into())
            };
        }
        match $name {
            "bernoulli" => {
                let d =
                    statrs::distribution::Bernoulli::new($p1).map_err(|e| err!("bernoulli", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "binomial" => {
                let d = statrs::distribution::Binomial::new($p2.unwrap(), $p1 as u64)
                    .map_err(|e| err!("binomial", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "geometric" => {
                let d =
                    statrs::distribution::Geometric::new($p1).map_err(|e| err!("geometric", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "hypergeometric" => {
                let d = statrs::distribution::Hypergeometric::new(
                    $p1 as u64,
                    $p2.unwrap() as u64,
                    $p3.unwrap() as u64,
                )
                .map_err(|e| err!("hypergeometric", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "negative_binomial" => {
                let d = statrs::distribution::NegativeBinomial::new($p1, $p2.unwrap())
                    .map_err(|e| err!("negative_binomial", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "poisson" => {
                let d = statrs::distribution::Poisson::new($p1).map_err(|e| err!("poisson", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as u64))),
                ))
            }
            "discrete_uniform" => {
                let d = statrs::distribution::DiscreteUniform::new($p1 as i64, $p2.unwrap() as i64)
                    .map_err(|e| err!("discrete_uniform", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x as i64))),
                ))
            }
            other => Err(PolarsError::ComputeError(
                format!("unknown discrete distribution: {other}").into(),
            )),
        }
    }};
}

/// Helper for discrete ppf (inverse_cdf takes f64, returns u64/i64).
macro_rules! dispatch_discrete_ppf {
    ($name:expr, $p1:expr, $p2:expr, $p3:expr, $ca:expr) => {{
        let name = $ca.name().clone();
        macro_rules! err {
            ($dist_name:expr, $e:expr) => {
                PolarsError::ComputeError(format!("{}: {}", $dist_name, $e).into())
            };
        }
        match $name {
            "bernoulli" => {
                let d =
                    statrs::distribution::Bernoulli::new($p1).map_err(|e| err!("bernoulli", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "binomial" => {
                let d = statrs::distribution::Binomial::new($p2.unwrap(), $p1 as u64)
                    .map_err(|e| err!("binomial", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "geometric" => {
                let d =
                    statrs::distribution::Geometric::new($p1).map_err(|e| err!("geometric", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "hypergeometric" => {
                let d = statrs::distribution::Hypergeometric::new(
                    $p1 as u64,
                    $p2.unwrap() as u64,
                    $p3.unwrap() as u64,
                )
                .map_err(|e| err!("hypergeometric", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "negative_binomial" => {
                let d = statrs::distribution::NegativeBinomial::new($p1, $p2.unwrap())
                    .map_err(|e| err!("negative_binomial", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "poisson" => {
                let d = statrs::distribution::Poisson::new($p1).map_err(|e| err!("poisson", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            "discrete_uniform" => {
                let d = statrs::distribution::DiscreteUniform::new($p1 as i64, $p2.unwrap() as i64)
                    .map_err(|e| err!("discrete_uniform", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.inverse_cdf(x) as f64)),
                ))
            }
            other => Err(PolarsError::ComputeError(
                format!("unknown discrete distribution: {other}").into(),
            )),
        }
    }};
}

#[polars_expr(output_type=Float64)]
fn discrete_pmf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_discrete_u64!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        pmf,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn discrete_cdf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_discrete_u64!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        cdf,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn discrete_ppf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_discrete_ppf!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn discrete_sf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_discrete_u64!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        sf,
        ca
    )?;
    Ok(out.into_series())
}
