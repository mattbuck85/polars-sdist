use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use statrs::distribution::{Continuous as ContinuousTrait, ContinuousCDF};

use super::dispatch_continuous_map;
use super::DistKwargs;

#[polars_expr(output_type=Float64)]
fn dist_pdf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_continuous_map!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        pdf,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn dist_ln_pdf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_continuous_map!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        ln_pdf,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn dist_cdf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_continuous_map!(
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
fn dist_ppf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_continuous_map!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        inverse_cdf,
        ca
    )?;
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn dist_sf(inputs: &[Series], kwargs: DistKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].f64()?;
    let out: Float64Chunked = dispatch_continuous_map!(
        kwargs.dist.as_str(),
        kwargs.param1,
        kwargs.param2,
        kwargs.param3,
        sf,
        ca
    )?;
    Ok(out.into_series())
}
