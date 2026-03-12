pub mod continuous;
pub mod discrete;
pub mod sampling;

use serde::Deserialize;

#[derive(Deserialize)]
pub struct DistKwargs {
    pub dist: String,
    pub param1: f64,
    pub param2: Option<f64>,
    pub param3: Option<f64>,
    pub seed: Option<u64>,
}

/// Dispatch macro for continuous distributions.
/// Constructs the distribution once, maps $method over Option<f64> iterator.
/// Returns PolarsResult<Float64Chunked>.
macro_rules! dispatch_continuous_map {
    ($name:expr, $p1:expr, $p2:expr, $p3:expr, $method:ident, $ca:expr) => {{
        let name = $ca.name().clone();
        macro_rules! err {
            ($dist_name:expr, $e:expr) => {
                PolarsError::ComputeError(format!("{}: {}", $dist_name, $e).into())
            };
        }
        match $name {
            "normal" => {
                let d = statrs::distribution::Normal::new($p1, $p2.unwrap())
                    .map_err(|e| err!("normal", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "lognormal" => {
                let d = statrs::distribution::LogNormal::new($p1, $p2.unwrap())
                    .map_err(|e| err!("lognormal", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "beta" => {
                let d = statrs::distribution::Beta::new($p1, $p2.unwrap())
                    .map_err(|e| err!("beta", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "gamma" => {
                let d = statrs::distribution::Gamma::new($p1, $p2.unwrap())
                    .map_err(|e| err!("gamma", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "cauchy" => {
                let d = statrs::distribution::Cauchy::new($p1, $p2.unwrap())
                    .map_err(|e| err!("cauchy", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "chi_squared" => {
                let d = statrs::distribution::ChiSquared::new($p1)
                    .map_err(|e| err!("chi_squared", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "exponential" => {
                let d = statrs::distribution::Exp::new($p1).map_err(|e| err!("exponential", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "fisher_snedecor" => {
                let d = statrs::distribution::FisherSnedecor::new($p1, $p2.unwrap())
                    .map_err(|e| err!("fisher_snedecor", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "gumbel" => {
                let d = statrs::distribution::Gumbel::new($p1, $p2.unwrap())
                    .map_err(|e| err!("gumbel", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "inverse_gamma" => {
                let d = statrs::distribution::InverseGamma::new($p1, $p2.unwrap())
                    .map_err(|e| err!("inverse_gamma", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "laplace" => {
                let d = statrs::distribution::Laplace::new($p1, $p2.unwrap())
                    .map_err(|e| err!("laplace", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "pareto" => {
                let d = statrs::distribution::Pareto::new($p1, $p2.unwrap())
                    .map_err(|e| err!("pareto", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "students_t" => {
                let d = statrs::distribution::StudentsT::new(0.0, 1.0, $p1)
                    .map_err(|e| err!("students_t", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "triangular" => {
                let d = statrs::distribution::Triangular::new($p1, $p2.unwrap(), $p3.unwrap())
                    .map_err(|e| err!("triangular", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "uniform" => {
                let d = statrs::distribution::Uniform::new($p1, $p2.unwrap())
                    .map_err(|e| err!("uniform", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            "weibull" => {
                let d = statrs::distribution::Weibull::new($p1, $p2.unwrap())
                    .map_err(|e| err!("weibull", e))?;
                Ok(Float64Chunked::from_iter_options(
                    name,
                    $ca.into_iter().map(|v| v.map(|x| d.$method(x))),
                ))
            }
            other => Err(PolarsError::ComputeError(
                format!("unknown continuous distribution: {other}").into(),
            )),
        }
    }};
}

pub(crate) use dispatch_continuous_map;
