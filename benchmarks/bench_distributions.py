"""Benchmark polars-sdist vs numpy/scipy for statistical distributions.

Usage:
    python benchmarks/bench_distributions.py [--sizes 1000,100000,1000000]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl
from scipy import stats

import polars_sdist
from polars_sdist import SdistNamespace as sdist


def _time_ms(fn, warmup: int = 1, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _row(dist: str, op: str, n: int, polars_ms: float, numpy_ms: float) -> dict:
    return {
        "distribution": dist,
        "operation": op,
        "n": n,
        "polars_ms": round(polars_ms, 3),
        "numpy_ms": round(numpy_ms, 3),
        "speedup": round(numpy_ms / polars_ms, 2),
    }


def bench_normal(n: int) -> list[dict]:
    results = []
    x = np.random.default_rng(42).standard_normal(n)
    df = pl.DataFrame({"x": x})
    p = np.random.default_rng(42).uniform(0.001, 0.999, n)
    df_p = pl.DataFrame({"p": p})

    for op, pl_fn, np_fn in [
        ("cdf", lambda: df.select(sdist(pl.col("x")).normal_cdf(mu=0, sigma=1)), lambda: stats.norm.cdf(x)),
        ("pdf", lambda: df.select(sdist(pl.col("x")).normal_pdf(mu=0, sigma=1)), lambda: stats.norm.pdf(x)),
        ("ppf", lambda: df_p.select(sdist(pl.col("p")).normal_ppf(mu=0, sigma=1)), lambda: stats.norm.ppf(p)),
        ("sample", lambda: polars_sdist.sample_normal(n=n, seed=42), lambda: np.random.default_rng(42).standard_normal(n)),
    ]:
        results.append(_row("normal", op, n, _time_ms(pl_fn), _time_ms(np_fn)))
    return results


def bench_lognormal(n: int) -> list[dict]:
    results = []
    x = np.random.default_rng(42).lognormal(0, 1, n)
    df = pl.DataFrame({"x": x})
    p = np.random.default_rng(42).uniform(0.001, 0.999, n)
    df_p = pl.DataFrame({"p": p})

    for op, pl_fn, np_fn in [
        ("cdf", lambda: df.select(sdist(pl.col("x")).lognormal_cdf(mu=0, sigma=1)), lambda: stats.lognorm.cdf(x, s=1)),
        ("pdf", lambda: df.select(sdist(pl.col("x")).lognormal_pdf(mu=0, sigma=1)), lambda: stats.lognorm.pdf(x, s=1)),
        ("ppf", lambda: df_p.select(sdist(pl.col("p")).lognormal_ppf(mu=0, sigma=1)), lambda: stats.lognorm.ppf(p, s=1)),
        ("sample", lambda: polars_sdist.sample_lognormal(n=n, seed=42), lambda: np.random.default_rng(42).lognormal(0, 1, n)),
    ]:
        results.append(_row("lognormal", op, n, _time_ms(pl_fn), _time_ms(np_fn)))
    return results


def bench_beta(n: int) -> list[dict]:
    results = []
    x = np.random.default_rng(42).beta(2, 5, n)
    df = pl.DataFrame({"x": x})
    p = np.random.default_rng(42).uniform(0.001, 0.999, n)
    df_p = pl.DataFrame({"p": p})

    for op, pl_fn, np_fn in [
        ("cdf", lambda: df.select(sdist(pl.col("x")).beta_cdf(alpha=2, beta=5)), lambda: stats.beta.cdf(x, 2, 5)),
        ("pdf", lambda: df.select(sdist(pl.col("x")).beta_pdf(alpha=2, beta=5)), lambda: stats.beta.pdf(x, 2, 5)),
        ("ppf", lambda: df_p.select(sdist(pl.col("p")).beta_ppf(alpha=2, beta=5)), lambda: stats.beta.ppf(p, 2, 5)),
        ("sample", lambda: polars_sdist.sample_beta(n=n, alpha=2, beta=5, seed=42), lambda: np.random.default_rng(42).beta(2, 5, n)),
    ]:
        results.append(_row("beta", op, n, _time_ms(pl_fn), _time_ms(np_fn)))
    return results


def bench_chi_squared(n: int) -> list[dict]:
    results = []
    x = np.random.default_rng(42).chisquare(5, n)
    df = pl.DataFrame({"x": x})
    p = np.random.default_rng(42).uniform(0.001, 0.999, n)
    df_p = pl.DataFrame({"p": p})

    for op, pl_fn, np_fn in [
        ("cdf", lambda: df.select(sdist(pl.col("x")).chi_squared_cdf(df=5)), lambda: stats.chi2.cdf(x, 5)),
        ("pdf", lambda: df.select(sdist(pl.col("x")).chi_squared_pdf(df=5)), lambda: stats.chi2.pdf(x, 5)),
        ("ppf", lambda: df_p.select(sdist(pl.col("p")).chi_squared_ppf(df=5)), lambda: stats.chi2.ppf(p, 5)),
        ("sample", lambda: polars_sdist.sample_chi_squared(n=n, df=5, seed=42), lambda: np.random.default_rng(42).chisquare(5, n)),
    ]:
        results.append(_row("chi_squared", op, n, _time_ms(pl_fn), _time_ms(np_fn)))
    return results


def bench_binomial(n: int) -> list[dict]:
    results = []
    k = np.random.default_rng(42).binomial(10, 0.5, n).astype(np.float64)
    df = pl.DataFrame({"k": k})

    for op, pl_fn, np_fn in [
        ("pmf", lambda: df.select(sdist(pl.col("k")).binomial_pmf(n=10, p=0.5)), lambda: stats.binom.pmf(k, 10, 0.5)),
        ("cdf", lambda: df.select(sdist(pl.col("k")).binomial_cdf(n=10, p=0.5)), lambda: stats.binom.cdf(k, 10, 0.5)),
        ("sample", lambda: polars_sdist.sample_binomial(n=n, trials=10, p=0.5, seed=42), lambda: np.random.default_rng(42).binomial(10, 0.5, n)),
    ]:
        results.append(_row("binomial", op, n, _time_ms(pl_fn), _time_ms(np_fn)))
    return results


ALL_BENCHMARKS = [bench_normal, bench_lognormal, bench_beta, bench_chi_squared, bench_binomial]


def main():
    parser = argparse.ArgumentParser(description="Benchmark polars-sdist vs numpy/scipy")
    parser.add_argument("--sizes", default="1000,100000,1000000")
    args = parser.parse_args()
    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    all_results = []
    for size in sizes:
        print(f"\n{'='*64}")
        print(f"  n = {size:,}")
        print(f"{'='*64}")
        for bench_fn in ALL_BENCHMARKS:
            results = bench_fn(size)
            all_results.extend(results)
            for r in results:
                arrow = ">>>" if r["speedup"] > 1 else "<<<"
                print(
                    f"  {r['distribution']:14s} {r['operation']:7s} "
                    f"polars={r['polars_ms']:8.2f}ms  "
                    f"scipy={r['numpy_ms']:8.2f}ms  "
                    f"{arrow} {r['speedup']:5.2f}x"
                )

    # Summary
    print(f"\n{'='*64}")
    print("  SUMMARY (geometric mean speedup by operation)")
    print(f"{'='*64}")
    by_op: dict[str, list[float]] = {}
    for r in all_results:
        by_op.setdefault(r["operation"], []).append(r["speedup"])
    for op, speedups in sorted(by_op.items()):
        geo = np.exp(np.mean(np.log(speedups)))
        faster = sum(1 for s in speedups if s > 1)
        print(f"  {op:7s}  geo_mean={geo:5.2f}x  polars faster in {faster}/{len(speedups)}")

    all_speedups = [r["speedup"] for r in all_results]
    print(f"\n  Overall: {np.exp(np.mean(np.log(all_speedups))):.2f}x geometric mean")


if __name__ == "__main__":
    main()
