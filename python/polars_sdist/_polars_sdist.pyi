import polars as pl

def sample_direct(
    dist: str,
    n: int,
    param1: float,
    param2: float | None = None,
    param3: float | None = None,
    seed: int | None = None,
) -> pl.Series: ...
