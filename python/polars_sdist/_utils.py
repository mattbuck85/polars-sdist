from __future__ import annotations

import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

LIB = Path(__file__).parent


def pl_plugin(*, symbol: str, args: list[pl.Expr], kwargs: dict) -> pl.Expr:
    return register_plugin_function(
        plugin_path=LIB,
        function_name=symbol,
        args=args,
        kwargs=kwargs,
        is_elementwise=True,
    )
