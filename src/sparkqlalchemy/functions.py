from typing import TYPE_CHECKING

from sqlalchemy import func as sa_func
from sqlalchemy import literal

from ._max_min_by import _MaxByFn, _MinByFn
from .column import Column, WhenExpr, _to_col

if TYPE_CHECKING:
    from typing import Any


def col(name: str) -> Column:
    """Return a :class:`Column` referencing a column by name, resolved at query-build time.

    Parameters
    ----------
    name : str
        The column name.

    Returns
    -------
    :class:`Column`
    """
    return Column._deferred(name)


def lit(value: "Any") -> Column:
    """Return a :class:`Column` wrapping a SQL literal value.

    Parameters
    ----------
    value : "Any"
        The literal value.

    Returns
    -------
    :class:`Column`
    """
    return Column(lambda _: literal(value), repr(value))


def sum(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the aggregate sum of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to sum.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.sum(c._resolver(reg)), f"sum({c._name})")


def avg(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the aggregate average of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to average.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.avg(c._resolver(reg)), f"avg({c._name})")


def mean(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the aggregate average of `column`.

    Alias for :func:`avg`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to average.

    Returns
    -------
    :class:`Column`
    """
    return avg(column)


def count(column: str | Column = "*") -> Column:
    """Return a :class:`Column` that counts rows or non-null values.

    Parameters
    ----------
    column : str | :class:`Column`, optional
        The column to count non-null values of. Pass ``"*"`` (the default)
        to count all rows regardless of nulls.

    Returns
    -------
    :class:`Column`
    """
    if isinstance(column, str) and column == "*":
        return Column(lambda _: sa_func.count(), "count(1)")
    c = _to_col(column)
    return Column(lambda reg: sa_func.count(c._resolver(reg)), f"count({c._name})")


def countDistinct(column: str | Column) -> Column:
    """Return a :class:`Column` that counts the distinct non-null values of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to count distinct values of.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(
        lambda reg: sa_func.count(sa_func.distinct(c._resolver(reg))),
        f"count_distinct({c._name})",
    )


def count_distinct(column: str | Column) -> Column:
    """Return a :class:`Column` that counts the distinct non-null values of `column`.

    Alias for :func:`countDistinct`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to count distinct values of.

    Returns
    -------
    :class:`Column`
    """
    return countDistinct(column)


def max(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the aggregate maximum of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to take the maximum of.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.max(c._resolver(reg)), f"max({c._name})")


def min(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the aggregate minimum of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to take the minimum of.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.min(c._resolver(reg)), f"min({c._name})")


def first(column: str | Column) -> Column:
    """Return the first encountered value of `column` in a group.

    Uses ``ANY_VALUE`` semantics for MySQL compatibility.

    Parameters
    ----------
    column : str | :class:`Column`
        The column whose first value to return.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.any_value(c._resolver(reg)), f"first({c._name})")


def max_by(column: str | Column, ord_column: str | Column) -> Column:
    """Return the value of `column` from the row with the maximum value of `ord_column` in each group.

    Compiles to native `max_by` where available (SQLite 3.44+,
    PostgreSQL 16+, DuckDB, ClickHouse) and falls back to portable
    equivalents on older PostgreSQL (`array_agg` with `ORDER BY`) and
    MySQL (`GROUP_CONCAT` with `ORDER BY`, auto-cast back to the
    column's original type).

    Parameters
    ----------
    column : str | :class:`Column`
        The column whose value to return.
    ord_column : str | :class:`Column`
        The column whose maximum determines which row's `column` value is returned.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    o = _to_col(ord_column)
    c_res, o_res = c._resolver, o._resolver
    return Column(
        lambda reg: _MaxByFn(c_res(reg), o_res(reg)),
        f"max_by({c._name}, {o._name})",
    )


def min_by(column: str | Column, ord_column: str | Column) -> Column:
    """Return the value of `column` from the row with the minimum value of `ord_column` in each group.

    Compiles to native `min_by` where available (SQLite 3.44+,
    PostgreSQL 16+, DuckDB, ClickHouse) and falls back to portable
    equivalents on older PostgreSQL (`array_agg` with `ORDER BY`) and
    MySQL (`GROUP_CONCAT` with `ORDER BY`, auto-cast back to the
    column's original type).

    Parameters
    ----------
    column : str | :class:`Column`
        The column whose value to return.
    ord_column : str | :class:`Column`
        The column whose minimum determines which row's `column` value is returned.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    o = _to_col(ord_column)
    c_res, o_res = c._resolver, o._resolver
    return Column(
        lambda reg: _MinByFn(c_res(reg), o_res(reg)),
        f"min_by({c._name}, {o._name})",
    )


def coalesce(*columns: str | Column) -> Column:
    """Return a :class:`Column` that evaluates to the first non-null value among `columns`.

    Parameters
    ----------
    *columns : str | :class:`Column`
        Columns to evaluate in order.

    Returns
    -------
    :class:`Column`
    """
    cols = [_to_col(c) for c in columns]
    resolvers = [c._resolver for c in cols]
    return Column(
        lambda reg: sa_func.coalesce(*(r(reg) for r in resolvers)),
        "coalesce(...)",
    )


def concat(*columns: str | Column) -> Column:
    """Return a :class:`Column` that concatenates `columns` as strings.

    Parameters
    ----------
    *columns : str | :class:`Column`
        Columns whose string representations are concatenated.

    Returns
    -------
    :class:`Column`
    """
    cols = [_to_col(c) for c in columns]
    resolvers = [c._resolver for c in cols]
    return Column(
        lambda reg: sa_func.concat(*(r(reg) for r in resolvers)),
        "concat(...)",
    )


def upper(column: str | Column) -> Column:
    """Return a :class:`Column` that converts `column` to upper case.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to convert.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.upper(c._resolver(reg)), f"upper({c._name})")


def lower(column: str | Column) -> Column:
    """Return a :class:`Column` that converts `column` to lower case.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to convert.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.lower(c._resolver(reg)), f"lower({c._name})")


def length(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the character length of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to measure.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.length(c._resolver(reg)), f"length({c._name})")


def trim(column: str | Column) -> Column:
    """Return a :class:`Column` that strips leading and trailing whitespace from `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to trim.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.trim(c._resolver(reg)), f"trim({c._name})")


def abs(column: str | Column) -> Column:
    """Return a :class:`Column` that computes the absolute value of `column`.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to apply the absolute value to.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(lambda reg: sa_func.abs(c._resolver(reg)), f"abs({c._name})")


def round(column: str | Column, scale: int = 0) -> Column:
    """Return a :class:`Column` that rounds `column` to `scale` decimal places.

    Parameters
    ----------
    column : str | :class:`Column`
        The column to round.
    scale : int, optional
        Number of decimal places to round to. Defaults to 0.

    Returns
    -------
    :class:`Column`
    """
    c = _to_col(column)
    return Column(
        lambda reg: sa_func.round(c._resolver(reg), scale), f"round({c._name})"
    )


def when(condition: Column, value: "Any") -> "WhenExpr":
    """Begin a SQL `CASE WHEN` expression.

    Parameters
    ----------
    condition : :class:`Column`
        The boolean condition for the first `WHEN` clause.
    value : "Any"
        The value to return when `condition` is true.

    Returns
    -------
    :class:`WhenExpr`
    """
    return WhenExpr(condition, value)
