"""
The `max_by` and `min_by` functions need special logic depending on the underlying
database type so we implement overrides here.
"""

import sqlite3 as _sqlite3
from typing import TYPE_CHECKING

from sqlalchemy import event as _sa_event
from sqlalchemy.engine import Engine as _Engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.sql.type_api import TypeEngine


class _MaxByFn(FunctionElement):
    """A SQL `max_by(value, ordering)` aggregate with per-dialect compilation."""

    inherit_cache = True
    name = "max_by"


class _MinByFn(FunctionElement):
    """A SQL `min_by(value, ordering)` aggregate with per-dialect compilation."""

    inherit_cache = True
    name = "min_by"


# -----
# Default: emit native max_by / min_by


@compiles(_MaxByFn)
def _compile_max_by_default(element: _MaxByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    return (
        f"max_by({compiler.process(value, **kw)}, {compiler.process(ordering, **kw)})"
    )


@compiles(_MinByFn)
def _compile_min_by_default(element: _MinByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    return (
        f"min_by({compiler.process(value, **kw)}, {compiler.process(ordering, **kw)})"
    )


# -----
# SQLite: implement in Python


class _MaxByAgg:
    """Python-level `max_by` aggregate for SQLite."""

    def __init__(self):
        self._max_ord: "Any" = None
        self._result: "Any" = None

    def step(self, value: "Any", ordering: "Any"):
        if self._max_ord is None or (ordering is not None and ordering > self._max_ord):
            self._max_ord = ordering
            self._result = value

    def finalize(self) -> "Any":
        return self._result


class _MinByAgg:
    """Python-level `min_by` aggregate for SQLite."""

    def __init__(self):
        self._min_ord: "Any" = None
        self._result: "Any" = None

    def step(self, value: "Any", ordering: "Any"):
        if self._min_ord is None or (ordering is not None and ordering < self._min_ord):
            self._min_ord = ordering
            self._result = value

    def finalize(self) -> "Any":
        return self._result


@_sa_event.listens_for(_Engine, "connect")
def _register_sqlite_aggregates(dbapi_connection: "Any", connection_record: "Any"):
    if isinstance(dbapi_connection, _sqlite3.Connection):
        dbapi_connection.create_aggregate("max_by", 2, _MaxByAgg)  # type: ignore
        dbapi_connection.create_aggregate("min_by", 2, _MinByAgg)  # type: ignore


# -----
# PostgreSQL: (array_agg(value ORDER BY ordering DESC/ASC))[1]


@compiles(_MaxByFn, "postgresql")
def _compile_max_by_pg(element: _MaxByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    v = compiler.process(value, **kw)
    o = compiler.process(ordering, **kw)
    return f"(array_agg({v} ORDER BY {o} DESC))[1]"


@compiles(_MinByFn, "postgresql")
def _compile_min_by_pg(element: _MinByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    v = compiler.process(value, **kw)
    o = compiler.process(ordering, **kw)
    return f"(array_agg({v} ORDER BY {o} ASC))[1]"


# ----
# MySQL: SUBSTRING_INDEX(GROUP_CONCAT(... ORDER BY ...), sep, 1)

_MYSQL_SEP = "\x1f"  # ASCII Unit Separator — unlikely in data


def _mysql_cast_target(sa_type: "TypeEngine") -> str | None:
    """Return a MySQL `CAST` target string for the given SQLAlchemy type, or `None` if no cast is needed (i.e. the value is already a string)."""
    from sqlalchemy import types as sa_types

    # Walk MRO so dialect-specific subtypes (e.g. mysql.BIGINT) are caught.
    for cls in type(sa_type).__mro__:
        if cls in (sa_types.Float, sa_types.Numeric):
            prec = getattr(sa_type, "precision", None) or 65
            scale = getattr(sa_type, "scale", None) or 6
            return f"DECIMAL({prec}, {scale})"
        if cls is sa_types.Integer:
            return "SIGNED"
        if cls is sa_types.DateTime:
            return "DATETIME"
        if cls is sa_types.Date:
            return "DATE"
        if cls is sa_types.Time:
            return "TIME"
        if cls is sa_types.Boolean:
            return "SIGNED"
    return None


@compiles(_MaxByFn, "mysql")
def _compile_max_by_mysql(element: _MaxByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    v = compiler.process(value, **kw)
    o = compiler.process(ordering, **kw)
    sep = _MYSQL_SEP
    inner = (
        f"SUBSTRING_INDEX(GROUP_CONCAT({v} ORDER BY {o} DESC "
        f"SEPARATOR '{sep}'), '{sep}', 1)"
    )
    target = _mysql_cast_target(value.type)
    return f"CAST({inner} AS {target})" if target else inner


@compiles(_MinByFn, "mysql")
def _compile_min_by_mysql(element: _MinByFn, compiler: "Any", **kw: "Any") -> str:
    value, ordering = list(element.clauses)
    v = compiler.process(value, **kw)
    o = compiler.process(ordering, **kw)
    sep = _MYSQL_SEP
    inner = (
        f"SUBSTRING_INDEX(GROUP_CONCAT({v} ORDER BY {o} ASC "
        f"SEPARATOR '{sep}'), '{sep}', 1)"
    )
    target = _mysql_cast_target(value.type)
    return f"CAST({inner} AS {target})" if target else inner
