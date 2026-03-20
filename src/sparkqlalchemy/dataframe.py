from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Self, TypeVar, cast, overload

from sqlalchemy import CursorResult, and_, literal, union_all
from sqlalchemy import delete as sa_delete
from sqlalchemy import func as sa_func
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select as sa_select
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session, aliased
from sqlalchemy.sql.elements import Label
from sqlalchemy.sql.util import ClauseAdapter
from typing_extensions import Sentinel

from . import functions as F
from .column import Column

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy import CompoundSelect, Select, Table

    from .column import SQLExpr


class Missing(Enum):
    MISSING = Sentinel("MISSING", "MISSING")

    def __repr__(self) -> str:
        return "MISSING"


MissingType = Literal[Missing.MISSING]
MISSING = Missing.MISSING


@dataclass(frozen=True)
class Row:
    """A lightweight result row with attribute, index, and dict access."""

    _data: dict[str, "Any"]

    def __init__(self, mapping: dict[str, "Any"]):
        # Use `object.__setattr__` to ensure immutability, and `dict(mapping)` to make it a copy of the dictionary.
        object.__setattr__(self, "_data", dict(mapping))

    @property
    def _keys(self) -> list[str]:
        return list(self._data.keys())

    def __getattr__(self, name: str) -> "Any":
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Row has no column '{name}'") from None

    def __getitem__(self, key: str | int) -> "Any":
        if isinstance(key, int):
            return self._data[self._keys[key]]
        return self._data[key]

    def asDict(self) -> dict:
        return dict(self._data)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Row({items})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Row):
            return self._data == other._data
        return NotImplemented

    def __len__(self) -> int:
        return len(self._data)


class GroupedData[DF: _DataFrameBase]:
    """An intermediate object returned by :meth:`_DataFrameBase.groupBy`.

    The only meaningful operation is :meth:`agg`, which returns a new :class:`_DataFrameBase` with the grouping columns plus aggregated columns.
    """

    def __init__(self, df: DF, group_exprs: list["SQLExpr"]):
        self._df = df
        self._group_exprs = group_exprs  # already-resolved SA expressions

    def agg(self, *exprs: str | Column) -> DF:
        """Apply aggregate expressions and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *exprs : str | :class:`Column`
            Aggregate column expressions (e.g. `F.sum("salary").alias("total")`).

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._df._build_agg(exprs, initial_entities=self._group_exprs)
        new._group_by_clauses = list(self._group_exprs)
        return new

    def count(self) -> DF:
        """Count the rows in each group and return a new :class:`_DataFrameBase`.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.agg(F.count("*").alias("count"))

    def sum(self, *cols: str) -> DF:
        """Sum one or more columns within each group and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str
            Names of the columns to sum.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.agg(*(F.sum(c).alias(f"sum({c})") for c in cols))

    def avg(self, *cols: str) -> DF:
        """Average one or more columns within each group and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str
            Names of the columns to average.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.agg(*(F.avg(c).alias(f"avg({c})") for c in cols))

    def mean(self, *cols: str) -> DF:
        """Average one or more columns within each group and return a new :class:`_DataFrameBase`.

        Alias for :meth:`GroupedData.avg`.

        Parameters
        ----------
        *cols : str
            Names of the columns to average.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.avg(*cols)

    def max(self, *cols: str) -> DF:
        """Return the maximum of one or more columns within each group as a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str
            Names of the columns to take the maximum of.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.agg(*(F.max(c).alias(f"max({c})") for c in cols))

    def min(self, *cols: str) -> DF:
        """Return the minimum of one or more columns within each group as a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str
            Names of the columns to take the minimum of.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.agg(*(F.min(c).alias(f"min({c})") for c in cols))


class _DataFrameBase[T]:
    """Base class for PySpark-style DataFrames backed by a SQLAlchemy ORM model.

    Contains all query-building, transformation, and inspection logic.
    Subclasses (:class:`DataFrame` and :class:`AsyncDataFrame`) provide
    the terminal methods that actually execute queries against the database.

    Parameters
    ----------
    session : :class:`~sqlalchemy.orm.Session` | :class:`~sqlalchemy.ext.asyncio.AsyncSession`
        An active SQLAlchemy ORM session.
    model : type
        An SQLAlchemy ORM mapped class.
    """

    def __init__(self, session: "Session | AsyncSession", model: type[T]):
        self._session = session
        self._model: type[T] = model
        self._sa_entity: "Any" = model  # May be replaced by aliased(model) via .alias()
        self._registry: dict[str, "SQLExpr | Column"] = {}
        self._alias_name: str | None = None

        # Populated by transformation methods
        self._select_entities: list["SQLExpr"] | None = None  # None ⇒ "all"
        self._where_clauses: list["SQLExpr"] = []
        self._group_by_clauses: list["SQLExpr"] = []
        self._having_clauses: list["SQLExpr"] = []
        self._order_by_clauses: list["SQLExpr"] = []
        self._joins: list[
            tuple["Any", "SQLExpr | None", str]
        ] = []  # (entity, on-clause, how)
        self._is_distinct: bool = False
        self._limit_val: int | None = None
        self._offset_val: int | None = None
        # Build initial column registry from the ORM model
        self._init_registry(model)

    def _init_registry(self, model: type):
        mapper = sa_inspect(model)
        for attr in mapper.column_attrs:
            self._registry[attr.key] = getattr(self._sa_entity, attr.key)

    @overload
    @classmethod
    def _factory(
        cls,
        *,
        df: "DataFrame[T]",
        target_type: "MissingType" = MISSING,
        session: Session | AsyncSession | MissingType = MISSING,
        model: type[T] | MissingType = MISSING,
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "DataFrame[T]":
        pass

    @overload
    @classmethod
    def _factory(
        cls,
        *,
        df: "AsyncDataFrame[T]",
        target_type: "MissingType" = MISSING,
        session: Session | AsyncSession | MissingType = MISSING,
        model: type[T] | MissingType = MISSING,
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "AsyncDataFrame[T]":
        pass

    @overload
    @classmethod
    def _factory(
        cls,
        *,
        df: "_DataFrameBase[T]",
        target_type: "type[DFT]",
        session: Session | AsyncSession | MissingType = MISSING,
        model: type[T] | MissingType = MISSING,
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "DFT":
        pass

    @overload
    @classmethod
    def _factory(
        cls,
        *,
        df: "MissingType" = MISSING,
        target_type: "type[DFT]",
        session: Session | AsyncSession,
        model: type[T],
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "DFT":
        pass

    @overload
    @classmethod
    def _factory(
        cls,
        *,
        df: "_DataFrameBase[T] | MissingType" = MISSING,
        target_type: "type | MissingType" = MISSING,
        session: Session | AsyncSession | MissingType = MISSING,
        model: type[T] | MissingType = MISSING,
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "_DataFrameBase[T]":
        pass

    @classmethod
    def _factory(
        cls,
        *,
        df: "_DataFrameBase[T] | MissingType" = MISSING,
        target_type: "type | MissingType" = MISSING,
        session: Session | AsyncSession | MissingType = MISSING,
        model: type[T] | MissingType = MISSING,
        sa_entity: "Any | MissingType" = MISSING,
        registry: dict[str, "SQLExpr | Column"] | MissingType = MISSING,
        alias_name: str | None | MissingType = MISSING,
        select_entities: list["SQLExpr"] | None | MissingType = MISSING,
        where_clauses: list["SQLExpr"] | MissingType = MISSING,
        group_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        having_clauses: list["SQLExpr"] | MissingType = MISSING,
        order_by_clauses: list["SQLExpr"] | MissingType = MISSING,
        joins: list[tuple["Any", "SQLExpr | None", str]] | MissingType = MISSING,
        is_distinct: bool | MissingType = MISSING,
        limit_val: int | None | MissingType = MISSING,
        offset_val: int | None | MissingType = MISSING,
    ) -> "_DataFrameBase[T]":
        """Creates a new DataFrame from the parameters that are present. If `df` is present, all missing parameters will default to the values in `df`; otherwise, `target_type`, `session`, and `model` must all be present, and all other parameters will default to the same defaults a fresh instance would have (mostly empty values).

        Parameters
        ----------
        df : _DataFrameBase[T], optional
            A DataFrame to template the new one off of. If `df` is present, all missing parameters will be copied from it. Otherwise, all of `target_type`, `session`, and `model` must be present.
        target_type : type, optional
            If present, must be either `DataFrame` or `AsyncDataFrame`, to create the correct type of object.
        session : Session | AsyncSession, optional
            If present, must be the correct type of session for the DataFrame type.
        model : type[T], optional
            The database table model.
        sa_entity : Any, optional
        registry : dict[str, "SQLExpr | Column"], optional
        alias_name : str | None, optional
        select_entities : list["SQLExpr"] | None, optional
        where_clauses : list["SQLExpr"], optional
        group_by_clauses : list["SQLExpr"], optional
        having_clauses : list["SQLExpr"], optional
        order_by_clauses : list["SQLExpr"], optional
        joins : list[tuple["Any", "SQLExpr | None", str]], optional
        is_distinct : bool | None, optional
        limit_val : int | None, optional
        offset_val : int | None, optional

        Returns
        -------
        _DataFrameBase[T]

        Raises
        ------
        AttributeError
            If `df` is not present and either `target_type`, `session`, or `model` is not present.
        ValueError
            If `session` is present and not the correct type of session for the DataFrame type.
        """
        if df is not MISSING:
            target_type = target_type if target_type is not MISSING else type(df)
            session = session if session is not MISSING else df._session
            model = model if model is not MISSING else df._model
            sa_entity = sa_entity if sa_entity is not MISSING else df._sa_entity
            registry = registry if registry is not MISSING else df._registry
            alias_name = alias_name if alias_name is not MISSING else df._alias_name
            select_entities = (
                select_entities
                if select_entities is not MISSING
                else df._select_entities
            )
            where_clauses = (
                where_clauses if where_clauses is not MISSING else df._where_clauses
            )
            group_by_clauses = (
                group_by_clauses
                if group_by_clauses is not MISSING
                else df._group_by_clauses
            )
            having_clauses = (
                having_clauses if having_clauses is not MISSING else df._having_clauses
            )
            order_by_clauses = (
                order_by_clauses
                if order_by_clauses is not MISSING
                else df._order_by_clauses
            )
            joins = joins if joins is not MISSING else df._joins
            is_distinct = is_distinct if is_distinct is not MISSING else df._is_distinct
            limit_val = limit_val if limit_val is not MISSING else df._limit_val
            offset_val = offset_val if offset_val is not MISSING else df._offset_val
        elif target_type is MISSING:
            raise AttributeError("Either `df` or `target_type` must be present.")
        elif session is MISSING:
            raise AttributeError("Either `df` or `session` must be present.")
        elif model is MISSING:
            raise AttributeError("Either `df` or `model` must be present.")
        else:
            sa_entity = sa_entity if sa_entity is not MISSING else model
            registry = registry if registry is not MISSING else {}
            alias_name = alias_name if alias_name is not MISSING else None
            select_entities = (
                select_entities if select_entities is not MISSING else None
            )
            where_clauses = where_clauses if where_clauses is not MISSING else []
            group_by_clauses = (
                group_by_clauses if group_by_clauses is not MISSING else []
            )
            having_clauses = having_clauses if having_clauses is not MISSING else []
            order_by_clauses = (
                order_by_clauses if order_by_clauses is not MISSING else []
            )
            joins = joins if joins is not MISSING else []
            is_distinct = is_distinct if is_distinct is not MISSING else False
            limit_val = limit_val if limit_val is not MISSING else None
            offset_val = offset_val if offset_val is not MISSING else None

        if (isinstance(session, Session) and (target_type is AsyncDataFrame)) or (
            isinstance(session, AsyncSession) and (target_type is DataFrame)
        ):
            raise ValueError("Invalid session type for the target DataFrame type.")

        new: _DataFrameBase[T] = object.__new__(target_type)
        new._session = session
        new._model = model
        new._sa_entity = sa_entity
        new._registry = dict(registry)
        new._alias_name = alias_name
        new._select_entities = (
            list(select_entities) if select_entities is not None else None
        )
        new._where_clauses = list(where_clauses)
        new._group_by_clauses = list(group_by_clauses)
        new._having_clauses = list(having_clauses)
        new._order_by_clauses = list(order_by_clauses)
        new._joins = list(joins)
        new._is_distinct = is_distinct
        new._limit_val = limit_val
        new._offset_val = offset_val
        return new

    @overload
    def _clone(self) -> Self:  # type: ignore
        pass

    def _clone(self) -> "_DataFrameBase[T]":
        """Return a deep copy of this object.

        Returns
        -------
        _DataFrameBase[T]
        """
        return _DataFrameBase._factory(df=self)

    @overload
    def _as_subquery(self) -> Self:  # type: ignore
        pass

    def _as_subquery(self) -> "_DataFrameBase[T]":
        """Wrap the current query as a SQL subquery and return a fresh instance backed by it."""
        subq = self._build_query().subquery()
        return _DataFrameBase._factory(
            target_type=type(self),
            session=self._session,
            model=self._model,
            sa_entity=subq,
            registry={c.name: subq.c[c.name] for c in subq.columns},
        )

    @property
    def _is_async(self) -> bool:
        return type(self) is AsyncDataFrame

    def create_async_copy(self, session: AsyncSession) -> "AsyncDataFrame[T]":
        """Return an async copy of this DataFrame. If it's already async, just clone it using the new session.

        Parameters
        ----------
        session : :class:`~sqlalchemy.ext.asyncio.AsyncSession`
            The asynchronous SQLAlchemy session to be used.

        Returns
        -------
        AsyncDataFrame[T]
        """
        if self._is_async:
            assert isinstance(self, AsyncDataFrame)
            new = self._clone()
            new._session = session
            return new

        return _DataFrameBase._factory(
            df=self,
            target_type=AsyncDataFrame,
            session=session,
        )

    @overload
    def _create_async_clone_if_needed(self, other: "DataFrame[Any]") -> Self:
        pass

    @overload
    def _create_async_clone_if_needed(
        self,
        other: "AsyncDataFrame[Any]",
    ) -> "AsyncDataFrame[T]":
        pass

    @overload
    def _create_async_clone_if_needed(
        self,
        other: "_DataFrameBase[Any]",
    ) -> "_DataFrameBase[T]":
        pass

    def _create_async_clone_if_needed(
        self,
        other: "_DataFrameBase[Any]",
    ) -> "_DataFrameBase[T]":
        """If this object is a sync DataFrame and `other` is an AsyncDataFrame, return an async copy of this object using `other`'s session; otherwise, return a copy of this object. Used by functions like `join` and `union`.

        Parameters
        ----------
        other : _DataFrameBase[Any]
            The other DataFrame to compare.

        Returns
        -------
        _DataFrameBase[T]
        """
        if self._is_async or (not other._is_async):
            new = self._clone()
        else:
            assert isinstance(other, AsyncDataFrame)
            new = self.create_async_copy(other._session)

        return new

    def alias(self, name: str) -> Self:
        """Give this :class:`_DataFrameBase` a table alias.

        Creates a true :class:`~sqlalchemy.orm.util.AliasedClass` entity so
        that self-joins and multi-table queries reference distinct table
        instances.  After aliasing, every base-model column is accessible both
        by its bare name and by the dot-qualified form `"alias.column"`.

        Columns from prior joins, `withColumn`, or `withColumnRenamed` are
        preserved.  Chaining aliases (`df.alias("a").alias("b")`) replaces
        the previous alias cleanly.

        Parameters
        ----------
        name : str
            The alias name.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._alias_name = name

        # Create a real SA aliased entity so self-joins work
        sa_alias = aliased(self._model, name=name)
        new._sa_entity = sa_alias

        # Build a ClauseAdapter that rewrites references from the old
        # entity (whether the bare model table or a previous alias) to
        # the new aliased table.  This keeps join conditions, WHERE
        # clauses, ORDER BY, etc. consistent with the new alias.
        old_selectable = sa_inspect(self._sa_entity).selectable
        new_selectable = sa_inspect(sa_alias, raiseerr=True).selectable
        adapter = ClauseAdapter(
            new_selectable,
            equivalents={
                c: {new_selectable.c[c.key]}
                for c in old_selectable.c
                if c.key in new_selectable.c
            },
        )

        def _adapt(expr: "Any") -> "Any":
            """Adapt a SA expression, passing through non-SA objects unchanged."""
            try:
                return adapter.traverse(expr)
            except AttributeError:
                return expr

        # Rewrite all stored clauses that may reference the old entity
        new._where_clauses = [_adapt(c) for c in new._where_clauses]
        new._group_by_clauses = [_adapt(c) for c in new._group_by_clauses]
        new._having_clauses = [_adapt(c) for c in new._having_clauses]
        new._order_by_clauses = [_adapt(c) for c in new._order_by_clauses]
        new._joins = [
            (entity, _adapt(cond) if cond is not None else None, how)
            for entity, cond, how in new._joins
        ]

        # Remove old base-model entries that are now stale and
        # re-register them under the new alias.
        mapper = sa_inspect(self._model, raiseerr=True)
        base_col_names = {a.key for a in mapper.column_attrs}
        old_prefix = f"{self._alias_name}." if self._alias_name else None
        keys_to_remove: set[str] = set()
        for key in new._registry:
            if key in base_col_names:
                # Preserve withColumn-computed labels; only remove raw column refs
                if not isinstance(new._registry[key], Label):
                    keys_to_remove.add(key)
            elif old_prefix and key.startswith(old_prefix):
                keys_to_remove.add(key)
        for key in keys_to_remove:
            del new._registry[key]

        # Adapt "Any" remaining registry entries (e.g. withColumn-computed
        # expressions) that may still reference the old entity.
        for key in list(new._registry.keys()):
            new._registry[key] = _adapt(new._registry[key])

        # Re-register base-model columns from the new aliased entity.
        # Skip bare-name registration if a withColumn-computed label
        # already occupies that key (it should take precedence).
        for attr in mapper.column_attrs:
            sa_col = getattr(sa_alias, attr.key)
            existing = new._registry.get(attr.key)
            if not isinstance(existing, Label):
                new._registry[attr.key] = sa_col  # bare: "salary"
            new._registry[f"{name}.{attr.key}"] = sa_col  # dotted: "a.salary"

        return new

    def _resolve(self, c: "Any") -> "SQLExpr":
        """Resolve a column reference to a concrete SQLAlchemy expression using this DataFrame's registry."""
        if isinstance(c, str):
            if c == "*":
                return literal(1)
            if c not in self._registry:
                raise KeyError(
                    f"Column '{c}' not found. "
                    f"Available: {sorted(self._registry.keys())}"
                )
            entry = self._registry[c]
            # If the registry entry is itself a Col, resolve it
            if isinstance(entry, Column):
                return entry.resolve(self._registry)
            return entry
        if isinstance(c, Column):
            return c.resolve(self._registry)
        # Assume it's already a raw SA expression
        return c

    def __getitem__(self, name: str) -> Column:
        """Return a :class:`Column` bound to a specific column in this :class:`_DataFrameBase`. Useful for disambiguating columns after a join.

        Parameters
        ----------
        name : str
            The column name.

        Returns
        -------
        :class:`Column`

        Raises
        ------
        KeyError
            `name` is not a column in this :class:`_DataFrameBase`.
        """
        if name not in self._registry:
            raise KeyError(
                f"Column '{name}' not found. Available: {sorted(self._registry.keys())}"
            )
        expr = self._registry[name]
        if isinstance(expr, Column):
            return expr
        return Column._wrap(expr, name)

    # -----
    # Transformation methods (each returns a *new* instance)

    def select(self, *cols: str | Column) -> Self:
        """Project a set of columns or expressions and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str | :class:`Column`
            Column names or column expressions to select.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._select_entities = []
        for c in cols:
            resolved = new._resolve(c)
            new._select_entities.append(resolved)
            _maybe_register_label(new._registry, resolved)
        return new

    def where(self, condition: "Any") -> Self:
        """Filter rows by a boolean condition and return a new :class:`_DataFrameBase`. If the DataFrame was grouped, this function is equivalent to using :meth:`_DataFrameBase.having`.

        Parameters
        ----------
        condition : :class:`Column` | "Any"
            A boolean column expression.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        if self._group_by_clauses:
            return self.having(condition)

        new = self._clone()
        new._where_clauses.append(new._resolve(condition))
        return new

    def filter(self, condition: "Any") -> Self:
        """Filter rows by a boolean condition and return a new :class:`_DataFrameBase`.

        Alias for :meth:`_DataFrameBase.where`.

        Parameters
        ----------
        condition : :class:`Column` | "Any"
            A boolean column expression.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.where(condition)

    def groupBy(self, *cols: str | Column) -> "GroupedData[Self]":
        """Group the :class:`_DataFrameBase` by the given columns and return a :class:`GroupedData` on which you call `agg`.

        Parameters
        ----------
        *cols : str | :class:`Column`
            Column names or expressions to group by.

        Returns
        -------
        :class:`GroupedData`
        """
        # If already aggregated, wrap in a subquery first
        df = self._subquery_if_grouped()
        group_exprs = [df._resolve(c) for c in cols]
        return GroupedData(df, group_exprs)

    def group_by(self, *cols: str | Column) -> "GroupedData[Self]":
        """Group the :class:`_DataFrameBase` by the given columns and return a :class:`GroupedData` on which you call `agg`.

        Alias for :meth:`_DataFrameBase.groupBy`.

        Parameters
        ----------
        *cols : str | :class:`Column`
            Column names or expressions to group by.

        Returns
        -------
        :class:`GroupedData`
        """
        return self.groupBy(*cols)

    def _subquery_if_grouped(self) -> Self:
        """Return this query as a subquery in case it is in a group by.

        Returns
        -------
        Self
        """
        return self._as_subquery() if self._group_by_clauses else self

    def _build_agg(
        self,
        exprs: tuple[str | Column, ...],
        initial_entities: "list[SQLExpr] | None" = None,
    ) -> Self:
        new = self._clone()
        new._select_entities = list(initial_entities) if initial_entities else []
        for e in exprs:
            resolved = new._resolve(e)
            new._select_entities.append(resolved)
            _maybe_register_label(new._registry, resolved)
        return new

    def agg(self, *exprs: str | Column) -> Self:
        """Aggregate the entire :class:`_DataFrameBase` without grouping and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *exprs : str | :class:`Column`
            Aggregate column expressions (e.g. ``F.sum("salary").alias("total")``).

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self._build_agg(exprs)

    def orderBy(self, *cols: str | Column) -> Self:
        """Sort by the given columns or expressions and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        *cols : str | :class:`Column`
            Column names or expressions to sort by.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._order_by_clauses = [new._resolve(c) for c in cols]
        return new

    def order_by(self, *cols: str | Column) -> Self:
        """Sort by the given columns or expressions and return a new :class:`_DataFrameBase`.

        Alias for :meth:`_DataFrameBase.orderBy`.

        Parameters
        ----------
        *cols : str | :class:`Column`
            Column names or expressions to sort by.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        return self.orderBy(*cols)

    def having(self, condition: "Any") -> Self:
        """Add a HAVING clause to filter groups after GROUP BY and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        condition : :class:`Column` | Any
            A boolean column expression applied after aggregation.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._having_clauses.append(new._resolve(condition))
        return new

    def _materialise_select(self) -> list["SQLExpr"]:
        """Populate `_select_entities` with all columns when still `None` and return the list.

        Called by `withColumn` / `withColumnRenamed` which need an explicit
        column list to modify.

        Returns
        -------
        list[SQLExpr]
        """
        if self._select_entities is None:
            if hasattr(self._sa_entity, "__table__") or hasattr(
                self._sa_entity, "__mapper__"
            ):
                # ORM model or aliased model — enumerate columns from the mapper
                mapper = sa_inspect(self._model, raiseerr=True)
                self._select_entities = [
                    getattr(self._sa_entity, a.key) for a in mapper.column_attrs
                ]
                for join_entity, _, _ in self._joins:
                    jmapper = sa_inspect(join_entity)
                    for a in jmapper.column_attrs:
                        self._select_entities.append(getattr(join_entity, a.key))
            else:
                # Subquery-backed entity (e.g. after union) — use registry
                self._select_entities = [self._resolve(c) for c in self._registry]
        return self._select_entities

    @staticmethod
    def _find_and_replace_entity(
        entities: list["SQLExpr"],
        target_name: str,
        replacement: "SQLExpr",
    ) -> bool:
        """Replace the first entity in `entities` whose name matches `target_name` and return `True` if a swap occurred.

        Parameters
        ----------
        entities : list[SQLExpr]
            The list of SQLAlchemy select entities to search.
        target_name : str
            The column name to match against.
        replacement : SQLExpr
            The expression to substitute in.

        Returns
        -------
        bool
        """
        for i, entity in enumerate(entities):
            ent_name = getattr(entity, "key", None) or getattr(entity, "name", None)
            if ent_name == target_name:
                entities[i] = replacement
                return True
        return False

    def withColumn(self, colName: str, col: str | Column) -> Self:
        """Add or replace a column and return a new :class:`_DataFrameBase`.

        If `select` has not yet been called, this materialises all base-model
        columns and appends the new one.  The new column is also registered by
        `colName` so that subsequent operations can reference it as a string.

        Parameters
        ----------
        colName : str
            The name of the new or replaced column.
        col : str | :class:`Column`
            The column expression to compute.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        labeled = new._resolve(col).label(colName)
        new._registry[colName] = labeled

        entities = new._materialise_select()
        if not self._find_and_replace_entity(entities, colName, labeled):
            entities.append(labeled)

        return new

    def withColumnRenamed(self, existing: str, new: str) -> Self:
        """Rename a column and return a new :class:`_DataFrameBase`. The old name is removed from the registry and the new name is registered.

        Parameters
        ----------
        existing : str
            The current name of the column.
        new : str
            The new name for the column.

        Returns
        -------
        :class:`_DataFrameBase`

        Raises
        ------
        KeyError
            `existing` is not a column in this :class:`_DataFrameBase`.
        """
        if existing not in self._registry:
            raise KeyError(
                f"Column '{existing}' not found. "
                f"Available: {sorted(self._registry.keys())}"
            )
        new_df = self._clone()
        old_expr = new_df._registry.pop(existing)
        if isinstance(old_expr, Column):
            old_expr = old_expr.resolve(new_df._registry)
        labeled = old_expr.label(new)
        new_df._registry[new] = labeled

        entities = new_df._materialise_select()
        self._find_and_replace_entity(entities, existing, labeled)

        return new_df

    @overload
    def join(
        self: "DataFrame[T]",
        other: "DataFrame[T]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "DataFrame[T]":
        pass

    @overload
    def join(
        self: "DataFrame[T]",
        other: "DataFrame[Any]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "DataFrame[Any]":
        pass

    @overload
    def join(
        self: "AsyncDataFrame[T]",
        other: "DataFrame[T]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "AsyncDataFrame[T]":
        pass

    @overload
    def join(
        self: "AsyncDataFrame[T]",
        other: "DataFrame[Any]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "AsyncDataFrame[Any]":
        pass

    @overload
    def join(
        self,
        other: "AsyncDataFrame[T]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "AsyncDataFrame[T]":
        pass

    @overload
    def join(
        self,
        other: "AsyncDataFrame[Any]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "AsyncDataFrame[Any]":
        pass

    def join(
        self,
        other: "_DataFrameBase[Any]",
        on: Column | str | list[str] | None = None,
        how: str = "inner",
    ) -> "_DataFrameBase[Any]":
        """Join with another :class:`_DataFrameBase` and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        other : :class:`_DataFrameBase`
            The right-hand :class:`_DataFrameBase`.
        on : :class:`Column` | str | list[str] | None, optional
            Join condition.  Accepts a :class:`Column` expression (including inequality joins), a single column-name string present in both DataFrames (equi-join), a list of column-name strings (multi-column equi-join), or `None` for a cross join. Defaults to `None`.
        how : str, optional
            Join type: `"inner"`, `"left"` / `"left_outer"`, `"right"` / `"right_outer"`, `"full"` / `"outer"` / `"full_outer"`, or `"cross"`. Defaults to `"inner"`.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        # If joining grouped queries they should be subqueried first
        new = self._subquery_if_grouped()._create_async_clone_if_needed(other)
        _other = other._subquery_if_grouped()

        # Auto-alias the right side for self-joins (same underlying model)
        # to prevent ambiguous column references.  Skip when either side is
        # backed by a subquery — subqueries are already distinct.
        def _is_orm_entity(entity: "Any") -> bool:
            return hasattr(entity, "__table__") or hasattr(entity, "__mapper__")

        if (
            _other._model is self._model
            and _other._alias_name is None
            and _is_orm_entity(new._sa_entity)
            and _is_orm_entity(_other._sa_entity)
        ):
            # Pick a name that won't collide with an existing alias
            auto_name = f"_{_other._model.__name__.lower()}_2"
            _other = _other.alias(auto_name)

        # Build a combined registry for resolving the ON clause.
        # Dot-prefixed entries (from aliases) are included automatically.
        combined_registry = {**new._registry, **_other._registry}

        # Resolve the join condition
        if isinstance(on, str):
            # Single column name equi-join.  Look up the bare name in each
            # side's own registry (so aliased names don't collide).
            left_expr = new._resolve(on)
            right_expr = _other._resolve(on)
            join_cond = left_expr == right_expr
        elif isinstance(on, (list, tuple)):
            parts = [new._resolve(c) == _other._resolve(c) for c in on]
            join_cond = and_(*parts)
        elif isinstance(on, Column):
            join_cond = on.resolve(combined_registry)
        else:
            join_cond = on  # raw SA expression or None (cross join)

        new._joins.append((_other._sa_entity, join_cond, how.lower()))

        # Merge the right-side registry into the new DataFrame's registry
        for key, val in _other._registry.items():
            new._registry[key] = val

        # If an explicit select() was called before the join, append the
        # right side's columns so they appear in the result automatically.
        # Respect the right side's own select() if it has one.
        if new._select_entities is not None:
            if _other._select_entities is not None:
                new._select_entities.extend(_other._select_entities)
            else:
                for key in _other._registry:
                    new._select_entities.append(new._resolve(key))

        return new

    def distinct(self) -> Self:
        """Return a new :class:`_DataFrameBase` with duplicate rows removed.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._is_distinct = True
        return new

    def limit(self, n: int) -> Self:
        """Limit the result to `n` rows and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        n : int
            Maximum number of rows.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._limit_val = n
        return new

    def offset(self, n: int) -> Self:
        """Skip the first `n` rows and return a new :class:`_DataFrameBase`.

        Parameters
        ----------
        n : int
            Number of rows to skip.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        new = self._clone()
        new._offset_val = n
        return new

    @overload
    def union(self: "DataFrame[T]", other: "DataFrame[T]") -> "DataFrame[T]":
        pass

    @overload
    def union(self: "DataFrame[T]", other: "DataFrame[Any]") -> "DataFrame[Any]":
        pass

    @overload
    def union(self: "AsyncDataFrame[T]", other: "DataFrame[T]") -> "AsyncDataFrame[T]":
        pass

    @overload
    def union(
        self: "AsyncDataFrame[T]",
        other: "DataFrame[Any]",
    ) -> "AsyncDataFrame[Any]":
        pass

    @overload
    def union(self, other: "AsyncDataFrame[T]") -> "AsyncDataFrame[T]":
        pass

    @overload
    def union(self, other: "AsyncDataFrame[Any]") -> "AsyncDataFrame[Any]":
        pass

    def union(self, other: "_DataFrameBase[Any]") -> "_DataFrameBase[Any]":
        """Combine two DataFrames with `UNION ALL` and return a new :class:`_DataFrameBase`. Both DataFrames must produce the same columns.

        Parameters
        ----------
        other : :class:`_DataFrameBase`
            The other :class:`_DataFrameBase` to union with.

        Returns
        -------
        :class:`_DataFrameBase`
        """
        left_stmt = self._build_query()
        right_stmt = other._build_query()
        subq = union_all(left_stmt, right_stmt).subquery()

        # Determine the target type and session (async wins if either side is async)
        if self._is_async or (not other._is_async):
            target_type = type(self)
            session = self._session
        else:
            target_type = type(other)
            session = other._session

        return _DataFrameBase._factory(
            target_type=target_type,
            session=session,
            model=self._model,
            sa_entity=subq,
            registry={c.name: subq.c[c.name] for c in subq.columns},
        )

    # -----
    # Query building

    def _build_query(self) -> "Select[Any] | CompoundSelect[Any]":
        """Compile the accumulated transformations into a :class:`~sqlalchemy.sql.Select`."""
        # Determine SELECT entities
        if self._select_entities is not None:
            entities = list(self._select_entities)
        elif self._joins:
            # Include all columns from both sides, matching PySpark semantics
            entities = [self._resolve(c) for c in self._registry]
        else:
            entities = [self._sa_entity]

        stmt = sa_select(*entities).select_from(self._sa_entity)

        # JOINs — each tuple stores (entity, on-clause, how)
        for join_entity, join_cond, how in self._joins:
            if how == "inner":
                stmt = stmt.join(join_entity, join_cond)
            elif how in ("left", "left_outer"):
                stmt = stmt.outerjoin(join_entity, join_cond)
            elif how in ("right", "right_outer"):
                stmt = stmt.join_from(
                    join_entity,
                    self._sa_entity,
                    join_cond,
                    isouter=True,
                )
            elif how in ("full", "full_outer", "outer"):
                stmt = stmt.outerjoin(join_entity, join_cond, full=True)
            elif how == "cross":
                stmt = stmt.join(join_entity, literal(True))

        # WHERE
        for clause in self._where_clauses:
            stmt = stmt.where(clause)

        # GROUP BY
        if self._group_by_clauses:
            stmt = stmt.group_by(*self._group_by_clauses)

        # HAVING
        for clause in self._having_clauses:
            stmt = stmt.having(clause)

        # ORDER BY
        if self._order_by_clauses:
            stmt = stmt.order_by(*self._order_by_clauses)

        # DISTINCT
        if self._is_distinct:
            stmt = stmt.distinct()

        # LIMIT / OFFSET
        if self._limit_val is not None:
            stmt = stmt.limit(self._limit_val)
        if self._offset_val is not None:
            stmt = stmt.offset(self._offset_val)

        return stmt

    # -----
    # Helpers for terminal methods

    @staticmethod
    def _rows_from_result(result) -> list[Row]:
        """Process a SQLAlchemy result object into a list of :class:`Row` objects."""
        rows: list[Row] = []
        for sa_row in result:
            # SA 2.0 Row objects expose `._mapping`
            if hasattr(sa_row, "_mapping"):
                mapping = dict(sa_row._mapping)
                # When selecting the whole model (`select(Model)`), the
                # mapping looks like `{"ModelName": <instance>}`.  Unpack
                # the ORM instance into its column attributes.
                if len(mapping) == 1:
                    single_val = next(iter(mapping.values()))
                    if hasattr(single_val, "__table__"):
                        mapper = sa_inspect(single_val.__class__)
                        mapping = {
                            a.key: getattr(single_val, a.key)
                            for a in mapper.column_attrs
                        }
                rows.append(Row(mapping))
            # Fallback for unexpected result shapes
            elif isinstance(sa_row, tuple):
                rows.append(Row({f"_{i}": v for i, v in enumerate(sa_row)}))
            else:
                rows.append(Row({"_0": sa_row}))
        return rows

    def _build_delete_stmt(self, condition: "Any | None" = None):
        """Build a DELETE statement from accumulated where clauses.

        Parameters
        ----------
        condition : :class:`Column` | "Any" | None, optional
            An additional boolean filter. Defaults to `None`.

        Returns
        -------
        The compiled DELETE statement.

        Raises
        ------
        RuntimeError
            No filter was provided.
        """
        df = self.where(condition) if condition is not None else self

        if not df._where_clauses:
            raise RuntimeError(
                "Refusing to delete without a filter. "
                "If you really want to delete every row, use .delete(lit(True))."
            )

        table, adapter = df._table_and_adapter()

        stmt = sa_delete(table)
        for clause in df._where_clauses:
            stmt = stmt.where(adapter.traverse(clause) if adapter else clause)  # type: ignore

        return stmt

    def _build_update_stmt(
        self,
        set_: dict[str, "Any"],
        where: "Any | None" = None,
    ):
        """Build an UPDATE statement from accumulated where clauses.

        Parameters
        ----------
        set_ : dict[str, :class:`Column` | "Any"]
            A mapping of column names to new values.
        where : :class:`Column` | "Any" | None, optional
            An additional boolean filter. Defaults to `None`.

        Returns
        -------
        The compiled UPDATE statement.

        Raises
        ------
        RuntimeError
            No filter was provided.
        """
        df = self.where(where) if where is not None else self

        if not df._where_clauses:
            raise RuntimeError(
                "Refusing to update without a filter. "
                "If you really want to update every row, pass where=lit(True)."
            )

        table, adapter = df._table_and_adapter()

        # Resolve set_ values — Col expressions need resolving against the
        # registry, and the results may need alias adaptation.  Bare strings
        # are treated as literal values, not column references.
        resolved_values: dict[str, "Any"] = {}
        for col_name, value in set_.items():
            if isinstance(value, Column):
                resolved = df._resolve(value)
                if adapter is not None:
                    resolved = adapter.traverse(resolved)  # type: ignore
                resolved_values[col_name] = resolved
            else:
                resolved_values[col_name] = value

        stmt = sa_update(table).values(resolved_values)
        for clause in df._where_clauses:
            stmt = stmt.where(adapter.traverse(clause) if adapter else clause)  # type: ignore

        return stmt

    @staticmethod
    def _format_show(rows: list[Row], n: int, truncate: int):
        """Print rows in PySpark-style table format."""
        if not rows:
            print("(empty DataFrame)")
            return

        keys = rows[0]._keys
        # Compute column widths
        str_rows = []
        for row in rows:
            str_rows.append([_trunc(str(row[k]), truncate) for k in keys])

        widths = [len(k) for k in keys]
        for sr in str_rows:
            for i, val in enumerate(sr):
                widths[i] = max(widths[i], len(val))

        def _fmt_row(values: list) -> str:
            cells = [v.ljust(widths[i]) for i, v in enumerate(values)]
            return "| " + " | ".join(cells) + " |"

        sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
        print(sep)
        print(_fmt_row(list(keys)))
        print(sep)
        for sr in str_rows:
            print(_fmt_row(sr))
        print(sep)
        if len(rows) == n:
            print(f"(showing first {n} rows)")

    def _table_and_adapter(self) -> tuple["Table", ClauseAdapter | None]:
        """Return the raw table and, if this :class:`_DataFrameBase` is aliased, a :class:`~sqlalchemy.sql.util.ClauseAdapter` that rewrites alias references back to the raw table.

        Returns
        -------
        tuple[:class:`~sqlalchemy.sql.Table`, :class:`~sqlalchemy.sql.util.ClauseAdapter` | None]
        """
        mapper = sa_inspect(self._model, raiseerr=True)
        table = mapper.local_table

        if self._alias_name is None:
            return table, None

        aliased_selectable = sa_inspect(self._sa_entity).selectable
        adapter = ClauseAdapter(
            table,
            equivalents={
                c: {table.c[c.key]} for c in aliased_selectable.c if c.key in table.c
            },
        )
        return table, adapter

    # -----
    # Inspection methods

    def explain(self, dialect: str | None = None) -> str:
        """Return the compiled SQL string for debugging.

        Parameters
        ----------
        dialect : str | None, optional
            If `None`, uses a generic compilation.  Pass `"mysql"`, `"sqlite"`, `"postgresql"`, etc. to see dialect-specific SQL. Defaults to `None`.

        Returns
        -------
        str
        """
        stmt = self._build_query()
        compile_kwargs = {"literal_binds": True}
        if dialect:
            from sqlalchemy.dialects import registry as dialect_registry

            dialect_cls = dialect_registry.load(dialect)
            compiled = stmt.compile(
                dialect=dialect_cls(), compile_kwargs=compile_kwargs
            )
        else:
            compiled = stmt.compile(compile_kwargs=compile_kwargs)
        return str(compiled)

    def printSchema(self):
        """Print the available columns and their types in a PySpark-style schema tree.

        The output lists each mapped column with its SQLAlchemy type and
        nullability, mirroring the format of PySpark's ``DataFrame.printSchema()``.
        """
        print(f"root (model: {self._model.__name__})")
        mapper = sa_inspect(self._model, raiseerr=True)
        for attr in mapper.column_attrs:
            for mapped_col in attr.columns:
                nullable = "nullable" if mapped_col.nullable else "not null"
                print(f" |-- {attr.key}: {mapped_col.type} ({nullable})")

    @property
    def columns(self) -> list[str]:
        """Return a sorted list of available column names.

        Returns
        -------
        list[str]
        """
        return sorted(self._registry.keys())

    def __repr__(self) -> str:
        alias_part = f" as '{self._alias_name}'" if self._alias_name else ""
        cols = ", ".join(self.columns[:8])
        suffix = ", …" if len(self.columns) > 8 else ""
        return (
            f"{type(self).__name__}[{self._model.__name__}{alias_part}]({cols}{suffix})"
        )


class DataFrame[T](_DataFrameBase[T]):
    """A PySpark-style DataFrame backed by a SQLAlchemy ORM model, using a synchronous :class:`~sqlalchemy.orm.Session`.

    All transformation methods return new :class:`DataFrame` instances (the
    original is never mutated).  The underlying SQL query is only built and
    executed when a terminal method is called (`collect`, `show`, `toPandas`,
    `count`, `first`).

    Parameters
    ----------
    session : :class:`~sqlalchemy.orm.Session`
        An active SQLAlchemy ORM session.
    model : type
        An SQLAlchemy ORM mapped class.
    """

    _session: Session

    def __init__(self, session: Session, model: type[T]):
        super().__init__(session, model)

    def collect(self) -> list[Row]:
        """Execute the query and return a list of :class:`Row` objects.

        Returns
        -------
        list[:class:`Row`]
        """
        stmt = self._build_query()
        result = self._session.execute(stmt)
        return self._rows_from_result(result)

    def show(self, n: int = 20, truncate: int = 30):
        """Print the first `n` rows in a PySpark-style table.

        Parameters
        ----------
        n : int, optional
            Maximum number of rows to display. Defaults to 20.
        truncate : int, optional
            Maximum width for each column before truncating with `…`. Defaults to 30.
        """
        rows = self.limit(n).collect()
        self._format_show(rows, n, truncate)

    def toPandas(self):
        """Execute the query and return a :class:`~pandas.DataFrame`.

        Returns
        -------
        :class:`~pandas.DataFrame`
        """
        import pandas as pd

        rows = self.collect()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([r.asDict() for r in rows])

    def to_pandas(self):
        """Execute the query and return a :class:`~pandas.DataFrame`.

        Alias for :meth:`DataFrame.toPandas`.

        Returns
        -------
        :class:`~pandas.DataFrame`
        """
        return self.toPandas()

    def count(self) -> int:
        """Return the number of rows.

        Returns
        -------
        int
        """
        stmt = sa_select(sa_func.count()).select_from(self._build_query().subquery())
        return self._session.execute(stmt).scalar_one()

    def first(self) -> Row | None:
        """Return the first row, or `None` if the :class:`DataFrame` is empty.

        Returns
        -------
        :class:`Row` | None
        """
        rows = self.limit(1).collect()
        return rows[0] if rows else None

    def take(self, n: int) -> list[Row]:
        """Return the first `n` rows as a list.

        Parameters
        ----------
        n : int
            The number of rows to return.

        Returns
        -------
        list[:class:`Row`]
        """
        return self.limit(n).collect()

    def delete(self, condition: "Any | None" = None) -> int:
        """Delete rows from the underlying table and return the number of rows deleted.

        Uses the accumulated `where` clauses (and an optional inline
        `condition`) to build a `DELETE` statement.  The deletion is flushed
        to the database but **not** committed — call
        :meth:`~sqlalchemy.orm.Session.commit` on the session when ready.

        Calling `delete` without "Any" filters will raise a :class:`RuntimeError`
        as a safety measure.  Pass `condition=lit(True)` to explicitly delete
        every row.

        Parameters
        ----------
        condition : :class:`Column` | "Any" | None, optional
            An additional boolean filter applied on top of "Any" existing
            `where` clauses.  Defaults to `None`.

        Returns
        -------
        int

        Raises
        ------
        RuntimeError
            No filter was provided (neither via `where` nor `condition`),
            which would delete every row in the table.
        """
        stmt = self._build_delete_stmt(condition)
        result = cast(CursorResult, self._session.execute(stmt))
        self._session.flush()
        return result.rowcount

    def update(
        self,
        set_: dict[str, "Any"],
        where: "Any | None" = None,
    ) -> int:
        """Update rows in the underlying table and return the number of rows updated.

        Uses the accumulated `where` clauses (and an optional inline `where`
        parameter) to build an `UPDATE ... SET ...` statement.  The update is
        flushed to the database but **not** committed — call
        :meth:`~sqlalchemy.orm.Session.commit` on the session when ready.

        Calling `update` without "Any" filters will raise a :class:`RuntimeError`
        as a safety measure.  Pass `where=lit(True)` to explicitly update
        every row.

        Parameters
        ----------
        set_ : dict[str, :class:`Column` | "Any"]
            A mapping of column names to new values.  Values can be
            :class:`Column` expressions (which reference other columns)
            or plain Python values.
        where : :class:`Column` | "Any" | None, optional
            An additional boolean filter applied on top of "Any" existing
            `where` clauses.  Defaults to `None`.

        Returns
        -------
        int

        Raises
        ------
        RuntimeError
            No filter was provided (neither via chained `where` calls nor the
            `where` parameter), which would update every row in the table.
        """
        stmt = self._build_update_stmt(set_, where)
        result = cast(CursorResult, self._session.execute(stmt))
        self._session.flush()
        return result.rowcount


class AsyncDataFrame[T](_DataFrameBase[T]):
    """A PySpark-style DataFrame backed by a SQLAlchemy ORM model, using an asynchronous :class:`~sqlalchemy.ext.asyncio.AsyncSession`.

    All transformation methods return new :class:`AsyncDataFrame` instances (the
    original is never mutated).  The underlying SQL query is only built and
    executed when a terminal method is called (`collect`, `show`, `toPandas`,
    `count`, `first`).

    Parameters
    ----------
    session : :class:`~sqlalchemy.ext.asyncio.AsyncSession`
        An active SQLAlchemy async session.
    model : type
        An SQLAlchemy ORM mapped class.
    """

    _session: AsyncSession

    def __init__(self, session: AsyncSession, model: type[T]):
        super().__init__(session, model)

    async def collect(self) -> list[Row]:
        """Execute the query and return a list of :class:`Row` objects.

        Returns
        -------
        list[:class:`Row`]
        """
        stmt = self._build_query()
        result = await self._session.execute(stmt)
        return self._rows_from_result(result)

    async def show(self, n: int = 20, truncate: int = 30):
        """Print the first `n` rows in a PySpark-style table.

        Parameters
        ----------
        n : int, optional
            Maximum number of rows to display. Defaults to 20.
        truncate : int, optional
            Maximum width for each column before truncating with `…`. Defaults to 30.
        """
        rows = await self.limit(n).collect()
        self._format_show(rows, n, truncate)

    async def toPandas(self):
        """Execute the query and return a :class:`~pandas.DataFrame`.

        Returns
        -------
        :class:`~pandas.DataFrame`
        """
        import pandas as pd

        rows = await self.collect()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([r.asDict() for r in rows])

    async def to_pandas(self):
        """Execute the query and return a :class:`~pandas.DataFrame`.

        Alias for :meth:`AsyncDataFrame.toPandas`.

        Returns
        -------
        :class:`~pandas.DataFrame`
        """
        return await self.toPandas()

    async def count(self) -> int:
        """Return the number of rows.

        Returns
        -------
        int
        """
        stmt = sa_select(sa_func.count()).select_from(self._build_query().subquery())
        return (await self._session.execute(stmt)).scalar_one()

    async def first(self) -> Row | None:
        """Return the first row, or `None` if the :class:`AsyncDataFrame` is empty.

        Returns
        -------
        :class:`Row` | None
        """
        rows = await self.limit(1).collect()
        return rows[0] if rows else None

    async def take(self, n: int) -> list[Row]:
        """Return the first `n` rows as a list.

        Parameters
        ----------
        n : int
            The number of rows to return.

        Returns
        -------
        list[:class:`Row`]
        """
        return await self.limit(n).collect()

    async def delete(self, condition: "Any | None" = None) -> int:
        """Delete rows from the underlying table and return the number of rows deleted.

        Uses the accumulated `where` clauses (and an optional inline
        `condition`) to build a `DELETE` statement.  The deletion is flushed
        to the database but **not** committed — call
        :meth:`~sqlalchemy.ext.asyncio.AsyncSession.commit` on the session when ready.

        Calling `delete` without "Any" filters will raise a :class:`RuntimeError`
        as a safety measure.  Pass `condition=lit(True)` to explicitly delete
        every row.

        Parameters
        ----------
        condition : :class:`Column` | "Any" | None, optional
            An additional boolean filter applied on top of "Any" existing
            `where` clauses.  Defaults to `None`.

        Returns
        -------
        int

        Raises
        ------
        RuntimeError
            No filter was provided (neither via `where` nor `condition`),
            which would delete every row in the table.
        """
        stmt = self._build_delete_stmt(condition)
        result = cast(CursorResult, await self._session.execute(stmt))
        await self._session.flush()
        return result.rowcount

    async def update(
        self,
        set_: dict[str, "Any"],
        where: "Any | None" = None,
    ) -> int:
        """Update rows in the underlying table and return the number of rows updated.

        Uses the accumulated `where` clauses (and an optional inline `where`
        parameter) to build an `UPDATE ... SET ...` statement.  The update is
        flushed to the database but **not** committed — call
        :meth:`~sqlalchemy.ext.asyncio.AsyncSession.commit` on the session when ready.

        Calling `update` without "Any" filters will raise a :class:`RuntimeError`
        as a safety measure.  Pass `where=lit(True)` to explicitly update
        every row.

        Parameters
        ----------
        set_ : dict[str, :class:`Column` | "Any"]
            A mapping of column names to new values.  Values can be
            :class:`Column` expressions (which reference other columns)
            or plain Python values.
        where : :class:`Column` | "Any" | None, optional
            An additional boolean filter applied on top of "Any" existing
            `where` clauses.  Defaults to `None`.

        Returns
        -------
        int

        Raises
        ------
        RuntimeError
            No filter was provided (neither via chained `where` calls nor the
            `where` parameter), which would update every row in the table.
        """
        stmt = self._build_update_stmt(set_, where)
        result = cast(CursorResult, await self._session.execute(stmt))
        await self._session.flush()
        return result.rowcount


DFT = TypeVar("DFT", bound=DataFrame | AsyncDataFrame)


def _maybe_register_label(registry: dict[str, "Any"], resolved: "SQLExpr"):
    """If `resolved` is a labelled expression, add it to `registry` under its label name."""
    if isinstance(resolved, Label):
        registry[resolved.name] = resolved


def _trunc(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def table[T](session: Session, model: type[T]) -> DataFrame[T]:
    """Create a :class:`DataFrame` from an ORM model. Convenience alias for the constructor.

    Parameters
    ----------
    session : :class:`~sqlalchemy.orm.Session`
        An active SQLAlchemy ORM session.
    model : type
        A SQLAlchemy ORM mapped class.

    Returns
    -------
    :class:`DataFrame`
    """
    return DataFrame(session, model)


def async_table[T](session: AsyncSession, model: type[T]) -> AsyncDataFrame[T]:
    """Create an :class:`AsyncDataFrame` from an ORM model. Convenience alias for the constructor.

    Parameters
    ----------
    session : :class:`~sqlalchemy.ext.asyncio.AsyncSession`
        An active SQLAlchemy async session.
    model : type
        A SQLAlchemy ORM mapped class.

    Returns
    -------
    :class:`AsyncDataFrame`
    """
    return AsyncDataFrame(session, model)
