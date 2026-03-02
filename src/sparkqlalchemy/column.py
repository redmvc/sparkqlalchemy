from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy import and_, case, cast, not_, or_

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, Literal

    from sqlalchemy import ColumnElement
    from sqlalchemy.orm.attributes import InstrumentedAttribute

    SQLExpr = ColumnElement[Any] | InstrumentedAttribute[Any]


@runtime_checkable
class _ColumnRegistry(Protocol):
    """A mapping-like object that resolves column names to SQLAlchemy expressions."""

    def __getitem__(self, key: str) -> "SQLExpr": ...
    def __contains__(self, key: object) -> bool: ...
    def keys(self) -> "Iterable[str]": ...


class _ResolvingRegistry:
    """A thin wrapper around a column registry that auto-resolves :class:`Column` entries during look-ups."""

    def __init__(self, raw: dict[str, "Column | SQLExpr"]):
        self._raw = raw

    # The resolver lambdas inside Col call  registry[name], so we need
    # __getitem__ to return concrete SA expressions.
    def __getitem__(self, key: str) -> "SQLExpr":
        entry = self._raw[key]
        if isinstance(entry, Column):
            return entry.resolve(self)
        return entry

    def __contains__(self, key: object) -> bool:
        return key in self._raw

    def keys(self) -> "Iterable[str]":
        return self._raw.keys()


class Column:
    """A lazily-resolved column expression, analogous to :class:`~pyspark.sql.Column`.

    Internally, every :class:`Column` stores a resolver callable that accepts a column registry and returns a concrete SQLAlchemy expression. Arithmetic, comparison, and boolean operators produce new :class:`Column` instances whose resolvers compose the operands' resolvers, so the full expression tree is built only at resolution time.
    """

    def __init__(
        self,
        resolver: "Callable[[_ColumnRegistry], SQLExpr]",
        name: str,
    ):
        self._resolver = resolver
        self._name = name

    @staticmethod
    def _wrap(expr: "SQLExpr", name: str | None = None) -> "Column":
        """Wrap a concrete SQLAlchemy expression in a :class:`Column`."""
        n = (
            name
            or getattr(expr, "key", None)
            or getattr(expr, "name", None)
            or str(expr)
        )
        return Column(lambda _, _e=expr: _e, n)

    @staticmethod
    def _deferred(name: str) -> "Column":
        """Create a deferred column reference that is resolved by name at query-build time."""

        def _resolver(reg: _ColumnRegistry) -> "SQLExpr":
            if name not in reg:
                available = sorted(reg.keys()) if hasattr(reg, "keys") else "?"
                raise KeyError(
                    f"Column '{name}' not found. Available columns: {available}"
                )
            return reg[name]

        return Column(_resolver, name)

    def resolve(
        self,
        registry: _ColumnRegistry | dict[str, "Column | SQLExpr"],
    ) -> "SQLExpr":
        """Resolve this column to a concrete SQLAlchemy expression using the given registry."""
        if not isinstance(registry, _ResolvingRegistry):
            registry = (
                _ResolvingRegistry(registry) if isinstance(registry, dict) else registry
            )
        return self._resolver(registry)

    # -----
    # Operators
    def __eq__(self, other: "Any") -> "Column":  # type: ignore
        return self._binop(other, lambda left, right: left == right, "==")

    def __ne__(self, other: "Any") -> "Column":  # type: ignore
        return self._binop(other, lambda left, right: left != right, "!=")

    def __gt__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left > right, ">")

    def __ge__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left >= right, ">=")

    def __lt__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left < right, "<")

    def __le__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left <= right, "<=")

    def __and__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: and_(left, right), "&")

    def __or__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: or_(left, right), "|")

    def __invert__(self) -> "Column":
        parent = self._resolver
        return Column(lambda reg: not_(parent(reg)), f"~{self._name}")

    def __add__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left + right, "+")

    def __radd__(self, other: "Any") -> "Column":
        return self._rbinop(other, lambda left, right: left + right, "+")

    def __sub__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left - right, "-")

    def __rsub__(self, other: "Any") -> "Column":
        return self._rbinop(other, lambda left, right: left - right, "-")

    def __mul__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left * right, "*")

    def __rmul__(self, other: "Any") -> "Column":
        return self._rbinop(other, lambda left, right: left * right, "*")

    def __truediv__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left / right, "/")

    def __mod__(self, other: "Any") -> "Column":
        return self._binop(other, lambda left, right: left % right, "%")

    def _binop(
        self,
        other: "Any",
        op: "Callable[[SQLExpr, Any], SQLExpr]",
        symbol: str,
    ) -> "Column":
        left = self._resolver
        if isinstance(other, Column):
            right = other._resolver
            rname = other._name
            return Column(
                lambda reg: op(left(reg), right(reg)),
                f"({self._name} {symbol} {rname})",
            )
        else:
            return Column(
                lambda reg, _o=other: op(left(reg), _o),
                f"({self._name} {symbol} {other!r})",
            )

    def _rbinop(
        self,
        other: "Any",
        op: "Callable[[Any, SQLExpr], SQLExpr]",
        symbol: str,
    ) -> "Column":
        right = self._resolver
        return Column(
            lambda reg, _o=other: op(_o, right(reg)),
            f"({other!r} {symbol} {self._name})",
        )

    # -----
    # PySpark-style methods

    def alias(self, name: str) -> "Column":
        """Assign a label to this column expression and return a new :class:`Column`.

        Parameters
        ----------
        name : str
            The label to assign.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).label(name), name)

    def asc(self) -> "Column":
        """Return a new :class:`Column` that sorts this expression in ascending order.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).asc(), self._name)

    def desc(self) -> "Column":
        """Return a new :class:`Column` that sorts this expression in descending order.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).desc(), self._name)

    def like(self, pattern: str) -> "Column":
        """Return a new :class:`Column` that applies a SQL ``LIKE`` filter.

        Parameters
        ----------
        pattern : str
            The ``LIKE`` pattern (use ``%`` and ``_`` as wildcards).

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).like(pattern), self._name)

    def ilike(self, pattern: str) -> "Column":
        """Return a new :class:`Column` that applies a case-insensitive SQL ``LIKE`` filter.

        Parameters
        ----------
        pattern : str
            The ``ILIKE`` pattern (use ``%`` and ``_`` as wildcards).

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).ilike(pattern), self._name)

    def rlike(self, pattern: str) -> "Column":
        """Return a new :class:`Column` that tests this expression against a regular expression.

        Parameters
        ----------
        pattern : str
            The regular expression pattern.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).regexp_match(pattern), self._name)

    def isin(self, *values: "Any") -> "Column":
        """Return a new :class:`Column` that tests whether this expression is in a set of values.

        Accepts either individual values or a single list/tuple/set argument.

        Parameters
        ----------
        *values : Any
            The values to test membership against.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        flat = (
            values[0]
            if len(values) == 1 and isinstance(values[0], (list, tuple, set))
            else values
        )
        return Column(lambda reg: parent(reg).in_(flat), self._name)

    def isNull(self) -> "Column":
        """Return a new :class:`Column` that tests whether this expression is ``NULL``.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).is_(None), self._name)

    def isNotNull(self) -> "Column":
        """Return a new :class:`Column` that tests whether this expression is not ``NULL``.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).isnot(None), self._name)

    def between(self, lower: "Any", upper: "Any") -> "Column":
        """Return a new :class:`Column` that tests whether this expression falls within an inclusive range.

        Parameters
        ----------
        lower : Any
            The lower bound (inclusive). May be a plain value or a :class:`Column`.
        upper : Any
            The upper bound (inclusive). May be a plain value or a :class:`Column`.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        lo = lower._resolver if isinstance(lower, Column) else lambda _: lower
        hi = upper._resolver if isinstance(upper, Column) else lambda _: upper
        return Column(lambda reg: parent(reg).between(lo(reg), hi(reg)), self._name)

    def cast(self, type_: "Any") -> "Column":
        """Cast this expression to a different SQL type and return a new :class:`Column`.

        Parameters
        ----------
        type_ : Any
            A SQLAlchemy type object (e.g. ``Integer()``, ``String()``).

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: cast(parent(reg), type_), self._name)

    def startswith(self, prefix: str) -> "Column":
        """Return a new :class:`Column` that tests whether this expression starts with `prefix`.

        Parameters
        ----------
        prefix : str
            The prefix string to match.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).startswith(prefix), self._name)

    def endswith(self, suffix: str) -> "Column":
        """Return a new :class:`Column` that tests whether this expression ends with `suffix`.

        Parameters
        ----------
        suffix : str
            The suffix string to match.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).endswith(suffix), self._name)

    def contains(self, substr: str) -> "Column":
        """Return a new :class:`Column` that tests whether this expression contains `substr`.

        Parameters
        ----------
        substr : str
            The substring to search for.

        Returns
        -------
        :class:`Column`
        """
        parent = self._resolver
        return Column(lambda reg: parent(reg).contains(substr), self._name)

    # -----
    # Display / safety

    def __repr__(self) -> str:
        return f"Col<{self._name}>"

    def __hash__(self) -> int:  # type: ignore[override]
        return id(self)

    def __bool__(self) -> bool:
        raise TypeError(
            "Cannot convert Col to bool. "
            "Use & (and), | (or), ~ (not) for boolean logic on columns."
        )


class WhenExpr(Column):
    """A builder for SQL `CASE WHEN ... THEN ... ELSE ...` expressions."""

    def __init__(self, condition: Column, value: "Any"):
        self._whens: list[tuple[Column, "Any"]] = [(condition, value)]
        self._else_value: "Any" = None

        def resolver(reg: _ColumnRegistry) -> "SQLExpr":
            clauses: list[tuple["SQLExpr", "Any"]] = []
            for cond_col, val in self._whens:
                resolved_cond = cond_col.resolve(reg)
                resolved_val = val.resolve(reg) if isinstance(val, Column) else val
                clauses.append((resolved_cond, resolved_val))
            kwargs: dict["Literal['else_']", "Any"] = {}
            if self._else_value is not None:
                else_val = (
                    self._else_value.resolve(reg)
                    if isinstance(self._else_value, Column)
                    else self._else_value
                )
                kwargs["else_"] = else_val
            return case(*clauses, **kwargs)

        super().__init__(resolver, "case")

    def when(self, condition: Column, value: "Any") -> "WhenExpr":
        """Add another ``WHEN … THEN …`` branch to this expression and return ``self``.

        Parameters
        ----------
        condition : :class:`Column`
            The boolean condition for the next ``WHEN`` clause.
        value : Any
            The value to return when `condition` is true.

        Returns
        -------
        :class:`WhenExpr`
        """
        self._whens.append((condition, value))
        return self

    def otherwise(self, value: "Any") -> "WhenExpr":
        """Set the ``ELSE`` branch of this expression and return ``self``.

        Parameters
        ----------
        value : Any
            The value to return when no ``WHEN`` condition matches.

        Returns
        -------
        :class:`WhenExpr`
        """
        self._else_value = value
        return self


def _to_col(c: str | Column) -> Column:
    """Coerce a string to a deferred :class:`Column` and pass :class:`Column` objects through unchanged."""
    if isinstance(c, str):
        return Column._deferred(c)
    return c
