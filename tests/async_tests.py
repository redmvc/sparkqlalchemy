"""
tests/async_tests.py — Integration tests for the AsyncDataFrame terminal methods.

The query-building logic lives in _DataFrameBase and is already exercised by
the sync test suite.  These tests verify that every *async* terminal method
correctly awaits the session and produces identical results to its sync
counterpart.

Run with::

    pytest tests/async_tests.py -v
"""

from __future__ import annotations

import os
import sys

import pytest
import pytest_asyncio
from sqlalchemy import Column, Float, ForeignKey, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sparkqlalchemy import AsyncDataFrame
from src.sparkqlalchemy import functions as F
from src.sparkqlalchemy.dataframe import async_table

# -----
# ORM Models (mirrors the sync test suite)


class Base(DeclarativeBase):
    pass


class Department(Base):
    __tablename__ = "departments"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    budget = Column(Float, default=0.0)


class Employee(Base):
    __tablename__ = "employees"
    id = Column(Integer, primary_key=True)
    first_name = Column(String(50), nullable=False)
    last_name = Column(String(50), nullable=False)
    department_id = Column(Integer, ForeignKey("departments.id"))
    salary = Column(Float, nullable=False)
    status = Column(String(20), default="active")


# -----
# Fixtures


@pytest_asyncio.fixture()
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine) as s:
        s.add_all(
            [
                Department(id=1, name="Engineering", budget=500_000),
                Department(id=2, name="Sales", budget=300_000),
                Department(id=3, name="HR", budget=150_000),
            ]
        )
        s.add_all(
            [
                Employee(
                    id=1,
                    first_name="Alice",
                    last_name="Smith",
                    department_id=1,
                    salary=120_000,
                    status="active",
                ),
                Employee(
                    id=2,
                    first_name="Bob",
                    last_name="Jones",
                    department_id=1,
                    salary=110_000,
                    status="active",
                ),
                Employee(
                    id=3,
                    first_name="Charlie",
                    last_name="Brown",
                    department_id=2,
                    salary=95_000,
                    status="active",
                ),
                Employee(
                    id=4,
                    first_name="Diana",
                    last_name="Prince",
                    department_id=2,
                    salary=105_000,
                    status="inactive",
                ),
                Employee(
                    id=5,
                    first_name="Eve",
                    last_name="Adams",
                    department_id=3,
                    salary=85_000,
                    status="active",
                ),
                Employee(
                    id=6,
                    first_name="Frank",
                    last_name="Miller",
                    department_id=1,
                    salary=130_000,
                    status="active",
                ),
            ],
        )
        await s.commit()
        yield s

    await engine.dispose()


# -----
# Tests


class TestAsyncCollect:
    @pytest.mark.asyncio
    async def test_collect_all(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        rows = await df.collect()
        assert len(rows) == 6

    @pytest.mark.asyncio
    async def test_collect_with_where(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        rows = await df.where(F.col("salary") > 100_000).collect()
        assert len(rows) == 4

    @pytest.mark.asyncio
    async def test_collect_with_select(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        rows = await df.select("first_name", "salary").collect()
        assert rows[0]._keys == ["first_name", "salary"]


class TestAsyncCount:
    @pytest.mark.asyncio
    async def test_count_all(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        assert await df.count() == 6

    @pytest.mark.asyncio
    async def test_count_filtered(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        assert await df.where(F.col("status") == "inactive").count() == 1


class TestAsyncFirst:
    @pytest.mark.asyncio
    async def test_first_returns_row(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee).orderBy("id")
        row = await df.first()
        assert row is not None
        assert row.first_name == "Alice"

    @pytest.mark.asyncio
    async def test_first_returns_none_when_empty(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee).where(F.col("salary") > 999_999)
        assert await df.first() is None


class TestAsyncTake:
    @pytest.mark.asyncio
    async def test_take(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee).orderBy("id")
        rows = await df.take(3)
        assert len(rows) == 3
        assert rows[0].first_name == "Alice"


class TestAsyncShow:
    @pytest.mark.asyncio
    async def test_show_does_not_raise(self, session: AsyncSession, capsys):
        df = AsyncDataFrame(session, Employee)
        await df.show(n=3)
        captured = capsys.readouterr()
        assert "+" in captured.out  # table border chars present


class TestAsyncToPandas:
    @pytest.mark.asyncio
    async def test_to_pandas(self, session: AsyncSession):
        pd = pytest.importorskip("pandas")
        df = AsyncDataFrame(session, Employee)
        pdf = await df.toPandas()
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 6

    @pytest.mark.asyncio
    async def test_to_pandas_snake_alias(self, session: AsyncSession):
        pytest.importorskip("pandas")
        df = AsyncDataFrame(session, Employee)
        pdf = await df.to_pandas()
        assert len(pdf) == 6

    @pytest.mark.asyncio
    async def test_to_pandas_empty(self, session: AsyncSession):
        pd = pytest.importorskip("pandas")
        df = AsyncDataFrame(session, Employee).where(F.col("salary") > 999_999)
        pdf = await df.toPandas()
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 0


class TestAsyncDelete:
    @pytest.mark.asyncio
    async def test_delete_with_condition(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        deleted = await df.delete(condition=F.col("status") == "inactive")
        assert deleted == 1
        assert await AsyncDataFrame(session, Employee).count() == 5

    @pytest.mark.asyncio
    async def test_delete_with_chained_where(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee).where(F.col("salary") < 90_000)
        deleted = await df.delete()
        assert deleted == 1  # Eve
        assert await AsyncDataFrame(session, Employee).count() == 5

    @pytest.mark.asyncio
    async def test_delete_without_filter_raises(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        with pytest.raises(RuntimeError, match="Refusing to delete"):
            await df.delete()


class TestAsyncUpdate:
    @pytest.mark.asyncio
    async def test_update_with_literal(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        updated = await df.update(
            set_={"status": "on_leave"},
            where=F.col("first_name") == "Alice",
        )
        assert updated == 1
        row = await (
            AsyncDataFrame(session, Employee)
            .where(F.col("first_name") == "Alice")
            .first()
        )
        assert row is not None
        assert row.status == "on_leave"

    @pytest.mark.asyncio
    async def test_update_without_filter_raises(self, session: AsyncSession):
        df = AsyncDataFrame(session, Employee)
        with pytest.raises(RuntimeError, match="Refusing to update"):
            await df.update(set_={"status": "x"})


class TestAsyncTableFactory:
    @pytest.mark.asyncio
    async def test_async_table_creates_async_dataframe(self, session: AsyncSession):
        df = async_table(session, Employee)
        assert isinstance(df, AsyncDataFrame)
        assert await df.count() == 6


class TestAsyncMethodChaining:
    """Verify that Self-returning transformation methods preserve AsyncDataFrame
    so that chained terminal calls work correctly."""

    @pytest.mark.asyncio
    async def test_chained_collect(self, session: AsyncSession):
        rows = await (
            AsyncDataFrame(session, Employee)
            .where(F.col("status") == "active")
            .orderBy("salary")
            .limit(2)
            .collect()
        )
        assert len(rows) == 2
        assert rows[0].first_name == "Eve"  # lowest salary among active

    @pytest.mark.asyncio
    async def test_groupby_agg_collect(self, session: AsyncSession):
        rows = await (
            AsyncDataFrame(session, Employee).groupBy("department_id").count().collect()
        )
        assert len(rows) == 3
