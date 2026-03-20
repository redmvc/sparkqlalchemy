"""
tests/test_sparkqlalchemy.py — Full integration tests for the sparkqlalchemy module.

Uses an in-memory SQLite database to verify every DataFrame operation
translates correctly to SQLAlchemy and returns the expected results.

Run with::

    pytest tests/sync_tests.py -v
"""

from __future__ import annotations

import os
import sqlite3
import sys
from datetime import date

import pytest
from sqlalchemy import Column, Date, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sparkqlalchemy import DataFrame, Row
from src.sparkqlalchemy import functions as F

# Detect any_value() aggregate availability (added in SQLite 3.44, absent from some builds)
try:
    sqlite3.connect(":memory:").execute("SELECT any_value(1)")
    _sqlite_has_any_value = True
except sqlite3.OperationalError:
    _sqlite_has_any_value = False

# -----
# ORM Models


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
    bonus = Column(Float, nullable=True)
    hire_date = Column(Date, nullable=True)
    review_date = Column(Date, nullable=True)
    status = Column(String(20), default="active")


# -----
# Fixtures


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    s = Session(engine)

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
                bonus=15_000,
                hire_date=date(2020, 3, 15),
                review_date=date(2024, 6, 1),
                status="active",
            ),
            Employee(
                id=2,
                first_name="Bob",
                last_name="Jones",
                department_id=1,
                salary=110_000,
                bonus=8_000,
                hire_date=date(2021, 7, 1),
                review_date=date(2023, 12, 15),
                status="active",
            ),
            Employee(
                id=3,
                first_name="Charlie",
                last_name="Brown",
                department_id=2,
                salary=95_000,
                bonus=None,
                hire_date=date(2019, 1, 10),
                review_date=None,
                status="active",
            ),
            Employee(
                id=4,
                first_name="Diana",
                last_name="Prince",
                department_id=2,
                salary=105_000,
                bonus=12_000,
                hire_date=date(2022, 11, 20),
                review_date=date(2025, 1, 10),
                status="inactive",
            ),
            Employee(
                id=5,
                first_name="Eve",
                last_name="Adams",
                department_id=3,
                salary=85_000,
                bonus=None,
                hire_date=date(2023, 5, 5),
                review_date=None,
                status="active",
            ),
            Employee(
                id=6,
                first_name="Frank",
                last_name="Miller",
                department_id=1,
                salary=130_000,
                bonus=20_000,
                hire_date=date(2018, 9, 1),
                review_date=date(2024, 3, 20),
                status="active",
            ),
        ]
    )
    s.commit()
    return s


# -----
# Tests


class TestWhere:
    def test_simple_comparison(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("salary") > 100_000).collect()
        assert len(rows) == 4  # Alice, Bob, Diana, Frank

    def test_compound_condition(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(
            (F.col("salary") > 100_000) & (F.col("status") == "active")
        ).collect()
        assert len(rows) == 3  # Alice, Bob, Frank

    def test_filter_alias(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.filter(F.col("status") == "inactive").collect()
        assert len(rows) == 1

    def test_or_operator(self, session: Session):
        """| produces a SQL OR combining two column conditions."""
        df = DataFrame(session, Employee)
        rows = df.where(
            (F.col("salary") < 90_000) | (F.col("salary") > 120_000)
        ).collect()
        # Eve (85k) and Frank (130k)
        assert len(rows) == 2
        names = {r.first_name for r in rows}
        assert names == {"Eve", "Frank"}

    def test_not_operator(self, session: Session):
        """~ produces a SQL NOT wrapping a column condition."""
        df = DataFrame(session, Employee)
        rows = df.where(~(F.col("status") == "active")).collect()
        # Only Diana is inactive
        assert len(rows) == 1
        assert rows[0].first_name == "Diana"


class TestSelect:
    def test_projects_specified_columns(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.select("first_name", "salary").collect()

        assert len(rows) == 6
        assert set(rows[0].asDict().keys()) == {"first_name", "salary"}

    def test_with_col_expression(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                F.col("first_name"), (F.col("salary") * F.lit(12)).alias("annual")
            )
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].annual == 85_000 * 12


class TestGroupByAgg:
    def test_group_by_with_agg(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.groupBy("department_id")
            .agg(
                F.sum("salary").alias("total_salary"),
                F.count("*").alias("headcount"),
            )
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3

        # Engineering: 120k + 110k + 130k = 360k, 3 people
        assert rows[0].department_id == 1
        assert rows[0].total_salary == 360_000
        assert rows[0].headcount == 3

        # Sales: 95k + 105k = 200k, 2 people
        assert rows[1].total_salary == 200_000

        # HR: 85k, 1 person
        assert rows[2].headcount == 1

    def test_snake_case_alias(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.group_by("status").agg(F.count("*").alias("n")).collect()
        assert len(rows) == 2

    def test_ungrouped_agg(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.agg(
            F.sum("salary").alias("total"),
            F.avg("salary").alias("average"),
            F.min("salary").alias("lowest"),
            F.max("salary").alias("highest"),
            F.count("*").alias("n"),
        ).collect()

        assert len(rows) == 1
        row = rows[0]
        assert row.total == 645_000
        assert row.n == 6
        assert row.lowest == 85_000
        assert row.highest == 130_000

    def test_grouped_data_count_shortcut(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.groupBy("department_id").count().orderBy("department_id").collect()
        assert rows[0]["count"] == 3
        assert rows[1]["count"] == 2
        assert rows[2]["count"] == 1

    def test_select_withcolumn_after_group(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.groupBy("department_id", "status")
            .agg(F.max("salary").alias("max_salary"))
            .orderBy("department_id", "status")
            .select("status", "max_salary")
            .withColumn("high_salary", F.col("max_salary") > F.lit(100_000))
            .collect()
        )

        assert len(rows) == 4
        assert set(rows[0].asDict().keys()) == {"status", "max_salary", "high_salary"}

        assert rows[0]["status"] == "active"
        assert rows[0]["max_salary"] == 130_000
        assert rows[0]["high_salary"]

        assert rows[1]["status"] == "active"
        assert rows[1]["max_salary"] == 95_000
        assert not rows[1]["high_salary"]

        assert rows[2]["status"] == "inactive"
        assert rows[2]["max_salary"] == 105_000
        assert rows[2]["high_salary"]

        assert rows[3]["status"] == "active"
        assert rows[3]["max_salary"] == 85_000
        assert not rows[3]["high_salary"]

    def test_where_then_ungrouped_agg(self, session: Session):
        """where() before agg() filters rows before the aggregate is computed."""
        rows = (
            DataFrame(session, Employee)
            .where(F.col("status") == "active")
            .agg(
                F.sum("salary").alias("total"),
                F.count("*").alias("n"),
            )
            .collect()
        )
        assert len(rows) == 1
        assert rows[0].n == 5
        assert rows[0].total == 540_000  # 120k+110k+95k+85k+130k

    def test_select_then_groupby(self, session: Session):
        """select() before groupBy() restricts the projection before grouping."""
        rows = (
            DataFrame(session, Employee)
            .select("department_id", "salary")
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total"))
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].total == 360_000
        assert rows[1].total == 200_000
        assert rows[2].total == 85_000

    def test_chained_where_clauses(self, session: Session):
        """Multiple .where() calls are combined with AND."""
        rows = (
            DataFrame(session, Employee)
            .where(F.col("status") == "active")
            .where(F.col("salary") > 100_000)
            .collect()
        )
        assert len(rows) == 3  # Alice (120k), Bob (110k), Frank (130k)

    def test_select_then_where_on_unprojected_column(self, session: Session):
        """WHERE can filter by a column not present in the SELECT list."""
        rows = (
            DataFrame(session, Employee)
            .select("first_name", "salary")
            .where(F.col("status") == "active")
            .orderBy("salary")
            .collect()
        )
        assert len(rows) == 5
        assert set(rows[0].asDict().keys()) == {"first_name", "salary"}
        assert rows[0].first_name == "Eve"  # lowest active salary

    def test_grouped_data_sum(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .sum("salary")
            .orderBy("department_id")
            .collect()
        )
        assert rows[0]["sum(salary)"] == 360_000
        assert rows[1]["sum(salary)"] == 200_000
        assert rows[2]["sum(salary)"] == 85_000

    def test_grouped_data_avg(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .avg("salary")
            .orderBy("department_id")
            .collect()
        )
        assert rows[0]["avg(salary)"] == pytest.approx(120_000)
        assert rows[1]["avg(salary)"] == pytest.approx(100_000)
        assert rows[2]["avg(salary)"] == pytest.approx(85_000)

    def test_grouped_data_mean(self, session: Session):
        """mean() is an alias for avg() and produces the same column label."""
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .mean("salary")
            .orderBy("department_id")
            .collect()
        )
        # mean delegates to avg, so the label is avg(salary) — matches PySpark behaviour
        assert rows[0]["avg(salary)"] == pytest.approx(120_000)
        assert rows[1]["avg(salary)"] == pytest.approx(100_000)
        assert rows[2]["avg(salary)"] == pytest.approx(85_000)

    def test_grouped_data_max(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .max("salary")
            .orderBy("department_id")
            .collect()
        )
        assert rows[0]["max(salary)"] == 130_000  # Frank
        assert rows[1]["max(salary)"] == 105_000  # Diana
        assert rows[2]["max(salary)"] == 85_000  # Eve

    def test_grouped_data_min(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .min("salary")
            .orderBy("department_id")
            .collect()
        )
        assert rows[0]["min(salary)"] == 110_000  # Bob
        assert rows[1]["min(salary)"] == 95_000  # Charlie
        assert rows[2]["min(salary)"] == 85_000  # Eve

    def test_consecutive_groupbys(self, session: Session):
        df = DataFrame(session, Employee)
        two_groupbys_df = (
            df.groupBy("department_id", "status")
            .agg(F.max("salary").alias("max_salary"))
            .groupBy("department_id")
            .agg(F.mean("max_salary").alias("mean_max_salary"))
        )
        two_groupbys_rows = two_groupbys_df.orderBy("department_id").collect()

        assert len(two_groupbys_rows) == 3
        assert set(two_groupbys_rows[0].asDict().keys()) == {
            "department_id",
            "mean_max_salary",
        }
        assert two_groupbys_rows[0]["mean_max_salary"] == 130_000
        assert two_groupbys_rows[1]["mean_max_salary"] == 100_000
        assert two_groupbys_rows[2]["mean_max_salary"] == 85_000

        three_groupbys_rows = (
            two_groupbys_df.withColumn(
                "high_salary", F.col("mean_max_salary") > 100_000
            )
            .groupBy("high_salary")
            .agg(
                F.min("mean_max_salary").alias("min_mean_max_salary"),
                F.max("mean_max_salary").alias("max_mean_max_salary"),
            )
            .orderBy("high_salary")
            .collect()
        )

        assert len(three_groupbys_rows) == 2
        assert set(three_groupbys_rows[0].asDict().keys()) == {
            "high_salary",
            "min_mean_max_salary",
            "max_mean_max_salary",
        }

        assert not three_groupbys_rows[0]["high_salary"]
        assert three_groupbys_rows[0]["min_mean_max_salary"] == 85_000
        assert three_groupbys_rows[0]["max_mean_max_salary"] == 100_000

        assert three_groupbys_rows[1]["high_salary"]
        assert three_groupbys_rows[1]["min_mean_max_salary"] == 130_000
        assert three_groupbys_rows[1]["max_mean_max_salary"] == 130_000


class TestHaving:
    def test_having_filters_groups(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.count("*").alias("n"))
            .having(F.col("n") > 1)
            .orderBy("department_id")
            .collect()
        )
        # Engineering=3, Sales=2 pass; HR=1 excluded
        assert len(rows) == 2
        assert rows[0].department_id == 1
        assert rows[0].n == 3
        assert rows[1].department_id == 2
        assert rows[1].n == 2

    def test_where_after_group_routes_to_having(self, session: Session):
        """.where() on a grouped DataFrame is silently re-routed to HAVING."""
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.count("*").alias("n"))
            .where(F.col("n") >= 2)
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 2
        assert {r.department_id for r in rows} == {1, 2}

    def test_where_before_groupby_and_having_after(self, session: Session):
        """Pre-group WHERE filters rows; HAVING filters the resulting groups."""
        rows = (
            DataFrame(session, Employee)
            .where(F.col("status") == "active")
            .groupBy("department_id")
            .agg(F.count("*").alias("n"))
            .having(F.col("n") > 1)
            .orderBy("department_id")
            .collect()
        )
        # Active: Alice, Bob, Frank (dept1=3), Charlie (dept2=1), Eve (dept3=1)
        # having(n > 1): only dept1 survives
        assert len(rows) == 1
        assert rows[0].department_id == 1
        assert rows[0].n == 3

    def test_multiple_chained_having_clauses(self, session: Session):
        """Multiple .having() calls are combined with AND."""
        rows = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(
                F.sum("salary").alias("total"),
                F.count("*").alias("n"),
            )
            .having(F.col("n") >= 2)
            .having(F.col("total") > 250_000)
            .orderBy("department_id")
            .collect()
        )
        # having(n >= 2): dept1 (360k, 3), dept2 (200k, 2)
        # having(total > 250_000): only dept1 survives
        assert len(rows) == 1
        assert rows[0].department_id == 1
        assert rows[0].total == 360_000


class TestOrderBy:
    def test_desc(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select("first_name", "salary").orderBy(F.col("salary").desc()).collect()
        )
        salaries = [r.salary for r in rows]
        assert salaries == sorted(salaries, reverse=True)

    def test_multiple_sort_keys(self, session: Session):
        rows = (
            DataFrame(session, Employee)
            .orderBy(F.col("department_id").asc(), F.col("salary").desc())
            .select("department_id", "first_name")
            .collect()
        )
        # Dept 1: Frank (130k) > Alice (120k) > Bob (110k)
        assert rows[0].first_name == "Frank"
        assert rows[1].first_name == "Alice"
        assert rows[2].first_name == "Bob"
        # Dept 2: Diana (105k) > Charlie (95k)
        assert rows[3].first_name == "Diana"
        assert rows[4].first_name == "Charlie"

    def test_asc_snake_case(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.select("first_name").order_by("first_name").collect()
        names = [r.first_name for r in rows]
        assert names == sorted(names)


class TestWithColumn:
    def test_adds_computed_column(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.withColumn("annual_bonus", F.col("salary") * F.lit(0.1))
            .select("first_name", "salary", "annual_bonus")
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert len(rows) == 1
        assert rows[0].annual_bonus == 12_000.0

    def test_column_usable_in_later_operations(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.withColumn("double_salary", F.col("salary") * F.lit(2))
            .select("first_name", "double_salary")
            .orderBy(F.col("double_salary").desc())
            .collect()
        )
        assert rows[0].double_salary == 260_000  # Frank: 130k * 2

    def test_replace_existing_column(self, session: Session):
        """withColumn() with an existing name replaces it rather than duplicating it."""
        rows = (
            DataFrame(session, Employee)
            .withColumn("salary", F.col("salary") + F.lit(5_000))
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].salary == 90_000  # 85k + 5k
        assert list(rows[0].asDict().keys()).count("salary") == 1

    def test_rename_nonexistent_column_raises(self, session: Session):
        df = DataFrame(session, Employee)
        with pytest.raises(KeyError):
            df.withColumnRenamed("nonexistent", "new_name")

    def test_rename(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.withColumnRenamed("first_name", "name")
            .select("name", "salary")
            .orderBy("salary")
            .collect()
        )
        assert hasattr(rows[0], "name")
        assert rows[0].name == "Eve"  # lowest salary


class TestJoin:
    def test_inner_join(self, session: Session):
        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], how="inner")
            .select("first_name", "name", "salary")
            .where(F.col("name") == "Engineering")
            .orderBy(F.col("salary").desc())
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].first_name == "Frank"
        assert rows[0].name == "Engineering"

    def test_left_join_preserves_nulls(self, session: Session):
        session.add(
            Employee(
                id=99, first_name="Ghost", last_name="X", department_id=None, salary=0
            )
        )
        session.commit()

        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], how="left")
            .select("first_name", "name")
            .orderBy("first_name")
            .collect()
        )
        ghost = [r for r in rows if r.first_name == "Ghost"][0]
        assert ghost.name is None

    def test_bracket_syntax(self, session: Session):
        emp = DataFrame(session, Employee)
        dept = DataFrame(session, Department)

        rows = (
            emp.join(dept, emp["department_id"] == dept["id"])
            .select("first_name", "name")
            .orderBy("first_name")
            .collect()
        )
        assert len(rows) == 6
        assert rows[0].first_name == "Alice"
        assert rows[0].name == "Engineering"

    def test_list_of_columns(self, session: Session):
        """Multi-key equi-join via a list of shared column names."""
        a = DataFrame(session, Employee)
        b = DataFrame(session, Employee)

        rows = a.join(b, ["department_id", "status"]).collect()
        # Dept 1 active: Alice, Bob, Frank → 3×3 = 9
        # Dept 2 active: Charlie → 1, inactive: Diana → 1
        # Dept 3 active: Eve → 1
        assert len(rows) == 12

    def test_inequality_join(self, session: Session):
        emp = DataFrame(session, Employee).alias("e")
        dept = DataFrame(session, Department).alias("d")

        rows = (
            emp.join(
                dept,
                (F.col("e.department_id") == F.col("d.id"))
                & (F.col("e.salary") > F.col("d.budget") / F.lit(5)),
                "inner",
            )
            .select("e.first_name", "e.salary", "d.name", "d.budget")
            .orderBy(F.col("e.salary").desc())
            .collect()
        )
        assert len(rows) == 6

    def test_self_join_with_aliases(self, session: Session):
        """Find employee pairs in the same dept where one earns more."""
        a = DataFrame(session, Employee).alias("a")
        b = DataFrame(session, Employee).alias("b")

        rows = (
            a.join(
                b,
                (F.col("a.department_id") == F.col("b.department_id"))
                & (F.col("a.salary") > F.col("b.salary")),
                "inner",
            )
            .select(
                F.col("a.first_name").alias("higher_earner"),
                F.col("b.first_name").alias("lower_earner"),
                F.col("a.department_id").alias("dept"),
            )
            .orderBy(F.col("higher_earner"))
            .collect()
        )
        assert len(rows) == 4
        names = [(r.higher_earner, r.lower_earner) for r in rows]
        assert ("Frank", "Alice") in names
        assert ("Frank", "Bob") in names
        assert ("Alice", "Bob") in names
        assert ("Diana", "Charlie") in names

    def test_right_join_preserves_unmatched_right_rows(self, session: Session):
        """RIGHT join keeps all rows from the right (joined) table, nulling left columns."""
        session.add(Department(id=4, name="Marketing", budget=50_000.0))
        session.commit()

        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], how="right")
            .select("first_name", "name")
            .orderBy("name")
            .collect()
        )

        dept_names = [r.name for r in rows]
        assert "Marketing" in dept_names
        marketing_row = next(r for r in rows if r.name == "Marketing")
        assert marketing_row.first_name is None

    @pytest.mark.skipif(
        sqlite3.sqlite_version_info < (3, 39, 0),
        reason="SQLite < 3.39 does not support FULL OUTER JOIN",
    )
    def test_full_outer_join_preserves_both_sides(self, session: Session):
        """FULL OUTER join returns all rows from both sides, nulling the missing half."""
        session.add(Department(id=4, name="Marketing", budget=50_000.0))
        session.add(
            Employee(
                id=99, first_name="Ghost", last_name="X", department_id=None, salary=0
            )
        )
        session.commit()

        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(
                dept_df,
                emp_df["department_id"] == dept_df["id"],
                how="full",
            )
            .select("first_name", "name")
            .collect()
        )

        first_names = [r.first_name for r in rows]
        dept_names = [r.name for r in rows]

        # Ghost has no department → name should be NULL on that row
        assert "Ghost" in first_names
        ghost_row = next(r for r in rows if r.first_name == "Ghost")
        assert ghost_row.name is None

        # Marketing has no employees → first_name should be NULL on that row
        assert "Marketing" in dept_names
        marketing_row = next(r for r in rows if r.name == "Marketing")
        assert marketing_row.first_name is None

    def test_cross_join(self, session: Session):
        """Cross join with no ON condition produces the full Cartesian product."""
        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        n = emp_df.join(dept_df, how="cross").count()
        # 6 employees × 3 departments = 18
        assert n == 18

    def test_three_way_join(self, session: Session):
        """Chaining two joins works and all three registries stay accessible."""
        e = DataFrame(session, Employee).alias("e")
        d = DataFrame(session, Department).alias("d")
        e2 = DataFrame(session, Employee).alias("e2")

        rows = (
            e.join(d, F.col("e.department_id") == F.col("d.id"), "inner")
            .join(e2, F.col("e.department_id") == F.col("e2.department_id"), "inner")
            .where(F.col("d.name") == "Engineering")
            .select(
                F.col("e.first_name").alias("emp"),
                F.col("e2.first_name").alias("colleague"),
            )
            .collect()
        )
        # Engineering has Alice, Bob, Frank (3 people) → 3 × 3 = 9 ordered pairs
        assert len(rows) == 9

    def test_join_left_grouped(self, session: Session):
        """Joining a grouped DF (left) with a plain DF (right) works."""
        emp_agg = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total_salary"))
        )
        dept_df = DataFrame(session, Department)

        rows = (
            emp_agg.join(
                dept_df,
                F.col("department_id") == dept_df["id"],
                "inner",
            )
            .select("name", "total_salary")
            .orderBy("total_salary")
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].name == "HR"
        assert rows[0].total_salary == 85_000
        assert rows[2].name == "Engineering"
        assert rows[2].total_salary == 360_000

    def test_join_right_grouped(self, session: Session):
        """Joining a plain DF (left) with a grouped DF (right) works."""
        dept_df = DataFrame(session, Department)
        emp_agg = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total_salary"))
        )

        rows = (
            dept_df.join(
                emp_agg,
                dept_df["id"] == F.col("department_id"),
                "inner",
            )
            .select("name", "total_salary")
            .orderBy("total_salary")
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].name == "HR"
        assert rows[0].total_salary == 85_000

    def test_join_both_grouped(self, session: Session):
        """Joining two grouped DFs works."""
        emp_count = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.count("*").alias("headcount"))
        )
        emp_salary = (
            DataFrame(session, Employee)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total_salary"))
        )

        rows = (
            emp_count.join(
                emp_salary,
                "department_id",
                "inner",
            )
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].headcount == 3
        assert rows[0].total_salary == 360_000
        assert rows[1].headcount == 2
        assert rows[1].total_salary == 200_000
        assert rows[2].headcount == 1
        assert rows[2].total_salary == 85_000

    def test_select_then_join(self, session: Session):
        """select() before join still includes right-side columns in the result."""
        emp_df = DataFrame(session, Employee).select("first_name", "department_id")
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], "inner")
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert len(rows) == 1
        keys = set(rows[0].asDict().keys())
        # Left side selected: first_name, department_id
        assert "first_name" in keys
        assert "department_id" in keys
        # Right side should be included automatically: id, name, budget
        assert "name" in keys
        assert "budget" in keys

    def test_select_then_join_both_selected(self, session: Session):
        """Both sides with select() before join — result has both projections."""
        emp_df = DataFrame(session, Employee).select("first_name", "department_id")
        dept_df = DataFrame(session, Department).select("id", "name")

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], "inner")
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert len(rows) == 1
        keys = set(rows[0].asDict().keys())
        assert keys == {"first_name", "department_id", "id", "name"}

    def test_three_way_join_without_explicit_select(self, session: Session):
        """Three tables joined with no select() — all columns from all three appear."""
        e = DataFrame(session, Employee).alias("e")
        d = DataFrame(session, Department).alias("d")
        e2 = DataFrame(session, Employee).alias("e2")

        rows = (
            e.join(d, F.col("e.department_id") == F.col("d.id"), "inner")
            .join(e2, F.col("e.department_id") == F.col("e2.department_id"), "inner")
            .where(
                (F.col("e.first_name") == "Alice") & (F.col("e2.first_name") == "Bob")
            )
            .collect()
        )
        assert len(rows) == 1
        keys = set(rows[0].asDict().keys())
        # Should have columns from all three registries
        assert "e.first_name" in keys or "first_name" in keys
        assert "d.name" in keys or "name" in keys

    def test_three_way_join_with_select_before_second_join(self, session: Session):
        """select() between joins — third table's columns still appear."""
        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], "inner")
            .select("first_name", "name")
            .join(
                DataFrame(session, Department).alias("d2"),
                emp_df["department_id"] == F.col("d2.id"),
                "inner",
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert len(rows) == 1
        keys = set(rows[0].asDict().keys())
        # Selected from first join: first_name, name
        assert "first_name" in keys
        assert "name" in keys
        # Third table (d2) columns should also appear
        assert "d2.name" in keys or "budget" in keys


class TestUnion:
    def test_union_combines_rows(self, session: Session):
        """union() appends all rows from both DataFrames (UNION ALL semantics)."""
        active_df = (
            DataFrame(session, Employee)
            .where(F.col("status") == "active")
            .select("first_name", "salary")
        )
        inactive_df = (
            DataFrame(session, Employee)
            .where(F.col("status") == "inactive")
            .select("first_name", "salary")
        )
        rows = active_df.union(inactive_df).collect()
        assert len(rows) == 6

    def test_union_preserves_duplicates(self, session: Session):
        """union() is UNION ALL — duplicate rows are kept, not deduplicated."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("department_id")
        )
        rows = dept1.union(dept1).collect()
        assert len(rows) == 6  # 3 rows × 2

    def test_union_count(self, session: Session):
        """count() on a union wraps it in a subquery and counts correctly."""
        dept1 = DataFrame(session, Employee).where(F.col("department_id") == 1)
        dept2 = DataFrame(session, Employee).where(F.col("department_id") == 2)
        assert dept1.union(dept2).count() == 5  # 3 + 2

    def test_three_way_union(self, session: Session):
        """Chaining union() three times combines all rows."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        dept3 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 3)
            .select("first_name", "salary")
        )
        rows = dept1.union(dept2).union(dept3).collect()
        assert len(rows) == 6  # 3 + 2 + 1

    def test_union_then_where(self, session: Session):
        """where() after union filters the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        rows = dept1.union(dept2).where(F.col("salary") > 100_000).collect()
        # Dept 1: Alice (120k), Bob (110k), Frank (130k); Dept 2: Diana (105k)
        assert len(rows) == 4

    def test_union_then_order_by(self, session: Session):
        """orderBy() after union sorts the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        rows = dept1.union(dept2).orderBy("salary").collect()
        salaries = [r.salary for r in rows]
        assert salaries == sorted(salaries)

    def test_union_then_select(self, session: Session):
        """select() after union projects the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        rows = dept1.union(dept2).select("first_name").collect()
        assert len(rows) == 5
        assert set(rows[0].asDict().keys()) == {"first_name"}

    def test_union_then_limit(self, session: Session):
        """limit() after union limits the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        rows = dept1.union(dept2).limit(3).collect()
        assert len(rows) == 3

    def test_union_then_group_by(self, session: Session):
        """groupBy() after union aggregates the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("department_id", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("department_id", "salary")
        )
        rows = (
            dept1.union(dept2)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total"), F.count("*").alias("n"))
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 2
        assert rows[0].department_id == 1
        assert rows[0].total == 360_000
        assert rows[0].n == 3
        assert rows[1].department_id == 2
        assert rows[1].total == 200_000
        assert rows[1].n == 2

    def test_group_by_then_union_then_group_by(self, session: Session):
        """Grouped DFs can be unioned and then re-grouped."""
        dept1_agg = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total"))
        )
        dept2_agg = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .groupBy("department_id")
            .agg(F.sum("salary").alias("total"))
        )
        rows = (
            dept1_agg.union(dept2_agg)
            .agg(F.sum("total").alias("grand_total"))
            .collect()
        )
        assert len(rows) == 1
        assert rows[0].grand_total == 560_000  # 360k + 200k

    def test_union_then_distinct(self, session: Session):
        """distinct() after union deduplicates the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("department_id")
        )
        # Union with itself produces 6 rows (3 + 3), distinct should collapse to 1
        rows = dept1.union(dept1).distinct().collect()
        assert len(rows) == 1
        assert rows[0].department_id == 1

    def test_union_then_with_column(self, session: Session):
        """withColumn() after union adds a computed column to the combined result."""
        dept1 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        dept2 = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )
        rows = (
            dept1.union(dept2)
            .withColumn("bonus", F.col("salary") * F.lit(0.1))
            .orderBy("first_name")
            .collect()
        )
        assert len(rows) == 5
        assert set(rows[0].asDict().keys()) == {"first_name", "salary", "bonus"}
        alice = next(r for r in rows if r.first_name == "Alice")
        assert alice.bonus == 12_000.0  # 120k * 0.1

    def test_union_does_not_mutate_inputs(self, session: Session):
        """union() does not mutate either input DataFrame."""
        a = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 1)
            .select("first_name", "salary")
        )
        b = (
            DataFrame(session, Employee)
            .where(F.col("department_id") == 2)
            .select("first_name", "salary")
        )

        a_cols_before = a.columns[:]
        b_cols_before = b.columns[:]

        _ = a.union(b)

        assert a.columns == a_cols_before
        assert b.columns == b_cols_before
        assert a.count() == 3  # unchanged
        assert b.count() == 2  # unchanged

    def test_union_with_different_projections(self, session: Session):
        """Union of DFs with matching column names but different source filters."""
        high_earners = (
            DataFrame(session, Employee)
            .where(F.col("salary") > 120_000)
            .select("first_name", "salary")
        )
        low_earners = (
            DataFrame(session, Employee)
            .where(F.col("salary") < 90_000)
            .select("first_name", "salary")
        )
        rows = high_earners.union(low_earners).orderBy("salary").collect()
        assert len(rows) == 2  # Eve (85k) and Frank (130k)
        assert rows[0].first_name == "Eve"
        assert rows[1].first_name == "Frank"


class TestDistinctLimitOffset:
    def test_distinct(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.select("status").distinct().collect()
        assert {r.status for r in rows} == {"active", "inactive"}

    def test_limit(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.orderBy("id").limit(3).collect()
        assert len(rows) == 3
        assert rows[0].id == 1

    def test_limit_offset(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.orderBy("id").limit(2).offset(2).collect()
        assert len(rows) == 2
        assert rows[0].id == 3


class TestTerminals:
    def test_count(self, session: Session):
        df = DataFrame(session, Employee)
        assert df.count() == 6
        assert df.where(F.col("status") == "active").count() == 5

    def test_first(self, session: Session):
        df = DataFrame(session, Employee)
        row = df.orderBy("salary").first()
        assert row is not None
        assert row.salary == 85_000

    def test_take(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.orderBy("id").take(2)
        assert len(rows) == 2

    def test_to_pandas(self, session: Session):
        pd = pytest.importorskip("pandas")
        df = DataFrame(session, Employee)
        pdf = df.select("first_name", "salary").toPandas()
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 6
        assert list(pdf.columns) == ["first_name", "salary"]

    def test_first_returns_none_when_empty(self, session: Session):
        df = DataFrame(session, Employee)
        assert df.where(F.col("salary") > 999_999).first() is None

    def test_explain(self, session: Session):
        df = DataFrame(session, Employee)
        sql = (
            df.select("first_name", "salary")
            .where(F.col("salary") > 100_000)
            .orderBy(F.col("salary").desc())
            .explain()
        )
        assert "first_name" in sql
        assert "salary" in sql
        assert "100000" in sql

    def test_show(self, session: Session, capsys: pytest.CaptureFixture[str]):
        df = DataFrame(session, Employee)
        df.select("first_name", "salary").orderBy("salary").show(n=3)
        captured = capsys.readouterr()
        assert "Eve" in captured.out
        assert "first_name" in captured.out

    def test_print_schema(self, session: Session, capsys: pytest.CaptureFixture[str]):
        df = DataFrame(session, Employee)
        df.printSchema()
        captured = capsys.readouterr()
        assert "Employee" in captured.out
        assert "first_name" in captured.out
        assert "salary" in captured.out

    def test_to_pandas_snake_case_alias(self, session: Session):
        """to_pandas() is an alias for toPandas() and returns the same result."""
        pd = pytest.importorskip("pandas")
        df = DataFrame(session, Employee)
        pdf = df.select("first_name", "salary").to_pandas()
        assert isinstance(pdf, pd.DataFrame)
        assert len(pdf) == 6


class TestColMethods:
    def test_isin(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").isin("Alice", "Bob")).collect()
        assert len(rows) == 2

    def test_is_not_null(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("status").isNotNull()).collect()
        assert len(rows) == 6

    def test_like(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").like("A%")).collect()
        assert len(rows) == 1  # Alice

    def test_between(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("salary").between(90_000, 115_000)).collect()
        assert len(rows) == 3  # Bob, Charlie, Diana

    def test_is_null(self, session: Session):
        session.add(
            Employee(
                id=99,
                first_name="Ghost",
                last_name="X",
                department_id=None,
                salary=0,
            )
        )
        session.commit()
        df = DataFrame(session, Employee)
        rows = df.where(F.col("department_id").isNull()).collect()
        assert len(rows) == 1
        assert rows[0].first_name == "Ghost"

    def test_isin_with_list_argument(self, session: Session):
        """isin() accepts a single list argument in addition to *args."""
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").isin(["Alice", "Bob"])).collect()
        assert len(rows) == 2

    def test_startswith(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").startswith("F")).collect()
        assert len(rows) == 1
        assert rows[0].first_name == "Frank"

    def test_endswith(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("last_name").endswith("s")).collect()
        assert len(rows) == 2  # Jones (Bob), Adams (Eve)

    def test_contains(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").contains("an")).collect()
        assert len(rows) == 2  # Fr-an-k, Di-an-a

    def test_ilike(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").ilike("ALICE")).collect()
        assert len(rows) == 1
        assert rows[0].first_name == "Alice"

    def test_cast(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(F.col("salary").cast(Integer()).alias("salary_int"))
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert rows[0].salary_int == 120_000

    def test_arithmetic(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                (F.col("salary") + F.lit(5000)).alias("raised"),
                (F.col("salary") * F.lit(12)).alias("annual_gross"),
            )
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].raised == 90_000
        assert rows[0].annual_gross == 85_000 * 12

    def test_rlike(self, session: Session):
        """rlike() filters rows whose column value matches a regular expression."""
        df = DataFrame(session, Employee)
        rows = df.where(F.col("first_name").rlike("^[AE]")).collect()
        assert len(rows) == 2  # Alice, Eve
        names = {r.first_name for r in rows}
        assert names == {"Alice", "Eve"}

    def test_between_with_column_bounds(self, session: Session):
        """between() accepts Column expressions as bounds, not just scalar values."""
        df = DataFrame(session, Employee)
        rows = (
            df.where(F.col("salary").between(F.lit(100_000), F.lit(125_000)))
            .orderBy("first_name")
            .collect()
        )
        # Alice 120k, Bob 110k, Diana 105k — all within [100k, 125k]
        assert len(rows) == 3
        names = {r.first_name for r in rows}
        assert names == {"Alice", "Bob", "Diana"}

    def test_modulo_operator(self, session: Session):
        """% computes the SQL modulo of a column and a scalar."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                (F.col("salary") % F.lit(20_000)).alias("rem"),
            )
            .where(F.col("first_name").isin("Alice", "Bob"))
            .orderBy("first_name")
            .collect()
        )
        assert rows[0].first_name == "Alice"
        assert rows[0].rem == 0  # 120_000 % 20_000
        assert rows[1].first_name == "Bob"
        assert rows[1].rem == 10_000  # 110_000 % 20_000

    def test_reverse_arithmetic_operators(self, session: Session):
        """__radd__, __rsub__, __rmul__ let a plain value appear on the left side."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                (200_000 - F.col("salary")).alias("gap"),
                (2 * F.col("salary")).alias("doubled"),
                (0 + F.col("salary")).alias("passthrough"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert rows[0].gap == 80_000  # 200k - 120k
        assert rows[0].doubled == 240_000  # 2 * 120k
        assert rows[0].passthrough == 120_000  # 0 + 120k


class TestFunctions:
    def test_count_distinct(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.agg(F.countDistinct("department_id").alias("n_depts")).collect()
        assert rows[0].n_depts == 3

    def test_scalar_functions(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                F.upper("first_name").alias("upper_name"),
                F.lower("last_name").alias("lower_name"),
                F.length("first_name").alias("name_len"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert rows[0].upper_name == "ALICE"
        assert rows[0].lower_name == "smith"
        assert rows[0].name_len == 5

    def test_coalesce(self, session: Session):
        df = DataFrame(session, Employee)
        rows = df.select(
            F.coalesce("status", F.lit("unknown")).alias("safe_status")
        ).collect()
        assert all(r.safe_status is not None for r in rows)

    def test_when_otherwise(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.when(F.col("salary") >= 120_000, F.lit("high"))
                .when(F.col("salary") >= 100_000, F.lit("mid"))
                .otherwise(F.lit("low"))
                .alias("band"),
            )
            .orderBy("first_name")
            .collect()
        )
        bands = {r.first_name: r.band for r in rows}
        assert bands["Frank"] == "high"
        assert bands["Alice"] == "high"
        assert bands["Bob"] == "mid"
        assert bands["Eve"] == "low"

    def test_concat(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                F.concat("first_name", F.lit(" "), "last_name").alias("full_name")
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        assert rows[0].full_name == "Alice Smith"

    def test_trim(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(F.trim("first_name").alias("name"))
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].name == "Eve"

    def test_abs(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(F.abs(F.col("salary") - F.lit(100_000)).alias("diff"))
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].diff == 15_000  # |85k - 100k|

    def test_round(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.select(F.round(F.col("salary") / F.lit(3), 2).alias("r"))
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        assert rows[0].r == pytest.approx(28_333.33, rel=1e-3)  # 85000 / 3

    def test_max_by(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.groupBy("department_id")
            .agg(F.max_by("first_name", "salary").alias("top_earner"))
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3
        # Eng (dept 1): Frank earns most (130k)
        assert rows[0].top_earner == "Frank"
        # Sales (dept 2): Diana earns most (105k)
        assert rows[1].top_earner == "Diana"
        # HR (dept 3): Eve only person (85k)
        assert rows[2].top_earner == "Eve"

    def test_min_by(self, session: Session):
        df = DataFrame(session, Employee)
        rows = (
            df.groupBy("department_id")
            .agg(F.min_by("first_name", "salary").alias("lowest_earner"))
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3
        # Eng (dept 1): Bob earns least (110k)
        assert rows[0].lowest_earner == "Bob"
        # Sales (dept 2): Charlie earns least (95k)
        assert rows[1].lowest_earner == "Charlie"
        # HR (dept 3): Eve only person
        assert rows[2].lowest_earner == "Eve"

    def test_when_no_otherwise_returns_null(self, session: Session):
        """when() without otherwise() evaluates to NULL for non-matching rows."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.when(F.col("salary") >= 120_000, F.lit("high")).alias("band"),
            )
            .orderBy("first_name")
            .collect()
        )
        bands = {r.first_name: r.band for r in rows}
        assert bands["Alice"] == "high"  # 120k
        assert bands["Frank"] == "high"  # 130k
        assert bands["Bob"] is None  # 110k — no else branch → NULL

    @pytest.mark.skipif(
        not _sqlite_has_any_value,
        reason="SQLite build lacks any_value()",
    )
    def test_first_aggregate_function(self, session: Session):
        """F.first() inside agg() returns one non-null value per group."""
        df = DataFrame(session, Employee)
        rows = (
            df.groupBy("department_id")
            .agg(F.first("first_name").alias("any_name"))
            .orderBy("department_id")
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].any_name in {"Alice", "Bob", "Frank"}  # Engineering
        assert rows[1].any_name in {"Charlie", "Diana"}  # Sales
        assert rows[2].any_name == "Eve"  # HR

    def test_count_col_counts_non_nulls(self, session: Session):
        """count(col) counts non-null values; count() counts all rows."""
        session.add(
            Employee(
                id=99, first_name="Ghost", last_name="X", department_id=None, salary=0
            )
        )
        session.commit()

        df = DataFrame(session, Employee)
        rows_star = df.agg(F.count().alias("n_all")).collect()
        rows_col = df.agg(F.count("department_id").alias("n_with_dept")).collect()

        assert rows_star[0].n_all == 7  # all 7 rows
        assert rows_col[0].n_with_dept == 6  # Ghost has NULL department_id

    def test_count_distinct_snake_case_alias(self, session: Session):
        """count_distinct() is the snake_case alias for countDistinct()."""
        df = DataFrame(session, Employee)
        rows = df.agg(F.count_distinct("department_id").alias("n_depts")).collect()
        assert rows[0].n_depts == 3

    def test_greatest_numeric(self, session: Session):
        """greatest() returns the largest value among numeric columns."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("salary", "bonus").alias("higher"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        # Alice: salary=120k, bonus=15k → greatest = 120k
        assert rows[0].higher == 120_000

    def test_greatest_with_null(self, session: Session):
        """greatest() ignores NULL — returns the non-null value."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("salary", "bonus").alias("higher"),
            )
            .where(F.col("first_name") == "Charlie")
            .collect()
        )
        # Charlie: salary=95k, bonus=NULL → greatest = 95k
        assert rows[0].higher == 95_000

    def test_greatest_dates(self, session: Session):
        """greatest() works with date columns."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("hire_date", "review_date").alias("latest"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        # Alice: hire_date=2020-03-15, review_date=2024-06-01 → latest = 2024-06-01
        assert rows[0].latest == date(2024, 6, 1)

    def test_least_numeric(self, session: Session):
        """least() returns the smallest value among numeric columns."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("salary", "bonus").alias("lower"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        # Alice: salary=120k, bonus=15k → least = 15k
        assert rows[0].lower == 15_000

    def test_least_with_null(self, session: Session):
        """least() ignores NULL — returns the non-null value."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("salary", "bonus").alias("lower"),
            )
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        # Eve: salary=85k, bonus=NULL → least = 85k
        assert rows[0].lower == 85_000

    def test_least_dates(self, session: Session):
        """least() works with date columns."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("hire_date", "review_date").alias("earliest"),
            )
            .where(F.col("first_name") == "Bob")
            .collect()
        )
        # Bob: hire_date=2021-07-01, review_date=2023-12-15 → earliest = 2021-07-01
        assert rows[0].earliest == date(2021, 7, 1)

    def test_greatest_all_null(self, session: Session):
        """greatest() returns NULL when all inputs are NULL."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("bonus", "bonus").alias("result"),
            )
            .where(F.col("first_name") == "Charlie")
            .collect()
        )
        # Charlie: bonus=NULL, bonus=NULL → NULL
        assert rows[0].result is None

    def test_least_all_null(self, session: Session):
        """least() returns NULL when all inputs are NULL."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("bonus", "bonus").alias("result"),
            )
            .where(F.col("first_name") == "Charlie")
            .collect()
        )
        assert rows[0].result is None

    def test_greatest_three_columns(self, session: Session):
        """greatest() with three numeric arguments returns the largest."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("salary", "bonus", F.lit(100_000)).alias("top"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        # Alice: salary=120k, bonus=15k, lit=100k → 120k
        assert rows[0].top == 120_000

    def test_greatest_three_columns_with_nulls(self, session: Session):
        """greatest() with three columns skips NULLs and returns the max of the rest."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.greatest("salary", "bonus", F.lit(100_000)).alias("top"),
            )
            .where(F.col("first_name") == "Charlie")
            .collect()
        )
        # Charlie: salary=95k, bonus=NULL, lit=100k → 100k (NULL skipped)
        assert rows[0].top == 100_000

    def test_least_three_columns(self, session: Session):
        """least() with three numeric arguments returns the smallest."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("salary", "bonus", F.lit(100_000)).alias("bottom"),
            )
            .where(F.col("first_name") == "Alice")
            .collect()
        )
        # Alice: salary=120k, bonus=15k, lit=100k → 15k
        assert rows[0].bottom == 15_000

    def test_least_three_columns_with_nulls(self, session: Session):
        """least() with three columns skips NULLs and returns the min of the rest."""
        df = DataFrame(session, Employee)
        rows = (
            df.select(
                "first_name",
                F.least("salary", "bonus", F.lit(100_000)).alias("bottom"),
            )
            .where(F.col("first_name") == "Eve")
            .collect()
        )
        # Eve: salary=85k, bonus=NULL, lit=100k → 85k (NULL skipped)
        assert rows[0].bottom == 85_000


class TestAlias:
    def test_basic_dot_notation(self, session: Session):
        df = DataFrame(session, Employee).alias("e")

        assert "salary" in df.columns
        assert "e.salary" in df.columns
        assert "e.first_name" in df.columns

        rows = (
            df.select("e.first_name", "e.salary")
            .where(F.col("e.salary") > 100_000)
            .orderBy(F.col("e.salary").desc())
            .collect()
        )
        assert len(rows) == 4
        assert rows[0].first_name == "Frank"

    def test_join_with_dot_notation(self, session: Session):
        emp = DataFrame(session, Employee).alias("e")
        dept = DataFrame(session, Department).alias("d")

        rows = (
            emp.join(dept, F.col("e.department_id") == F.col("d.id"), "inner")
            .select("e.first_name", "d.name", "e.salary")
            .where(F.col("d.name") == "Engineering")
            .orderBy(F.col("e.salary").desc())
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].first_name == "Frank"
        assert rows[0].name == "Engineering"

    def test_chained_alias(self, session: Session):
        """df.alias('a').alias('b') should fully replace 'a' with 'b'."""
        df = DataFrame(session, Employee).alias("a").alias("b")

        assert "b.first_name" in df.columns
        assert "b.salary" in df.columns
        assert "a.first_name" not in df.columns
        assert "a.salary" not in df.columns
        assert "first_name" in df.columns  # bare names still work

        rows = (
            df.select("b.first_name", "b.salary")
            .where(F.col("b.salary") > 100_000)
            .orderBy(F.col("b.salary").desc())
            .collect()
        )
        assert len(rows) == 4
        assert rows[0].first_name == "Frank"

    def test_alias_after_join(self, session: Session):
        """Aliasing after a join preserves the joined columns."""
        emp = DataFrame(session, Employee)
        dept = DataFrame(session, Department)

        joined = emp.join(dept, emp["department_id"] == dept["id"])
        assert "first_name" in joined.columns
        assert "name" in joined.columns

        aliased_df = joined.alias("e")
        assert "e.first_name" in aliased_df.columns
        assert "e.salary" in aliased_df.columns
        assert "name" in aliased_df.columns  # from Department
        assert "budget" in aliased_df.columns  # from Department

        rows = (
            aliased_df.select("e.first_name", "name", "e.salary")
            .where(F.col("name") == "Engineering")
            .orderBy(F.col("e.salary").desc())
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].first_name == "Frank"
        assert rows[0].name == "Engineering"

    def test_alias_after_with_column(self, session: Session):
        """Aliasing after withColumn preserves the computed column."""
        df = (
            DataFrame(session, Employee)
            .withColumn("bonus", F.col("salary") * F.lit(0.1))
            .alias("e")
        )
        assert "bonus" in df.columns
        assert "e.first_name" in df.columns

        rows = (
            df.select("e.first_name", "bonus")
            .where(F.col("e.first_name") == "Alice")
            .collect()
        )
        assert rows[0].bonus == 12_000.0


class TestDelete:
    def test_delete_with_where(self, session: Session):
        df = DataFrame(session, Employee)
        assert df.count() == 6

        deleted = df.where(F.col("status") == "inactive").delete()
        session.commit()

        assert deleted == 1  # Diana
        assert df.count() == 5
        remaining = {r.first_name for r in df.collect()}
        assert "Diana" not in remaining

    def test_delete_with_inline_condition(self, session: Session):
        df = DataFrame(session, Employee)
        deleted = df.delete(F.col("salary") < 100_000)
        session.commit()

        assert deleted == 2  # Charlie (95k), Eve (85k)
        assert df.count() == 4

    def test_delete_compound_condition(self, session: Session):
        df = DataFrame(session, Employee)
        deleted = df.where(F.col("department_id") == 1).delete(
            F.col("salary") < 120_000
        )
        session.commit()

        assert deleted == 1  # Bob (110k in Engineering)
        assert df.count() == 5

    def test_delete_all_requires_explicit_flag(self, session: Session):
        df = DataFrame(session, Employee)
        with pytest.raises(RuntimeError, match="Refusing to delete"):
            df.delete()

    def test_delete_all_with_lit_true(self, session: Session):
        df = DataFrame(session, Employee)
        deleted = df.delete(F.lit(True))
        session.commit()

        assert deleted == 6
        assert df.count() == 0

    def test_delete_no_matches(self, session: Session):
        df = DataFrame(session, Employee)
        deleted = df.where(F.col("salary") > 999_999).delete()
        session.commit()

        assert deleted == 0
        assert df.count() == 6

    def test_delete_does_not_mutate_original(self, session: Session):
        df = DataFrame(session, Employee)
        filtered = df.where(F.col("status") == "inactive")

        filtered.delete()
        session.commit()

        # Original df reference should still have no where clauses
        assert df._where_clauses == []

    def test_delete_with_alias(self, session: Session):
        df = DataFrame(session, Employee).alias("e")
        deleted = df.where(F.col("e.salary") < 100_000).delete()
        session.commit()

        assert deleted == 2  # Charlie (95k), Eve (85k)
        assert DataFrame(session, Employee).count() == 4


class TestUpdate:
    def test_update_with_literal_value(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.update(
            set_={"status": "retired"},
            where=F.col("first_name") == "Eve",
        )
        session.commit()

        assert updated == 1
        eve = df.where(F.col("first_name") == "Eve").first()
        assert eve is not None
        assert eve.status == "retired"

    def test_update_with_expression(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.update(
            set_={"salary": F.col("salary") * F.lit(1.1)},
            where=F.col("department_id") == 1,
        )
        session.commit()

        assert updated == 3  # Alice, Bob, Frank
        frank = df.where(F.col("first_name") == "Frank").first()
        assert frank is not None
        assert frank.salary == pytest.approx(130_000 * 1.1)

    def test_update_with_chained_where_only(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.where(F.col("first_name") == "Eve").update(
            set_={"salary": F.lit(0)}
        )
        session.commit()

        assert updated == 1
        eve = df.where(F.col("first_name") == "Eve").first()
        assert eve is not None
        assert eve.salary == 0

    def test_update_with_chained_where(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.where(F.col("department_id") == 1).update(
            set_={"salary": F.lit(999)},
            where=(F.col("first_name") == "Bob"),
        )
        session.commit()

        assert updated == 1
        bob = df.where(F.col("first_name") == "Bob").first()
        assert bob is not None
        assert bob.salary == 999
        # Alice unchanged
        alice = df.where(F.col("first_name") == "Alice").first()
        assert alice is not None
        assert alice.salary == 120_000

    def test_update_multiple_columns(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.update(
            set_={"salary": F.lit(0), "status": "terminated"},
            where=F.col("first_name") == "Diana",
        )
        session.commit()

        assert updated == 1
        diana = df.where(F.col("first_name") == "Diana").first()
        assert diana is not None
        assert diana.salary == 0
        assert diana.status == "terminated"

    def test_update_no_matches(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.update(
            set_={"salary": F.lit(0)},
            where=F.col("salary") > 999_999,
        )
        session.commit()

        assert updated == 0
        assert df.where(F.col("salary") == 0).count() == 0

    def test_update_all_requires_explicit_flag(self, session: Session):
        df = DataFrame(session, Employee)
        with pytest.raises(RuntimeError, match="Refusing to update"):
            df.update(set_={"status": "active"})

    def test_update_all_with_lit_true(self, session: Session):
        df = DataFrame(session, Employee)
        updated = df.update(set_={"status": "reviewed"}, where=F.lit(True))
        session.commit()

        assert updated == 6
        assert df.where(F.col("status") == "reviewed").count() == 6

    def test_update_with_alias(self, session: Session):
        df = DataFrame(session, Employee).alias("e")
        updated = df.update(
            set_={"salary": F.col("e.salary") + F.lit(5000)},
            where=F.col("e.first_name") == "Eve",
        )
        session.commit()

        assert updated == 1
        eve = DataFrame(session, Employee).where(F.col("first_name") == "Eve").first()
        assert eve is not None
        assert eve.salary == 90_000


class TestComplex:
    def test_full_pipeline(self, session: Session):
        emp_df = DataFrame(session, Employee)
        dept_df = DataFrame(session, Department)

        rows = (
            emp_df.join(dept_df, emp_df["department_id"] == dept_df["id"], "inner")
            .where(F.col("status") == "active")
            .withColumn("bonus", F.col("salary") * F.lit(0.15))
            .select("first_name", "name", "salary", "bonus")
            .orderBy(F.col("bonus").desc())
            .limit(3)
            .collect()
        )
        assert len(rows) == 3
        assert rows[0].first_name == "Frank"
        assert rows[0].bonus == 130_000 * 0.15
        assert rows[0].name == "Engineering"


class TestRow:
    def test_attribute_access(self):
        row = Row({"name": "Alice", "age": 30, "city": "NYC"})
        assert row.name == "Alice"
        assert row.age == 30

    def test_index_access(self):
        row = Row({"name": "Alice", "age": 30})
        assert row[0] == "Alice"
        assert row[1] == 30

    def test_dict_access(self):
        row = Row({"name": "Alice", "age": 30})
        assert row["name"] == "Alice"
        assert row.asDict() == {"name": "Alice", "age": 30}

    def test_len(self):
        row = Row({"a": 1, "b": 2, "c": 3})
        assert len(row) == 3

    def test_equality(self):
        r1 = Row({"x": 1})
        r2 = Row({"x": 1})
        r3 = Row({"x": 2})
        assert r1 == r2
        assert r1 != r3


class TestMisc:
    def test_immutability(self, session: Session):
        df = DataFrame(session, Employee)
        _ = df.where(F.col("salary") > 100_000)
        _ = df.select("first_name")

        assert df._where_clauses == []
        assert df._select_entities is None

    def test_columns_property(self, session: Session):
        df = DataFrame(session, Employee)
        assert "first_name" in df.columns
        assert "salary" in df.columns

    def test_getitem_nonexistent_column_raises(self, session: Session):
        df = DataFrame(session, Employee)
        with pytest.raises(KeyError):
            _ = df["nonexistent"]

    def test_repr(self, session: Session):
        df = DataFrame(session, Employee)
        assert "Employee" in repr(df)

        aliased_df = df.alias("e")
        assert "e" in repr(aliased_df)
