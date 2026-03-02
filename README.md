# SparkQLAlchemy
PySpark-like DataFrame API backed by SQLAlchemy.

Provides a familiar PySpark-style chaining API that translates to [SQLAlchemy ORM Session](https://docs.sqlalchemy.org/en/20/orm/session.html) queries under the hood.  Each DataFrame operation returns a new immutable SparkQLAlchemy DataFrame, and the actual SQL is only executed when a terminal method (`collect`, `show`, `toPandas`, `count`, `first`) is called.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from sparkqlalchemy import DataFrame
from sparkqlalchemy import functions as F


class Base(DeclarativeBase): ...


class User(Base): ...


engine = create_engine(...)
Base.metadata.create_all(engine)
session = Session(engine)

df = DataFrame(session, User)
results = (
    df.select("name", "department", "salary")
    .where(
        (F.col("department") == F.lit("Engineering"))
        & (F.col("salary") > F.lit(80_000))
    )
    .groupBy("department")
    .agg(
        F.sum("salary").alias("total_salary"),
        F.count("*").alias("headcount"),
    )
    .orderBy(F.col("total_salary").desc())
    .collect()
)
```

This is a work in progress which I started working on because I wanted better querying syntax in my [Discord Channel Bridge Bot](https://github.com/redmvc/Discord-Channel-Bridge-Bot). It is not yet finished and does not implement every PySpark function. It might also have bugs or other issues that I haven't thoroughly tested yet.