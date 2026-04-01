# Validate SQLModel on all Non-Database Sourced Data

As noted by other issues and pull requests, when setting `table=True` in a `SQLModel`, Pydantic validation does not run, and this breaks the contract that "[a **SQLModel** model is also a **Pydantic** model](https://sqlmodel.tiangolo.com/features/#based-on-pydantic)". This PR builds on prior ones and also hopes to address concerns with changing the intentional validation bypass for table models.

First, for those new to this issue, here is an example of how SQLModels behave differently when they are a `table`:

```python
from pydantic import BaseModel, ValidationError
from sqlmodel import SQLModel, Field


class HeroBase(BaseModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int


class HeroSQLBase(SQLModel):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int


class HeroSQLTable(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int


try:
    HeroBase(name="Deadpond", age="not an int")
    print("HeroBase: created with invalid data!")
except ValidationError:
    print("HeroBase: ValidationError raised")

try:
    HeroSQLBase(name="Deadpond", age="not an int")
    print("HeroSQLBase: created with invalid data!")
except ValidationError:
    print("HeroSQLBase: ValidationError raised")

try:
    HeroSQLTable(name="Deadpond", age="not an int")
    print("HeroSQLTable: created with invalid data!")
except ValidationError:
    print("HeroSQLTable: ValidationError raised")
```

```
HeroBase: ValidationError raised
HeroSQLBase: ValidationError raised
HeroSQLTable: created with invalid data!
```

The same is true for `@field_validator` and `@model_validator`. Both will be silently ignored when `table=True`.

### Use Case

I am building a library that includes a base SQLModel class (without `table=True`) that has validators to normalize and populate data. Downstream developers using my library would then create their own `table=True` class that **inherits** from the base:

```python
# ---- library code ----
import hashlib
import posixpath
import uuid

from pydantic import field_validator, model_validator
from sqlmodel import SQLModel, Field


class DocumentBase(SQLModel):
    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), max_length=256, primary_key=True
    )
    path: str
    content: str
    content_hash: str = ""

    @field_validator("path")
    @classmethod
    def normalize_path(cls, v: str) -> str:
        if not v.startswith("/"):
            v = "/" + v
        return posixpath.normpath(v)

    @model_validator(mode="after")
    def compute_content_hash(self) -> "DocumentBase":
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return self


# ---- downstream developer code ----
class Document(DocumentBase, table=True):
    project_field: str | None = None
```

The base class works correctly on its own, but it can't hold the downstream `project_field`:

```python
>>> Document(path="not_normalized/../path", content="hello", project_field="important info!")
DocumentBase(
    path='/path',                   # normalized
    content='hello',
    content_hash='2cf24db...',      # computed
  	# project_field missing!
)
```

But when a downstream developer inherits with `table=True`, both validators are silently skipped:

```python
>>> Document(path="not_normalized/../path", content="hello", project_field="important info!")
Document(
    path='not_normalized/../path',  # not normalized!
    content='hello',
    content_hash='',                # not computed!
    project_field='important info!',
)
```

My library must rely on validation from the custom defined inherited class, but it will not work out of the box. My issue could be resolved with the solution described in the [multiple models doc](https://sqlmodel.tiangolo.com/tutorial/fastapi/multiple-models/#multiple-models-with-duplicated-fields), but that approach would mean I would be asking developers to create twice as many classes, one for validation and one for table mapping. Using SQLModel was chosen for this project as it promised to provide a unified model between Pydantic and SQLAlchemy, which in my case, means any initialized model derived from `DocumentBase` is valid, both in python and in the database.

### The Change

The change is in `sqlmodel_init()` in `_compat.py`. Previously, table models called `sqlmodel_table_construct()` which skips validation entirely. Now, all models go through `validate_python()`, and table models do a post-validation step to re-trigger SQLAlchemy instrumentation via `setattr`:

```python
def sqlmodel_init(*, self: "SQLModel", data: dict[str, Any]) -> None:
    old_dict = self.__dict__.copy()
    self.__pydantic_validator__.validate_python(
        data,
        self_instance=self,
    )
    if not is_table_model_class(self.__class__):
        object.__setattr__(
            self,
            "__dict__",
            {**old_dict, **self.__dict__},
        )
    else:
        fields_set = self.__pydantic_fields_set__.copy()
        for key, value in {**old_dict, **self.__dict__}.items():
            setattr(self, key, value)
        object.__setattr__(self, "__pydantic_fields_set__", fields_set)
        for key in self.__sqlmodel_relationships__:
            value = data.get(key, Undefined)
            if value is not Undefined:
                setattr(self, key, value)
```

This mirrors the existing pattern used by `sqlmodel_validate()` (the `model_validate()` path) which already validates table models successfully.

### Addressing Prior Concerns

#### "SQLAlchemy needs to assign values after instantiation" (#52)

The concern raised in #52 was that relationships need to be assignable after construction, so validation can't run on `__init__`.

Relationships are not part of `model_fields` — they live in `__sqlmodel_relationships__` and are handled separately, outside of Pydantic validation. `validate_python()` never sees or validates relationship attributes. Both sides of a bidirectional relationship can be created independently, exactly as before:

```python
from sqlmodel import SQLModel, Field, Relationship, Session, create_engine


class Team(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    heroes: list["Hero"] = Relationship(back_populates="team")


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str
    team_id: int | None = Field(default=None, foreign_key="team.id")
    team: Team | None = Relationship(back_populates="heroes")


# Create each side independently — no relationship passed
team = Team(name="Preventers")
hero = Hero(name="Deadpond")

# Assign relationship after construction
hero.team = team

engine = create_engine("sqlite:///:memory:")
SQLModel.metadata.create_all(engine)

with Session(engine) as session:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    print(f"{hero.name}'s team: {hero.team.name}")
```

```
Deadpond's team: Preventers
```

#### Performance on ORM Reads

Validation does **not** run when loading from the database. SQLAlchemy does not call `__init__` when hydrating instances from query results ([SQLAlchemy docs: Constructors and Object Initialization](https://docs.sqlalchemy.org/en/20/orm/constructors.html)). This is unchanged, as it is safe to assume that data loaded from the database is valid.

To verify, here is a test that writes invalid data directly to the database and confirms it loads without triggering validation:

```python
from pydantic import field_validator
from sqlmodel import SQLModel, Field, Session, create_engine, select


class Hero(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str

    @field_validator("name")
    @classmethod
    def name_must_be_short(cls, v: str) -> str:
        if len(v) > 5:
            raise ValueError("too long")
        return v


engine = create_engine("sqlite:///:memory:")
SQLModel.metadata.create_all(engine)

# Insert valid data through the model
with Session(engine) as session:
    session.add(Hero(name="short"))
    session.commit()

# Write invalid data directly to the database, bypassing the model
with engine.connect() as conn:
    conn.execute(
        Hero.__table__.update()
        .where(Hero.__table__.c.id == 1)
        .values(name="this is way too long")
    )
    conn.commit()

# Load from database — no validation runs, invalid data loads fine
with Session(engine) as session:
    loaded = session.exec(select(Hero)).first()
    print(f"Loaded from DB: {loaded.name!r}")
```

```
Loaded from DB: 'this is way too long'
```

### Breaking Change

This could represent a behavior change for code that previously constructed `table=True` models with invalid data.

### Related Issues and PRs

- [#52](https://github.com/fastapi/sqlmodel/issues/52) — SQLModel doesn't raise ValidationError
- [#453](https://github.com/fastapi/sqlmodel/issues/453) — Why does a SQLModel class with `table=True` not validate data?
- [#134](https://github.com/fastapi/sqlmodel/issues/134) — Pydantic Validators does not raise ValueError if conditions are not met
- [#1041](https://github.com/fastapi/sqlmodel/pull/1041) — Ensure that type checks are executed when setting `table=True`
- [#227](https://github.com/fastapi/sqlmodel/pull/227) — Class Initialisation Validation Kwarg

Thank you for reviewing! 
