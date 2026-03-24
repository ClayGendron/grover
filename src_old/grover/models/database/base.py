"""ValidatedSQLModel — SQLModel base that runs Pydantic validation on explicit construction.

SQLModel ``table=True`` models skip validation in ``__init__`` (they use
``sqlmodel_table_construct`` instead of ``validate_python``).  This override
adds validation back while preserving no-validation for:

- **ORM loads** — SQLAlchemy doesn't call ``__init__`` when hydrating from DB.
- **model_validate()** — sets ``finish_init=False``; handles its own validation.
"""

from __future__ import annotations

from sqlmodel import SQLModel
from sqlmodel._compat import finish_init


class ValidatedSQLModel(SQLModel):
    """SQLModel base that runs Pydantic validation on explicit construction."""

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        # Non-table models already validate via Pydantic's normal path.
        if not self.__class__.model_config.get("table", False):
            return
        # model_validate() sets finish_init=False and runs its own
        # validate_python — skip here to avoid double-validation.
        if not finish_init.get():
            return
        # validate_python(..., self_instance=self) mutates __dict__
        # in-place, which can wipe _sa_instance_state that SQLAlchemy's
        # instrumentation set during super().__init__. Save and restore
        # it so ORM operations (session.add, session.merge) still work.
        sa_state = self.__dict__.get("_sa_instance_state")
        field_values = {}
        for field_name in self.__class__.model_fields:
            if hasattr(self, field_name):
                field_values[field_name] = getattr(self, field_name)
        self.__pydantic_validator__.validate_python(field_values, self_instance=self)
        if sa_state is not None:
            self.__dict__["_sa_instance_state"] = sa_state
