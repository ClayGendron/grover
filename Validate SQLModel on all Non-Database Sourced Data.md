# Validate SQLModel on all Non-Database Sourced Data

As noted by other issues and pull requests, when setting `table=True` in a `SQLModel` pydantic validation does not run, and this breaks the contract that "[a **SQLModel** model is also a **Pydantic** model](https://sqlmodel.tiangolo.com/features/#based-on-pydantic)". This PR build on prior ones and also hopes to address concerns with changing the intentional validation bypass for tables.

First, for those new to this this issue, here is an example of how SQLModel's behave differently when they are a `table`.

```python
from pydantic import BaseModel, ValidationError
from sqlmodel import SQLModel, Field

class HeroBase(BaseModel):
    id: int = Field(primary_key=True, default=1)
    name: str
    age: int

class HeroSQLBase(SQLModel):
    id: int = Field(primary_key=True, default=2)
    name: str
    age: int

class HeroSQLTable(SQLModel, table=True):
    id: int = Field(primary_key=True, default=3)
    name: str
    age: int

try:
    h1 = HeroBase(name="Superman", age="not an int")
    print("HeroBase: Superman created!")
except ValidationError:
    print("HeroBase: No Superman!")

try:
    h2 = HeroSQLBase(name="Superman", age="not an int")
    print("HeroSQLBase: Superman created!")
except ValidationError:
    print("HeroSQLBase: No Superman!")
    
try:
    h3 = HeroSQLTable(name="Superman", age="not an int")
    print("HeroSQLTable: Superman created!")
except ValidationError:
    print("HeroSQLTable: No Superman!")
    
>>> HeroBase: No Superman!
>>> HeroSQLBase: No Superman!
>>> HeroSQLTable: Superman created!
```

The same case is true if you have pydantic `field_validators` or `model_validators`, they will be silently ignored when `table=True`.

### Use Case and Reason for Change

I am building a library that includes a base SQLModel class, which does not have `table=True` like the example `DocumentBase` class below, that has specific validators that run to validate and populate data. Downstream developers using my library would then create their own class that is `table=True` that **inherits** from my library base class.

```python
# ---- library code ----
import hashlib
import posixpath
import uuid

from pydantic import field_validator, model_validator
from sqlmodel import SQLModel, Field


class DocumentBase(SQLModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), max_length=256, primary_key=True)
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


(
    Document(path="not_normalized/../path", content="no hash yet", project_field="important info!"),
    DocumentBase(path="not_normalized/../path", content="no hash yet", project_field="important info!")
)

# no path normalization or content_hash!
>>>(Document(path='not_normalized/../path', content='no hash yet', project_field='important info!', id='41d4bc4b-748f-4b61-9efd-949ad449bd50', content_hash=''),

# no project_field!
>>> DocumentBase(id='98d5d5e3-8509-4fe3-9558-51ba206a1f51', path='/path', content='no hash yet', content_hash='88c91e82a2ead0ae1f732ee864f5605a5e2e65f43418bf87b7b8bf38755ed024'))
```

Using SQLModel was chosen for this project its promise to provide a unified model that validates incoming data before writing to the database and has clean type safe ways to query data from the database.

This use case can be achieved as explained in the [multiple-models doc](https://sqlmodel.tiangolo.com/tutorial/fastapi/multiple-models/#multiple-models-with-duplicated-fields), but I would be asking developers to create twice as many classes when they have customizations for my base models. As seen above, I can't use my example `DocumentBase` to validate data internally when there are custom defined fields.

### Performance Conerns on ORM Reads

TODO: Talk about how validation is bypassed on ORM Reads and tests were created

### SQLAlchemy Assigning Values After Initialization

TODO: Talk about how this is maintained and tests were created