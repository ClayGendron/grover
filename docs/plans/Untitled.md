```python
from grover import Grover, LocalFileSystem, DatabaseFileSystem

fs = Grover()
fs.mount(LocalFileSystem()) # mounts current repo, doesn't allow operations outside of this file system
fs.mount(DatabaseFileSystem()) # dev would need to configure, but mounts a database file system

fs.read(path='') # could read from either
```

Could also have multiple local and database file systems.

1. Local always has versioning with a local sqlite db (stored in user root .grover/version.db or something) local only needs a database session for this



```python
fs = Grover()
fs = GroverAsync()

db1 = DatabaseFileSystem(
	session_factor="mssql connection",
  schema="app",
  dialect="mssql"
)

db2 = DatabaseFileSystem(
	session_factor="mssql connection",
  schema="test",
  dialect="mssql"
)

db3 = DatabaseFileSystem(
	session_factor="postgres connection",
  schema="app",
  dialect="postgres"
)

fs.mount(
	path='/app',
  backend=db1,
)

fs.mount(
	path='/test',
  backend=db2,
)

fs.mount(
	path='/app2',
  backend=db3,
)

fs.write(path='/app2/test.py')


db1 = DatabaseFileSystem(
	session_factor="mssql connection",
  schema="app",
  dialect="mssql",
  files_table="files_tablename",
  file_versions_table="file_versions_tablename"
)

```

