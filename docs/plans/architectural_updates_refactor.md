# Typing Everywhere

All methods should typed results, please review ther repo and create a plan to ensure that is that case.

**The pragmatic middle ground** most mature Python projects land on:

- Enforce return types on all public functions/methods — non-negotiable.
- Enforce on private methods too, but allow `# type: ignore` or exceptions for genuinely trivial cases.
- Skip it for lambdas and one-line closures where inference handles it.

# Filesystem Owns Data

I moved the file connections model model over to the database file system (please add to local), and that is where I should live for all file systems. The insight is that the filesystem is the source of truth, updates to other systems (like and external vector store or keyword search or graph data structure) are driven by the event bus and not directly by methods. 

Do for example, we need to add a `add_connection` and `delete_connection` methods to filesystem protocols. This creates the database record and then uses the event bus to update the graph strucuture.

This means we should add a `vector` or `embedding` attribute to each of the file models which is a list float. We also need to create a new `Vector` object that could set a specific length to the object like `Vector[1028]`. We implemented something like this in the `quiverdb` repo, so please review that.

Also, all models should have a `path` attribute which is the file unique identifier or the chunk/version `file.py#chunk` `file.py@version`. 

We need to extend read only permissions for a mount/directory to enforce that not only is the file system objects read only but the graph and external stores (like vector stores) are read only. This will be helped out if we enfoce all updates to the core objects (files, versions, chunks, and connections) go through the file system first.

# Mount Registry

Is this used anywhere?

# File Operation and File Search Types

I want these to live in their own directory under grover and outside of fs, graph, and search. The core reasoning being that any of those objects could return a file operation or file search, so the types should not be tied to them in any way.

Please also audit all types to ensure they are needed and fall into our two base types. For example `FileInfo` and `VersionInfo` are likely no longer needed or could be derivatives of `FileOperation`.

Additionally, were are attributes the base types should have.

```python
class FileOperationResult:
	path: str
  content: str
  message: str
  success: bool
  line_start: int
  line_offset: int
  version: int
  
class FileSearchResult:
  candidates: list[FileSearchCandidate]
  message: str
  success: bool
  
class FileSearchCandidate:
	path: str
  evidence: list[Evidence]
```





