I want to compose a mount in this fashion. It has a filesystem, graph, and search attributes that are public attributes.

```python

g = Grover()

mount = Mount(
	path='/test'
  filesystem=DatabaseFilesystem(session_factory=sqlachemy_session_factory),
  graph=RuskworkxGraph(),
  search=SearchEngine(
  	lexical=PineconeLexicalSearch(config),
    vector=PineconeVectorSearch(config),
    hybrid=PineconeHybridSearch(config),
    embedding=OpenAIEmbedding(config)
  )
)

g.add_mount(mount)

# could also do

g.add_mount(
	path='/test_two',
  filesystem=DatabaseFilesystem(session_factory=sqlachemy_session_factory),
  graph=RuskworkxGraph(),
  search=SearchEngine(
  	lexical=PineconeLexicalSearch(config),
    vector=PineconeVectorSearch(config),
    hybrid=PineconeHybridSearch(config),
    embedding=OpenAIEmbedding(config)
  )
)
```

Now, there are various methods that a `Mount` and there for `Grover` may implement.

**File Operations**

- `read`
- `write`
- `edit`
- `delete`
- `move`
- `copy`
- etc.

**Graph Algorithms**

- `successors`
- `predessors`
- `page_rank`
- `min_spanning_tree`
- `community_detection`
- etc.

**Search Functions**

- `grep`
- `glob`
- `vector_search`
- `lexical_search`
- `hybrid_search`
- `embed_text`
- etc.

To manage when two attributes of the three main `Mount` attributes (either `filesystem`, `graph`, or `search`) have overlapping abilities (lets say `filesystem` and `search` both implement `grep`) we use the `Protocol` detection to see what protocol each of the attributes implements and raise an error if there are duplicate protocols implemented (like `filesystem and search implement GrepGlobProtocol`). We do not automatically choose.

A `Mount` does not have to implement all `Protocols` and will raise an error if a method is not implement `HybridSearchProtocol not implemented. Check mount SearchEngine configuration`. 

Each method has it own return type `GrepResult` `HybridSearchResult` `ReadResult` `RestoreResult`. For all of these types, they must fall into a base type category of either:

- `FileOperationResult`
- `FileSearchResult`

`FileOperationResult` means that something happened and it was related to the *content* of the file. You may have read the file, edited it, deleted it, moved it, or made a directory, or many other options. A `FileOperationResult` requires the caller *know* what file or directory reference's content it wants to interact with.

`FileSearchResult` mean that only file *references* (or paths) are returned. No content is returned with a `FileSearchResult`. The caller does not *know* the exact files it needs to reference and is working to search the filesystem to then perform a file operation. 

In this example, `read` would return a `FileOperationResult`, `list_dir` and `tree` would return a `FileSearchResult`.

`FileSearchResult` objects may be chained together like `grep -> vector_search -> min_spanning_tree`. `FileOperationResult` object do not chain togeather, nor chain with `FileSearchResults`.

