# `grover`: The Virtual File System for Enterprise Knowledge

```bash
pip install grover
```

A strong AI agent is built from two main components, an **LLM** and **context**. Because terminology in the AI space is not well definied and changes rapidly, here is how we define those two terms:

- **LLM:** A generative large language model that has a strong understanding of language, code, and how to use tools.
- **Context:** The prompts, tools, environment, data, and search components that are compose the harness and runtime for the LLM.

AI labs are doing the work needed to make sure the LLM's are capability of performing agentic work — and also some of the context work with products like Claude Code and Codex — but that leaves individuals and organizations with a lot of remaining work to do to in building agentic systems that perform well and solve real problems. `grover` is being built to make this context work easier for developers, as well as LLM's.

`grover` is a **file system** at heart, but it is more generally a **semantic layer** for agents to naviagate and compose content from the scale of a single repository to an enterprise-wide knowledgebase.

What differentiates `grover` from other frameworks is that it has a distributed, database-first design allowing developers to store the data they want accessible for their agent within their existing infrastructure, and it has the capability integrated with vector search and graph traversal providers to accomplish more advanced and percise search. These three components create the `GroverFileSystem`.

## What is the `GroverFileSystem`?

The main class of this library is called `GroverFileSystem`, and it is the core abstraction that enables developers to build a clean API around a database-backed file system, with the ability to use other storage sources. This file system exposes many familiar concepts like `read`, `write`, `edit`,`list_dir`, along with `grep` and `glob`, and these are tools that the current SOTA LLM's are trained to know how to use effectively to navigate code bases, or more generally computer environments. However, the traditional file system concept is just one of the core concepts in `grover`.

### Core Components

1. **File System:** At the core of `grover` sits a versioned, chunkable, permission-aware, database-backed file system for text files and directories. All operations done with `grover` are reversable and protected against data-loss.
2. **Vector Search:** It is easy to plug in and connect to vector search providers to enable semantic search across the fille system, as well as automatic embedding and indexing of new data when desired.
3. **Graph Traversal:** The connections between files are treated as first class objects in `grover`, and the file system is built to utilize graph data structures to find connections and perform graph algorithms.

These three components share references between each other using a `GroverPath` which acts as each objects unqiue identifier.