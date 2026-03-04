# Refactoring Grover Towards Filesystem Centric Implementation

In my thoughts about how to structure Grover, I have wanted to pursue maximum flexibility to allow devs to compose the framework however they would like. We have achieved a strong start to this infrastructure, but I have made the decision that all the flexiblity comes at the cost simplicity and providing strong value through a well designed framework.

This document spells out my plans for refactoring Grover towards a Filesystem centric organization instead of its current split design with `Mount` and `graph` and `search_engine` components.

## Filesystem is Source of Truth

All data is derived from the actual **files** of the filesystem. If we are working with a database filesystem, then the `grover_files` or `Files` are the source of truth. With a local filesystem, the disk is the source of truth, and in the future if we accepts other sources, those sources of the files will be the source of truth.

## All Mounts Require a Database

This is a bit of a change and I want to make sure it is well understood.

We create the following models as database tables:

- `File`
- `FileVersion`
- `FileChunk`
- `FileConnection`
- `FileShare`

These tables are created and apart of every filesystem implementation, no deviation. The only deviation is

- For local file systems or where the database doesn't contain the file source of truth, the `File` table is not the source of truth, it is a projection/view that contains metadata. The database because a metadata/versioning/search layer, not atomic file operation layer.

With this in mind, we should change our filesystem organization.

DatabaseFileSystem becomes a new base class for all filesystems. This class exposes methods that help update and maintain the five database tables defined by our models. 

Our LocalFileSystem inherits DatabaseFileSystem, but then updates the key methods to make sure it points to the local disk instead of the database to `read` `write` `edit` `move` and perform other file operations. It then calls is super class to update the metadata tables so they stay in sync. Other future filesystems would follow a similar pattern.

UserScopedFileSystem would basically be a DatabaseFileSystem with that prefixes all paths with a user directory and allows sharing.

All state is managed by the filesystem and its database, no persisted state is managed Grover or its mounting system.

## Filesystems Now Store Graph and Search Engine

Graphs and Search Engines are no longer their own objects on the mount, they are apart of the file system. Search Engine will no longer exist, and a filesystem will be initialized with the specific providers.

- graph_provider
- search_provider
  - by default, based on the database dialect, we should store the content of files, versions, and chunks as fulltext searchable. This way if devs are using the database as their search provider, it is clean to setup. We will need to update the models or other steps to ensure this happens.
- embedding_provider

Our graph and search providers need to provide methods to persist data that changes to the external source. So if we embed and save a new version of a document to the database, we need to call the search provider with the data needed to update the vector store, or if there is a new connection, that needs to be added to the graph database provider.

The filesystem would then have methods like `embed_text` that call the provider to perform the method. If provider is `None` then we still return a result type, but `success = False` and message is about there not being a provider to complete this operation.

All methods, like search and graph operation methods, need to be implemented by the DatabaseFileSystem that call the given provider. They are no longer implemented outside of it where the `Mount` routes to the proper backend.

We will also add two new providers

- version_provider
- chunk_provider

These expose the methods to create and update versions/chunks. We already have this code, we just need to re-compose it into a provider that is stateless. These properties should be defaulted, but if a dev wants to create their own process for versioning and chunking, they can compose their own provider and it works seamlessly.

With these changes, the `Mount` concept is no longer needed, or it should be kept as a very basic data class. The GroverAsync class will still create a `MountRegistry` to properly route to the correct filesystem based on path.

*We will need to think of an elegant solution to manage all of these methods because we have both `protocols` and `facade` right now, but we should likely get rid of most of our file system, graph, and search `protocols` as they are implemented by DatabaseFileSystem and inherited by other filesystems.*





