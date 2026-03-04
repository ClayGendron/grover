"""Demo — GroverAsync with DatabaseFileSystem + UserScopedFileSystem mounts."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from grover import GroverAsync, IndexingMode
from grover.fs.user_scoped_fs import UserScopedFileSystem
from grover.models.share import FileShare


async def main() -> None:
    project_engine = create_async_engine("sqlite+aiosqlite:///demo_project.db", echo=False)
    users_engine = create_async_engine("sqlite+aiosqlite:///demo_users.db", echo=False)

    g = GroverAsync(indexing_mode=IndexingMode.MANUAL)

    # ── Mount 1: plain DatabaseFileSystem at /project ──
    await g.add_mount("/project", engine=project_engine)

    # ── Mount 2: UserScopedFileSystem at /users ──
    user_fs = UserScopedFileSystem(FileShare)
    await g.add_mount("/users", user_fs, engine=users_engine)

    # ================================================================
    # /project — shared project files (no user scoping)
    # ================================================================
    print("=" * 60)
    print("/project mount — shared project files")
    print("=" * 60)

    # Create directory structure
    await g.mkdir("/project/src")
    await g.mkdir("/project/src/models")
    await g.mkdir("/project/src/routes")
    await g.mkdir("/project/tests")
    await g.mkdir("/project/docs")

    # Write source files
    await g.write("/project/src/app.py", """\
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(debug=True)
""")

    await g.write("/project/src/models/user.py", """\
from dataclasses import dataclass

@dataclass
class User:
    id: str
    name: str
    email: str

    def display_name(self) -> str:
        return self.name or self.email
""")

    await g.write("/project/src/models/project.py", """\
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Project:
    id: str
    name: str
    owner_id: str
    created_at: datetime

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-")
""")

    await g.write("/project/src/routes/api.py", """\
from flask import Blueprint, jsonify, request

api = Blueprint("api", __name__)

@api.route("/users", methods=["GET"])
def list_users():
    return jsonify([])

@api.route("/users/<user_id>", methods=["GET"])
def get_user(user_id: str):
    return jsonify({"id": user_id})

@api.route("/projects", methods=["GET"])
def list_projects():
    return jsonify([])
""")

    await g.write("/project/src/routes/auth.py", """\
from flask import Blueprint, request, redirect

auth = Blueprint("auth", __name__)

@auth.route("/login", methods=["POST"])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    # TODO: real auth
    return redirect("/")

@auth.route("/logout")
def logout():
    return redirect("/login")
""")

    await g.write("/project/tests/test_app.py", """\
import pytest

def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json == {"status": "ok"}

def test_list_users(client):
    response = client.get("/api/users")
    assert response.status_code == 200

def test_get_user(client):
    response = client.get("/api/users/abc123")
    assert response.status_code == 200
""")

    await g.write("/project/tests/test_models.py", """\
from src.models.user import User
from src.models.project import Project

def test_user_display_name():
    u = User(id="1", name="Alice", email="alice@example.com")
    assert u.display_name() == "Alice"

def test_user_display_name_fallback():
    u = User(id="2", name="", email="bob@example.com")
    assert u.display_name() == "bob@example.com"

def test_project_slug():
    from datetime import datetime
    p = Project(id="1", name="My Project", owner_id="1", created_at=datetime.now())
    assert p.slug == "my-project"
""")

    await g.write("/project/docs/readme.md", """\
# My Project

A demo Flask application with user and project models.

## Setup

    pip install flask
    python src/app.py

## API Endpoints

- GET /api/users — list all users
- GET /api/users/:id — get a user
- GET /api/projects — list all projects
""")

    await g.write("/project/pyproject.toml", """\
[project]
name = "myproject"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["flask>=3.0"]

[project.optional-dependencies]
dev = ["pytest", "coverage"]
""")

    # Read back a file
    result = await g.read("/project/src/app.py")
    print(f"\nread /project/src/app.py:\n{result.content}\n")

    # Edit a file
    result = await g.edit(
        "/project/src/app.py",
        'app.run(debug=True)',
        'app.run(host="0.0.0.0", port=8080, debug=True)',
    )
    print(f"edit: {result.message}")

    # List root
    result = await g.list_dir("/project")
    print(f"\nlist_dir /project: {[c.path for c in result.candidates]}")

    result = await g.list_dir("/project/src")
    print(f"list_dir /project/src: {[c.path for c in result.candidates]}")

    result = await g.list_dir("/project/src/models")
    print(f"list_dir /project/src/models: {[c.path for c in result.candidates]}")

    # Check existence
    result = await g.exists("/project/src/app.py")
    print(f"\nexists /project/src/app.py: {result.exists}")
    result = await g.exists("/project/nope.py")
    print(f"exists /project/nope.py: {result.exists}")

    # File info
    result = await g.get_info("/project/src/app.py")
    print(f"\ninfo /project/src/app.py: size={result.size_bytes} v={result.version}")

    # Versions (should have 2 — create + edit)
    result = await g.list_versions("/project/src/app.py")
    print(f"versions: {len(result.candidates)} version(s)")

    # Move
    result = await g.move("/project/docs/readme.md", "/project/README.md")
    print(f"\nmove: {result.message}")

    # Copy
    result = await g.copy("/project/pyproject.toml", "/project/pyproject_backup.toml")
    print(f"copy: {result.message}")

    # Delete (soft)
    result = await g.delete("/project/pyproject_backup.toml")
    print(f"delete: {result.message}")

    # Note: list_trash() iterates ALL mounts. The /users mount requires
    # user_id, so we skip the global call here and list per-user trash below.

    # ================================================================
    # /users — user-scoped files with sharing
    # ================================================================
    print("\n" + "=" * 60)
    print("/users mount — user-scoped files with sharing")
    print("=" * 60)

    alice = "alice"
    bob = "bob"

    # Alice creates her files
    await g.write("/users/notes.md", "# Alice's Notes\n\nTODO: finish the demo\n", user_id=alice)
    await g.write("/users/config.json", '{"theme": "dark", "lang": "en"}\n', user_id=alice)
    await g.mkdir("/users/drafts", user_id=alice)
    await g.write("/users/drafts/ideas.md", "## Ideas\n\n- idea 1\n- idea 2\n", user_id=alice)
    await g.write("/users/drafts/outline.md", "## Outline\n\n1. Intro\n2. Body\n3. Conclusion\n", user_id=alice)

    # Bob creates his files
    await g.write("/users/notes.md", "# Bob's Notes\n\nDifferent content, same path!\n", user_id=bob)
    await g.write("/users/todo.md", "- [x] setup\n- [ ] deploy\n", user_id=bob)

    # Each user sees only their own files
    result = await g.list_dir("/users", user_id=alice)
    print(f"\nalice's /users: {[c.path for c in result.candidates]}")

    result = await g.list_dir("/users", user_id=bob)
    print(f"bob's /users: {[c.path for c in result.candidates]}")

    # Read — same path, different content
    result = await g.read("/users/notes.md", user_id=alice)
    print(f"\nalice reads /users/notes.md: {result.content!r}")

    result = await g.read("/users/notes.md", user_id=bob)
    print(f"bob reads /users/notes.md: {result.content!r}")

    # Alice shares a draft with Bob
    result = await g.share("/users/drafts/ideas.md", bob, "read", user_id=alice)
    print(f"\nshare: {result.message}")

    # Bob reads via @shared
    result = await g.read("/users/@shared/alice/drafts/ideas.md", user_id=bob)
    print(f"bob reads shared: {result.content!r}")

    # Bob sees what's shared with him
    result = await g.list_shared_with_me(user_id=bob)
    print(f"shared with bob: {[c.path for c in result.candidates]}")

    # Alice edits her draft
    result = await g.edit(
        "/users/drafts/ideas.md",
        "- idea 2",
        "- idea 2\n- idea 3 (new!)",
        user_id=alice,
    )
    print(f"\nalice edits: {result.message}")

    # Bob sees the updated content
    result = await g.read("/users/@shared/alice/drafts/ideas.md", user_id=bob)
    print(f"bob reads updated shared: {result.content!r}")

    # Alice deletes a file
    result = await g.delete("/users/config.json", user_id=alice)
    print(f"\nalice deletes: {result.message}")

    # Alice's trash
    result = await g.list_trash(user_id=alice)
    print(f"alice's trash: {[c.path for c in result.candidates]}")

    # ================================================================
    # Cleanup
    # ================================================================
    await g.close()
    await project_engine.dispose()
    await users_engine.dispose()
    print("\ndone — demo_project.db + demo_users.db created")


if __name__ == "__main__":
    asyncio.run(main())
