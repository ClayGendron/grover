"""Hand-rolled parser for the CLI query language."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, overload

from grover.query.ast import (
    CopyCommand,
    DeleteCommand,
    EditCommand,
    ExceptStage,
    GlobCommand,
    GraphTraversalCommand,
    GrepCommand,
    IntersectStage,
    KindsCommand,
    LexicalSearchCommand,
    LsCommand,
    MeetingGraphCommand,
    MkconnCommand,
    MkdirCommand,
    MoveCommand,
    PipelineNode,
    QueryNode,
    QueryPlan,
    RankCommand,
    ReadCommand,
    RenderMode,
    SemanticSearchCommand,
    SortCommand,
    StageNode,
    StatCommand,
    TopCommand,
    TreeCommand,
    UnionNode,
    VectorSearchCommand,
    Visibility,
    WriteCommand,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from grover.paths import ObjectKind


class QuerySyntaxError(ValueError):
    """Raised when the query string is syntactically invalid."""


class QueryExecutionError(ValueError):
    """Raised when a parsed query cannot be executed."""


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    position: int


@dataclass(frozen=True)
class CommandSpec:
    """Declarative parser spec for one CLI command family."""

    canonical_name: str
    aliases: tuple[str, ...]
    flags: dict[str, int]
    builder: Callable[[list[str], dict[str, str | bool]], StageNode]


def tokenize(query: str) -> tuple[Token, ...]:
    """Tokenize a CLI query string."""
    tokens: list[Token] = []
    i = 0
    while i < len(query):
        ch = query[i]
        match ch:
            case " " | "\t" | "\r" | "\n":
                i += 1
            case "(":
                tokens.append(Token("lparen", ch, i))
                i += 1
            case ")":
                tokens.append(Token("rparen", ch, i))
                i += 1
            case "|":
                tokens.append(Token("pipe", ch, i))
                i += 1
            case "&":
                tokens.append(Token("amp", ch, i))
                i += 1
            case "'" | '"':
                quote = ch
                start = i
                i += 1
                chars: list[str] = []
                while i < len(query):
                    current = query[i]
                    match current:
                        case "\\":
                            if i + 1 >= len(query):
                                raise QuerySyntaxError(f"Unterminated escape sequence at position {i}")
                            chars.append(query[i + 1])
                            i += 2
                        case _ if current == quote:
                            tokens.append(Token("string", "".join(chars), start))
                            i += 1
                            break
                        case _:
                            chars.append(current)
                            i += 1
                else:
                    raise QuerySyntaxError(f"Unterminated string starting at position {start}")
            case _:
                start = i
                while i < len(query):
                    current = query[i]
                    match current:
                        case " " | "\t" | "\r" | "\n" | "(" | ")" | "|" | "&":
                            break
                        case _:
                            i += 1
                tokens.append(Token("word", query[start:i], start))

    return tuple(tokens)


class _Parser:
    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self._tokens = tokens
        self._index = 0

    def parse(self) -> QueryNode:
        if not self._tokens:
            raise QuerySyntaxError("Query cannot be empty")
        node = self._parse_union()
        if not self._at_end():
            token = self._peek()
            raise QuerySyntaxError(f"Unexpected token {token.value!r} at position {token.position}")
        return node

    def _parse_union(self) -> QueryNode:
        operands = [self._parse_pipeline()]
        while self._match("amp"):
            operands.append(self._parse_pipeline())
        return operands[0] if len(operands) == 1 else UnionNode(tuple(operands))

    def _parse_pipeline(self) -> QueryNode:
        source = self._parse_primary()
        stages: list[StageNode] = []
        while self._match("pipe"):
            stages.append(self._parse_stage())
        return source if not stages else PipelineNode(source=source, stages=tuple(stages))

    def _parse_primary(self) -> QueryNode:
        if self._match("lparen"):
            expr = self._parse_union()
            self._expect("rparen", "Expected ')' to close grouped expression")
            return expr
        return self._parse_stage()

    def _parse_stage(self) -> StageNode:
        token = self._expect("word", "Expected a command name")
        name = token.value.lower()
        if name in {"intersect", "except"}:
            self._expect("lparen", f"Expected '(' after {name}")
            query = self._parse_union()
            self._expect("rparen", "Expected ')' to close subquery")
            return IntersectStage(query) if name == "intersect" else ExceptStage(query)

        args: list[str] = []
        while not self._at_end():
            token = self._peek()
            if token.kind in {"pipe", "amp", "rparen"}:
                break
            if token.kind not in {"word", "string"}:
                raise QuerySyntaxError(f"Unexpected token {token.value!r} at position {token.position}")
            args.append(self._advance().value)
        return _build_command(name, args)

    def _at_end(self) -> bool:
        return self._index >= len(self._tokens)

    def _peek(self) -> Token:
        return self._tokens[self._index]

    def _advance(self) -> Token:
        token = self._tokens[self._index]
        self._index += 1
        return token

    def _match(self, kind: str) -> bool:
        if self._at_end() or self._peek().kind != kind:
            return False
        self._index += 1
        return True

    def _expect(self, kind: str, message: str) -> Token:
        if self._at_end():
            raise QuerySyntaxError(message)
        token = self._peek()
        if token.kind != kind:
            raise QuerySyntaxError(f"{message} at position {token.position}")
        self._index += 1
        return token


def parse_query(query: str) -> QueryPlan:
    """Parse *query* into an AST and an ordered execution plan."""
    ast = _Parser(tokenize(query)).parse()
    return QueryPlan(
        ast=ast,
        methods=_planned_methods(ast),
        render_mode=_render_mode(ast),
    )


def _build_read(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    return ReadCommand(paths=tuple(positionals))


def _build_stat(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    return StatCommand(paths=tuple(positionals))


def _build_ls(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    return LsCommand(paths=tuple(positionals), visibility=_parse_visibility(options))


def _build_tree(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) > 1:
        raise QuerySyntaxError("tree accepts at most one explicit path")
    return TreeCommand(
        paths=tuple(positionals),
        max_depth=_parse_int_option(options, "--depth"),
        visibility=_parse_visibility(options),
    )


def _build_delete(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    return DeleteCommand(paths=tuple(positionals))


def _build_edit(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) < 2:
        raise QuerySyntaxError("edit requires old and new strings, with an optional path prefix")
    return EditCommand(
        old=positionals[-2],
        new=positionals[-1],
        paths=tuple(positionals[:-2]),
        replace_all="--all" in options,
    )


def _build_write(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    overwrite = _parse_overwrite(options)
    if len(positionals) != 2:
        raise QuerySyntaxError("write requires a path and content string")
    return WriteCommand(path=positionals[0], content=positionals[1], overwrite=overwrite)


def _build_mkdir(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    if not positionals:
        raise QuerySyntaxError("mkdir requires at least one path")
    return MkdirCommand(paths=tuple(positionals))


def _build_move(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    overwrite = _parse_overwrite(options)
    if len(positionals) == 1:
        return MoveCommand(dest=positionals[0], overwrite=overwrite)
    if len(positionals) == 2:
        return MoveCommand(src=positionals[0], dest=positionals[1], overwrite=overwrite)
    raise QuerySyntaxError("mv requires 'src dest' or, in a pipeline, just 'dest-root'")


def _build_copy(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    overwrite = _parse_overwrite(options)
    if len(positionals) == 1:
        return CopyCommand(dest=positionals[0], overwrite=overwrite)
    if len(positionals) == 2:
        return CopyCommand(src=positionals[0], dest=positionals[1], overwrite=overwrite)
    raise QuerySyntaxError("cp requires 'src dest' or, in a pipeline, just 'dest-root'")


def _build_mkconn(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    if len(positionals) == 2:
        return MkconnCommand(connection_type=positionals[0], target=positionals[1])
    if len(positionals) == 3:
        return MkconnCommand(
            source=positionals[0],
            connection_type=positionals[1],
            target=positionals[2],
        )
    raise QuerySyntaxError("mkconn requires 'source type target' or, in a pipeline, 'type target'")


def _build_glob(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) != 1:
        raise QuerySyntaxError("glob requires exactly one pattern")
    return GlobCommand(pattern=positionals[0], visibility=_parse_visibility(options))


def _build_grep(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) != 1:
        raise QuerySyntaxError("grep requires exactly one pattern")
    if "--ignore-case" in options and "--case-sensitive" in options:
        raise QuerySyntaxError("grep cannot combine --ignore-case and --case-sensitive")
    return GrepCommand(
        pattern=positionals[0],
        case_sensitive="--ignore-case" not in options,
        max_results=_parse_int_option(options, "--max-results"),
        visibility=_parse_visibility(options),
    )


def _build_search(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) != 1:
        raise QuerySyntaxError("search requires exactly one query string")
    return SemanticSearchCommand(
        query=positionals[0],
        k=_parse_int_option(options, "--k", default=15),
        visibility=_parse_visibility(options),
    )


def _build_lsearch(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if len(positionals) != 1:
        raise QuerySyntaxError("lsearch requires exactly one query string")
    return LexicalSearchCommand(
        query=positionals[0],
        k=_parse_int_option(options, "--k", default=15),
        visibility=_parse_visibility(options),
    )


def _build_vsearch(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    if not positionals:
        raise QuerySyntaxError("vsearch requires one or more numeric vector values")
    return VectorSearchCommand(
        vector=_parse_vector(positionals),
        k=_parse_int_option(options, "--k", default=15),
        visibility=_parse_visibility(options),
    )


def _build_meetinggraph(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    return MeetingGraphCommand(
        paths=tuple(positionals),
        minimal="--min" in options,
        visibility=_parse_visibility(options),
    )


TraversalMethod = Literal["predecessors", "successors", "ancestors", "descendants", "neighborhood"]
RankMethod = Literal[
    "pagerank",
    "betweenness_centrality",
    "closeness_centrality",
    "degree_centrality",
    "in_degree_centrality",
    "out_degree_centrality",
    "hits",
]


def _build_graph_traversal(method_name: TraversalMethod) -> Callable[[list[str], dict[str, str | bool]], StageNode]:
    def _builder(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
        return GraphTraversalCommand(
            method_name=method_name,
            paths=tuple(positionals),
            depth=_parse_int_option(options, "--depth", default=2) if method_name == "neighborhood" else 2,
            visibility=_parse_visibility(options),
        )

    return _builder


def _build_rank(method_name: RankMethod) -> Callable[[list[str], dict[str, str | bool]], StageNode]:
    def _builder(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
        return RankCommand(method_name, tuple(positionals), _parse_visibility(options))

    return _builder


def _build_sort(positionals: list[str], options: dict[str, str | bool]) -> StageNode:
    operation = None
    if len(positionals) > 1:
        raise QuerySyntaxError("sort accepts at most one positional operation name")
    if positionals and "--by" in options:
        raise QuerySyntaxError("sort cannot combine a positional operation with --by")
    if positionals:
        operation = positionals[0]
    elif "--by" in options:
        operation = str(options["--by"])
    return SortCommand(operation=operation, reverse="--asc" not in options)


def _build_top(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    if len(positionals) != 1:
        raise QuerySyntaxError("top requires exactly one integer")
    return TopCommand(k=_parse_int(positionals[0], "top"))


def _build_kinds(positionals: list[str], _options: dict[str, str | bool]) -> StageNode:
    if not positionals:
        raise QuerySyntaxError("kinds requires at least one kind")
    return KindsCommand(kinds=tuple(_parse_kind_name(kind) for kind in positionals))


_COMMAND_SPECS = (
    CommandSpec("read", ("read", "cat"), {}, _build_read),
    CommandSpec("stat", ("stat",), {}, _build_stat),
    CommandSpec("ls", ("ls", "list"), {"--all": 0, "--include": 1}, _build_ls),
    CommandSpec("tree", ("tree",), {"--depth": 1, "--all": 0, "--include": 1}, _build_tree),
    CommandSpec("delete", ("rm", "delete", "del"), {}, _build_delete),
    CommandSpec("edit", ("edit",), {"--all": 0}, _build_edit),
    CommandSpec("write", ("write",), {"--overwrite": 0, "--no-overwrite": 0}, _build_write),
    CommandSpec("mkdir", ("mkdir",), {}, _build_mkdir),
    CommandSpec("move", ("mv", "move"), {"--overwrite": 0, "--no-overwrite": 0}, _build_move),
    CommandSpec("copy", ("cp", "copy"), {"--overwrite": 0, "--no-overwrite": 0}, _build_copy),
    CommandSpec("mkconn", ("mkconn",), {}, _build_mkconn),
    CommandSpec("glob", ("glob",), {"--all": 0, "--include": 1}, _build_glob),
    CommandSpec(
        "grep",
        ("grep",),
        {"--all": 0, "--include": 1, "--ignore-case": 0, "--case-sensitive": 0, "--max-results": 1},
        _build_grep,
    ),
    CommandSpec("search", ("search",), {"--all": 0, "--include": 1, "--k": 1}, _build_search),
    CommandSpec("lsearch", ("lsearch",), {"--all": 0, "--include": 1, "--k": 1}, _build_lsearch),
    CommandSpec("vsearch", ("vsearch", "vectorsearch"), {"--all": 0, "--include": 1, "--k": 1}, _build_vsearch),
    CommandSpec(
        "pred",
        ("pred", "predecessors"),
        {"--all": 0, "--include": 1},
        _build_graph_traversal("predecessors"),
    ),
    CommandSpec(
        "succ",
        ("succ", "successors"),
        {"--all": 0, "--include": 1},
        _build_graph_traversal("successors"),
    ),
    CommandSpec(
        "anc",
        ("anc", "ancestors"),
        {"--all": 0, "--include": 1},
        _build_graph_traversal("ancestors"),
    ),
    CommandSpec(
        "desc",
        ("desc", "descendants"),
        {"--all": 0, "--include": 1},
        _build_graph_traversal("descendants"),
    ),
    CommandSpec(
        "nbr",
        ("nbr", "neighborhood"),
        {"--all": 0, "--include": 1, "--depth": 1},
        _build_graph_traversal("neighborhood"),
    ),
    CommandSpec(
        "meetinggraph",
        ("meetinggraph", "meeting", "meeting_subgraph"),
        {"--all": 0, "--include": 1, "--min": 0},
        _build_meetinggraph,
    ),
    CommandSpec("pagerank", ("pagerank",), {"--all": 0, "--include": 1}, _build_rank("pagerank")),
    CommandSpec(
        "betweenness",
        ("betweenness",),
        {"--all": 0, "--include": 1},
        _build_rank("betweenness_centrality"),
    ),
    CommandSpec(
        "closeness",
        ("closeness",),
        {"--all": 0, "--include": 1},
        _build_rank("closeness_centrality"),
    ),
    CommandSpec("degree", ("degree",), {"--all": 0, "--include": 1}, _build_rank("degree_centrality")),
    CommandSpec(
        "indegree",
        ("indegree",),
        {"--all": 0, "--include": 1},
        _build_rank("in_degree_centrality"),
    ),
    CommandSpec(
        "outdegree",
        ("outdegree",),
        {"--all": 0, "--include": 1},
        _build_rank("out_degree_centrality"),
    ),
    CommandSpec("hits", ("hits",), {"--all": 0, "--include": 1}, _build_rank("hits")),
    CommandSpec("sort", ("sort",), {"--by": 1, "--asc": 0}, _build_sort),
    CommandSpec("top", ("top",), {}, _build_top),
    CommandSpec("kinds", ("kinds",), {}, _build_kinds),
)

_COMMAND_REGISTRY = {alias: spec for spec in _COMMAND_SPECS for alias in spec.aliases}


def _build_command(name: str, args: list[str]) -> StageNode:
    spec = _COMMAND_REGISTRY.get(name)
    if spec is None:
        raise QuerySyntaxError(f"Unknown command: {name}")
    positionals, options = _split_flags(args, spec.flags)
    return spec.builder(positionals, options)


def _split_flags(args: list[str], spec: dict[str, int]) -> tuple[list[str], dict[str, str | bool]]:
    positionals: list[str] = []
    options: dict[str, str | bool] = {}
    index = 0
    while index < len(args):
        current = args[index]
        if current.startswith("--"):
            if current not in spec:
                raise QuerySyntaxError(f"Unknown flag: {current}")
            if current in options:
                raise QuerySyntaxError(f"Duplicate flag: {current}")
            arity = spec[current]
            if arity == 0:
                options[current] = True
                index += 1
                continue
            if index + 1 >= len(args):
                raise QuerySyntaxError(f"Flag {current} requires a value")
            value = args[index + 1]
            if value.startswith("--") and value in spec:
                raise QuerySyntaxError(f"Flag {current} requires a value")
            options[current] = value
            index += 2
            continue
        positionals.append(current)
        index += 1
    return positionals, options


def _parse_visibility(options: dict[str, str | bool]) -> Visibility:
    if "--all" in options and "--include" in options:
        raise QuerySyntaxError("Cannot combine --all and --include")
    include_all = "--all" in options
    include_kinds: tuple[ObjectKind, ...] = ()
    if "--include" in options:
        raw = str(options["--include"])
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        if not parts:
            raise QuerySyntaxError("--include requires at least one kind")
        include_kinds = tuple(_parse_kind_name(part) for part in parts)
    return Visibility(include_all=include_all, include_kinds=include_kinds)


def _parse_overwrite(options: dict[str, str | bool]) -> bool:
    if "--overwrite" in options and "--no-overwrite" in options:
        raise QuerySyntaxError("Cannot combine --overwrite and --no-overwrite")
    return "--no-overwrite" not in options


@overload
def _parse_int_option(
    options: dict[str, str | bool],
    flag: str,
    *,
    default: int,
) -> int: ...


@overload
def _parse_int_option(
    options: dict[str, str | bool],
    flag: str,
    *,
    default: None = None,
) -> int | None: ...


def _parse_int_option(
    options: dict[str, str | bool],
    flag: str,
    *,
    default: int | None = None,
) -> int | None:
    if flag not in options:
        return default
    return _parse_int(str(options[flag]), flag)


def _parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise QuerySyntaxError(f"{label} requires an integer, got {value!r}") from exc


def _parse_vector(values: list[str]) -> tuple[float, ...]:
    if len(values) == 1:
        value = values[0].strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if parts:
            values = parts
    vector: list[float] = []
    for value in values:
        try:
            vector.append(float(value))
        except ValueError as exc:
            raise QuerySyntaxError(f"vsearch requires numeric values, got {value!r}") from exc
    if not vector:
        raise QuerySyntaxError("vsearch requires at least one numeric value")
    return tuple(vector)


def _parse_kind_name(name: str) -> ObjectKind:
    canonical = name.lower()
    match canonical:
        case "file" | "files":
            return "file"
        case "directory" | "directories" | "dir" | "dirs":
            return "directory"
        case "chunk" | "chunks":
            return "chunk"
        case "version" | "versions":
            return "version"
        case "connection" | "connections":
            return "connection"
        case "api" | "apis":
            return "api"
        case _:
            raise QuerySyntaxError(f"Unknown kind: {name}")


def _planned_methods(node: QueryNode) -> tuple[str, ...]:
    match node:
        case PipelineNode(source=source, stages=stages):
            methods = list(_planned_methods(source))
            for stage in stages:
                methods.extend(_planned_methods(stage))
            return tuple(methods)
        case UnionNode(operands=operands):
            methods: list[str] = []
            for operand in operands:
                methods.extend(_planned_methods(operand))
            return tuple(methods)
        case ReadCommand():
            return ("read",)
        case StatCommand():
            return ("stat",)
        case LsCommand():
            return ("ls",)
        case TreeCommand():
            return ("tree",)
        case DeleteCommand():
            return ("delete",)
        case EditCommand():
            return ("edit",)
        case WriteCommand():
            return ("write",)
        case MkdirCommand():
            return ("mkdir",)
        case MoveCommand():
            return ("move",)
        case CopyCommand():
            return ("copy",)
        case MkconnCommand():
            return ("mkconn",)
        case GlobCommand():
            return ("glob",)
        case GrepCommand():
            return ("grep",)
        case SemanticSearchCommand():
            return ("semantic_search",)
        case LexicalSearchCommand():
            return ("lexical_search",)
        case VectorSearchCommand():
            return ("vector_search",)
        case GraphTraversalCommand(method_name=method_name):
            return (method_name,)
        case MeetingGraphCommand(minimal=True):
            return ("min_meeting_subgraph",)
        case MeetingGraphCommand():
            return ("meeting_subgraph",)
        case RankCommand(method_name=method_name):
            return (method_name,)
        case SortCommand():
            return ("sort",)
        case TopCommand():
            return ("top",)
        case KindsCommand():
            return ("kinds",)
        case IntersectStage(query=query) | ExceptStage(query=query):
            return _planned_methods(query)
        case _:  # pragma: no cover
            raise ValueError(f"Unknown node type: {node}")


def _render_mode(node: QueryNode) -> RenderMode:
    match node:
        case PipelineNode(stages=stages) if stages:
            return _render_mode(stages[-1])
        case PipelineNode(source=source):
            return _render_mode(source)
        case UnionNode(operands=operands):
            modes: set[RenderMode] = {_render_mode(operand) for operand in operands}
            return modes.pop() if len(modes) == 1 else "query_list"
        case ReadCommand():
            return "content"
        case StatCommand():
            return "stat"
        case LsCommand():
            return "ls"
        case TreeCommand():
            return "tree"
        case (
            DeleteCommand()
            | EditCommand()
            | WriteCommand()
            | MkdirCommand()
            | MoveCommand()
            | CopyCommand()
            | MkconnCommand()
        ):
            return "action"
        case (
            GlobCommand()
            | GrepCommand()
            | SemanticSearchCommand()
            | LexicalSearchCommand()
            | VectorSearchCommand()
        ):
            return "query_list"
        case (
            GraphTraversalCommand()
            | MeetingGraphCommand()
            | RankCommand()
            | SortCommand()
            | TopCommand()
            | KindsCommand()
        ):
            return "query_list"
        case IntersectStage() | ExceptStage():
            return "query_list"
        case _:  # pragma: no cover
            raise ValueError(f"Unknown node type: {node}")
