"""Tests for query/parser.py — tokenizer, parser, builders, helpers."""

from __future__ import annotations

import pytest

from grover.query.parser import QuerySyntaxError, parse_query, tokenize

# ===========================================================================
# Tokenizer
# ===========================================================================


class TestTokenizer:
    def test_empty_query(self):
        assert tokenize("") == ()

    def test_whitespace_only(self):
        assert tokenize("   ") == ()

    def test_pipe_token(self):
        tokens = tokenize("read /a | stat /b")
        kinds = [t.kind for t in tokens]
        assert "pipe" in kinds

    def test_amp_token(self):
        tokens = tokenize("read /a & read /b")
        kinds = [t.kind for t in tokens]
        assert "amp" in kinds

    def test_paren_tokens(self):
        tokens = tokenize("(read /a)")
        kinds = [t.kind for t in tokens]
        assert kinds[0] == "lparen"
        assert kinds[-1] == "rparen"

    def test_quoted_string(self):
        tokens = tokenize('read "hello world"')
        assert any(t.value == "hello world" and t.kind == "string" for t in tokens)

    def test_single_quoted_string(self):
        tokens = tokenize("read 'hello world'")
        assert any(t.value == "hello world" and t.kind == "string" for t in tokens)

    def test_quoted_string_with_escape(self):
        tokens = tokenize(r'read "hello\"world"')
        assert any("hello" in t.value for t in tokens if t.kind == "string")

    def test_unterminated_string(self):
        with pytest.raises(QuerySyntaxError, match="Unterminated string"):
            tokenize('read "hello')

    def test_unterminated_escape(self):
        with pytest.raises(QuerySyntaxError, match="Unterminated escape"):
            tokenize('read "hello\\')


# ===========================================================================
# Parser syntax
# ===========================================================================


class TestParserSyntax:
    def test_empty_query_raises(self):
        with pytest.raises(QuerySyntaxError, match="cannot be empty"):
            parse_query("")

    def test_unexpected_token_raises(self):
        with pytest.raises(QuerySyntaxError, match="Unexpected token"):
            parse_query("read /a )")

    def test_grouped_expression(self):
        plan = parse_query("(read /a) & (read /b)")
        assert "read" in plan.methods

    def test_intersect_subquery(self):
        plan = parse_query('glob "*.py" | intersect(glob "*.txt")')
        assert plan.ast is not None

    def test_except_subquery(self):
        plan = parse_query('glob "*.py" | except(glob "*.txt")')
        assert plan.ast is not None


# ===========================================================================
# Builder commands
# ===========================================================================


class TestBuilders:
    def test_stat(self):
        plan = parse_query("stat /a.py")
        assert plan.methods == ("stat",)
        assert plan.render_mode == "stat"

    def test_delete(self):
        plan = parse_query("rm /a.py")
        assert plan.methods == ("delete",)
        assert plan.render_mode == "action"

    def test_edit_requires_old_and_new(self):
        with pytest.raises(QuerySyntaxError, match="edit requires old and new"):
            parse_query("edit only_one_arg")

    def test_edit_with_path(self):
        plan = parse_query("edit /a.py old new")
        assert plan.methods == ("edit",)

    def test_write_wrong_arg_count(self):
        with pytest.raises(QuerySyntaxError, match="write requires a path and content"):
            parse_query("write /a.py")

    def test_write_with_overwrite(self):
        plan = parse_query('write /a.py "content" --overwrite')
        assert plan.methods == ("write",)

    def test_write_with_no_overwrite(self):
        plan = parse_query('write /a.py "content" --no-overwrite')
        assert plan.methods == ("write",)

    def test_mkdir_no_args_raises(self):
        with pytest.raises(QuerySyntaxError, match="mkdir requires"):
            parse_query("mkdir")

    def test_mkdir_multiple_paths(self):
        plan = parse_query("mkdir /a /b /c")
        assert plan.methods == ("mkdir",)

    def test_move_one_arg(self):
        plan = parse_query("mv /dest")
        assert plan.methods == ("move",)

    def test_move_two_args(self):
        plan = parse_query("mv /src /dest")
        assert plan.methods == ("move",)

    def test_move_too_many_args(self):
        with pytest.raises(QuerySyntaxError, match="mv requires"):
            parse_query("mv /a /b /c")

    def test_copy_one_arg(self):
        plan = parse_query("cp /dest")
        assert plan.methods == ("copy",)

    def test_copy_two_args(self):
        plan = parse_query("cp /src /dest")
        assert plan.methods == ("copy",)

    def test_copy_too_many_args(self):
        with pytest.raises(QuerySyntaxError, match="cp requires"):
            parse_query("cp /a /b /c")

    def test_mkconn_two_args(self):
        plan = parse_query("mkconn imports /b.py")
        assert plan.methods == ("mkconn",)

    def test_mkconn_three_args(self):
        plan = parse_query("mkconn /a.py imports /b.py")
        assert plan.methods == ("mkconn",)

    def test_mkconn_wrong_args(self):
        with pytest.raises(QuerySyntaxError, match="mkconn requires"):
            parse_query("mkconn only_one")

    def test_glob_wrong_count(self):
        with pytest.raises(QuerySyntaxError, match="glob requires"):
            parse_query("glob")

    def test_grep_wrong_count(self):
        with pytest.raises(QuerySyntaxError, match="grep requires"):
            parse_query("grep")

    def test_grep_conflicting_case_flags(self):
        with pytest.raises(QuerySyntaxError, match="cannot combine"):
            parse_query("grep pattern --ignore-case --case-sensitive")

    def test_search(self):
        plan = parse_query('search "test query"')
        assert plan.methods == ("semantic_search",)

    def test_lsearch(self):
        plan = parse_query('lsearch "test query"')
        assert plan.methods == ("lexical_search",)

    def test_vsearch_single_bracket(self):
        plan = parse_query('vsearch "[0.1, 0.2, 0.3]"')
        assert plan.methods == ("vector_search",)

    def test_vsearch_multiple_values(self):
        plan = parse_query("vsearch 0.1 0.2 0.3")
        assert plan.methods == ("vector_search",)

    def test_meetinggraph(self):
        plan = parse_query("meetinggraph /a.py /b.py")
        assert plan.methods == ("meeting_subgraph",)

    def test_meetinggraph_minimal(self):
        plan = parse_query("meetinggraph /a.py /b.py --min")
        assert plan.methods == ("min_meeting_subgraph",)

    def test_graph_traversal_with_depth(self):
        plan = parse_query("nbr /a.py --depth 3")
        assert plan.methods == ("neighborhood",)

    def test_predecessors(self):
        plan = parse_query("pred /a.py")
        assert plan.methods == ("predecessors",)

    def test_sort_by_flag(self):
        plan = parse_query('glob "*.py" | sort --by score')
        assert plan.methods == ("glob", "sort")

    def test_sort_asc(self):
        plan = parse_query('glob "*.py" | sort --asc')
        assert plan.methods == ("glob", "sort")

    def test_sort_positional_and_by_conflict(self):
        with pytest.raises(QuerySyntaxError, match="cannot combine"):
            parse_query('glob "*.py" | sort score --by name')

    def test_sort_too_many_positionals(self):
        with pytest.raises(QuerySyntaxError, match="at most one"):
            parse_query('glob "*.py" | sort a b')

    def test_top_wrong_count(self):
        with pytest.raises(QuerySyntaxError, match="top requires"):
            parse_query('glob "*.py" | top')

    def test_kinds_no_args(self):
        with pytest.raises(QuerySyntaxError, match="kinds requires"):
            parse_query('glob "*.py" | kinds')

    def test_kinds_valid(self):
        plan = parse_query('glob "*.py" | kinds file directory')
        assert plan.methods == ("glob", "kinds")

    def test_unknown_command(self):
        with pytest.raises(QuerySyntaxError, match="Unknown command"):
            parse_query("foobar /a")


# ===========================================================================
# Visibility / overwrite / kind parsing
# ===========================================================================


class TestVisibilityParsing:
    def test_all_flag(self):
        plan = parse_query("ls --all")
        assert plan.ast is not None

    def test_include_chunks(self):
        plan = parse_query("ls --include chunks")
        assert plan.ast is not None

    def test_all_and_include_conflict(self):
        with pytest.raises(QuerySyntaxError, match="Cannot combine --all and --include"):
            parse_query("ls --all --include file")

    def test_include_comma_separated(self):
        plan = parse_query("ls --include file,directory,chunk")
        assert plan.ast is not None

    def test_overwrite_both_conflict(self):
        with pytest.raises(QuerySyntaxError, match="Cannot combine"):
            parse_query('write /a.py "x" --overwrite --no-overwrite')


class TestKindNames:
    def test_all_valid_aliases(self):
        for name in ("file", "files", "dir", "dirs", "directory", "directories",
                     "chunk", "chunks", "version", "versions",
                     "connection", "connections", "api", "apis"):
            plan = parse_query(f"ls --include {name}")
            assert plan.ast is not None

    def test_invalid_kind(self):
        with pytest.raises(QuerySyntaxError, match="Unknown kind"):
            parse_query("ls --include bogus")


# ===========================================================================
# Flag splitting
# ===========================================================================


class TestFlagSplitting:
    def test_unknown_flag(self):
        with pytest.raises(QuerySyntaxError, match="Unknown flag"):
            parse_query("ls --bogus")

    def test_duplicate_flag(self):
        with pytest.raises(QuerySyntaxError, match="Duplicate flag"):
            parse_query("grep pattern --ignore-case --ignore-case")

    def test_flag_requiring_value_at_end(self):
        with pytest.raises(QuerySyntaxError, match="requires a value"):
            parse_query("grep pattern --max-results")

    def test_flag_followed_by_another_flag(self):
        with pytest.raises(QuerySyntaxError, match="requires a value"):
            parse_query("tree --depth --all")


# ===========================================================================
# _render_mode coverage
# ===========================================================================


class TestRenderMode:
    def test_content_for_read(self):
        assert parse_query("read /a").render_mode == "content"

    def test_stat_for_stat(self):
        assert parse_query("stat /a").render_mode == "stat"

    def test_ls_for_ls(self):
        assert parse_query("ls /a").render_mode == "ls"

    def test_tree_for_tree(self):
        assert parse_query("tree /a").render_mode == "tree"

    def test_action_for_write(self):
        assert parse_query('write /a "x"').render_mode == "action"

    def test_action_for_delete(self):
        assert parse_query("rm /a").render_mode == "action"

    def test_action_for_edit(self):
        assert parse_query("edit /a old new").render_mode == "action"

    def test_action_for_mkdir(self):
        assert parse_query("mkdir /a").render_mode == "action"

    def test_action_for_move(self):
        assert parse_query("mv /a /b").render_mode == "action"

    def test_action_for_copy(self):
        assert parse_query("cp /a /b").render_mode == "action"

    def test_action_for_mkconn(self):
        assert parse_query("mkconn imports /b").render_mode == "action"

    def test_query_list_for_glob(self):
        assert parse_query('glob "*.py"').render_mode == "query_list"

    def test_query_list_for_grep(self):
        assert parse_query("grep pattern").render_mode == "query_list"

    def test_query_list_for_search(self):
        assert parse_query('search "q"').render_mode == "query_list"

    def test_query_list_for_lsearch(self):
        assert parse_query('lsearch "q"').render_mode == "query_list"

    def test_query_list_for_vsearch(self):
        assert parse_query("vsearch 0.1 0.2").render_mode == "query_list"

    def test_query_list_for_graph_traversal(self):
        assert parse_query("pred /a").render_mode == "query_list"

    def test_query_list_for_meetinggraph(self):
        assert parse_query("meetinggraph /a /b").render_mode == "query_list"

    def test_query_list_for_rank(self):
        assert parse_query("pagerank").render_mode == "query_list"

    def test_query_list_for_sort_in_pipeline(self):
        plan = parse_query('glob "*.py" | sort')
        assert plan.render_mode == "query_list"

    def test_query_list_for_top_in_pipeline(self):
        plan = parse_query('glob "*.py" | top 5')
        assert plan.render_mode == "query_list"

    def test_query_list_for_kinds_in_pipeline(self):
        plan = parse_query('glob "*.py" | kinds file')
        assert plan.render_mode == "query_list"

    def test_query_list_for_intersect(self):
        plan = parse_query('glob "*.py" | intersect(glob "*.txt")')
        assert plan.render_mode == "query_list"

    def test_query_list_for_except(self):
        plan = parse_query('glob "*.py" | except(glob "*.txt")')
        assert plan.render_mode == "query_list"

    def test_pipeline_render_mode_from_last_stage(self):
        plan = parse_query('glob "*.py" | read')
        assert plan.render_mode == "content"

    def test_union_mixed_modes_becomes_query_list(self):
        plan = parse_query("read /a & ls /b")
        assert plan.render_mode == "query_list"

    def test_pipeline_no_stages_uses_source(self):
        plan = parse_query("read /a")
        assert plan.render_mode == "content"


# ===========================================================================
# Additional parser edge cases
# ===========================================================================


class TestParserEdgeCases:
    def test_unexpected_token_in_args(self):
        """Line 179: non-word/string token in command args raises."""
        with pytest.raises(QuerySyntaxError, match="Unexpected token"):
            parse_query("read (")

    def test_expect_at_end_of_tokens(self):
        """Line 202: _expect when at end of tokens."""
        with pytest.raises(QuerySyntaxError, match="Expected"):
            parse_query("intersect(read /a")  # missing closing paren

    def test_expect_wrong_kind(self):
        """Line 205: _expect with wrong token kind at position."""
        with pytest.raises(QuerySyntaxError, match="at position"):
            parse_query("intersect read /a)")  # ( expected, got word

    def test_tree_too_many_paths(self):
        """Line 234: tree with >1 path."""
        with pytest.raises(QuerySyntaxError, match="at most one"):
            parse_query("tree /a /b")

    def test_search_wrong_count(self):
        """Line 321: search requires exactly one query."""
        with pytest.raises(QuerySyntaxError, match="search requires"):
            parse_query("search")

    def test_lsearch_wrong_count(self):
        """Line 331: lsearch requires exactly one query."""
        with pytest.raises(QuerySyntaxError, match="lsearch requires"):
            parse_query("lsearch")

    def test_vsearch_no_args(self):
        """Line 341: vsearch requires values."""
        with pytest.raises(QuerySyntaxError, match="vsearch requires"):
            parse_query("vsearch")

    def test_include_empty_value(self):
        """Line 552: --include with empty comma-only value."""
        with pytest.raises(QuerySyntaxError, match="--include requires"):
            parse_query('ls --include ","')

    def test_parse_int_non_integer(self):
        """Lines 595-596: _parse_int with non-integer."""
        with pytest.raises(QuerySyntaxError, match="requires an integer"):
            parse_query("top abc")
