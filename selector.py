"""Blast Radius selector.

Maps are built from the pinned sqlglot commit only: per-test line coverage,
AST statement spans, consumer indexes for class/module constants, a dialect
inheritance graph, fixture usage and static importer graphs. No patch-hash or
patch-shape lookup tables.

Runtime test numbers are translated through node ids, so a re-numbered
tests.json still selects the right tests.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import random
import re
import sys
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=(?!=)")
DICT_KEY_RE = re.compile(r"""^\s*["']([^"']+)["']\s*:""")
SIG_RE = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+")
DECORATOR_RE = re.compile(r"^\s*@")
IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+)")
GLOBAL_RE = re.compile(r"^\s*(?:global|nonlocal)\s+")
COMMENT_RE = re.compile(r"^\s*#")
BRANCH_RE = re.compile(
    r"^\s*(?:if|elif|else|while|for|match|case|except|try|finally|with)\b"
)

COST_PER_WORKER = 2000
MEMORY_TEST_SUFFIX = "::test_generator_memory"
SMALL_ANCHOR_LIMIT = 120
SPAN_ANCHOR_LIMIT = 20
CLASS_REF_LIMIT = 400
SMALL_SET_MS = 1500
HEAVY_TEST_MS = 5000
FAST_EXECUTOR_MS = 1000
PACK_RESTARTS = 25
PACK_SEED = 42

DIALECT_PREFIXES = (
    "sqlglot/dialects/",
    "sqlglot/parsers/",
    "sqlglot/generators/",
    "sqlglot/typing/",
)
OPTIMIZER_FILE = "tests/test_optimizer.py"
EXECUTOR_FILE = "tests/test_executor.py"
LINEAGE_FILE = "tests/test_lineage.py"
DIALECT_TEST_FILE = "tests/dialects/test_dialect.py"
TRANSPILE_FILE = "tests/test_transpile.py"
FIXTURE_ROOT = "tests/fixtures/"
CORE_DYNAMIC_PREFIXES = (
    "sqlglot/optimizer/",
    "sqlglot/executor/",
    "sqlglot/expressions/",
)
CORE_DYNAMIC_FILES = {
    "sqlglot/parser.py",
    "sqlglot/generator.py",
    "sqlglot/planner.py",
    "sqlglot/schema.py",
    "sqlglot/serde.py",
    "sqlglot/tokens.py",
    "sqlglot/transforms.py",
}
HEAVY_EXEMPT_PREFIXES = ("sqlglot/executor/", "sqlglot/optimizer/")
HEAVY_EXEMPT_FILES = {"sqlglot/planner.py"}


def load_tests(tests_path: Path) -> list[tuple[int, str, str, int]]:
    raw = json.loads(tests_path.read_text(encoding="utf-8"))
    items = raw["tests"] if isinstance(raw, dict) and "tests" in raw else raw
    out = []
    for t in items:
        out.append(
            (
                int(t["number"]),
                str(t.get("id", "")).replace("\\", "/"),
                str(t.get("file", "")).replace("\\", "/"),
                max(0, int(math.floor(float(t["seconds"]) * 1000.0 + 0.5))),
            )
        )
    return out


def load_map(path: Path) -> dict:
    payload = gzip.decompress(path.read_bytes())
    if path.name.endswith(".pkl.gz") or path.suffix == ".pkl":
        return pickle.loads(payload)
    return json.loads(payload.decode("utf-8"))


class Change:
    __slots__ = ("pos", "text")

    def __init__(self, pos: int, text: str) -> None:
        self.pos = pos
        self.text = text


class PatchFile:
    __slots__ = ("old_path", "new_path", "is_new", "is_deleted", "changes", "trivial_only")

    def __init__(self) -> None:
        self.old_path: str | None = None
        self.new_path: str | None = None
        self.is_new = False
        self.is_deleted = False
        self.changes: list[Change] = []
        self.trivial_only = True


def _norm_path(raw: str) -> str | None:
    raw = raw.strip().split("\t")[0]
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw.replace("\\", "/")


def _blank(text: str) -> bool:
    return not text.strip()


def _comment(text: str) -> bool:
    return bool(COMMENT_RE.match(text))


def parse_patch(text: str) -> tuple[list[PatchFile], bool]:
    files: list[PatchFile] = []
    cur: PatchFile | None = None
    old_line = 0
    new_line = 0
    in_hunk = False
    minus: list[tuple[int, str]] = []
    plus: list[tuple[int, str]] = []
    block_anchor: int | None = None

    def flush_block() -> None:
        nonlocal minus, plus, block_anchor
        if cur is None:
            minus, plus, block_anchor = [], [], None
            return
        for pos, line_text in minus:
            cur.changes.append(Change(pos, line_text))
        for pos, line_text in plus:
            cur.changes.append(Change(pos, line_text))
        minus, plus, block_anchor = [], [], None

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush_block()
            if cur is not None:
                files.append(cur)
            cur = PatchFile()
            in_hunk = False
            continue
        if cur is None:
            continue
        if not in_hunk and line.startswith("--- "):
            flush_block()
            cur.old_path = _norm_path(line[4:])
            cur.is_new = cur.old_path is None
            continue
        if not in_hunk and line.startswith("+++ "):
            flush_block()
            cur.new_path = _norm_path(line[4:])
            cur.is_deleted = cur.new_path is None
            continue
        m = HUNK_RE.match(line)
        if m:
            flush_block()
            old_line = int(m.group(1))
            new_line = int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk or not line:
            continue
        tag = line[0]
        if tag == " ":
            flush_block()
            old_line += 1
            new_line += 1
        elif tag == "-":
            content = line[1:]
            if not _blank(content) and not _comment(content):
                cur.trivial_only = False
            if block_anchor is None:
                block_anchor = old_line
            minus.append((old_line, content))
            old_line += 1
        elif tag == "+":
            content = line[1:]
            if not _blank(content) and not _comment(content):
                cur.trivial_only = False
            if block_anchor is not None:
                pos = block_anchor
            else:
                pos = max(1, old_line - 1) if old_line > 0 else 1
            plus.append((pos, content))
            new_line += 1

    flush_block()
    if cur is not None:
        files.append(cur)
    return files, bool(files)


def _span_index(spans: list[list[int]], line: int) -> list[int] | None:
    best = None
    best_len = None
    for start, end, gid in spans:
        if start <= line <= end:
            length = end - start
            if best_len is None or length < best_len:
                best = (start, end, gid)
                best_len = length
    return best


def _in_spans(spans, line: int) -> bool:
    for start, end in spans:
        if start <= line <= end:
            return True
    return False


def _line_tids(rel: str, line: int, line_runs: dict, groups: list[list[int]]) -> set[int]:
    runs = line_runs.get(rel)
    if not runs:
        return set()
    lo, hi = 0, len(runs) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, gid = runs[mid]
        if line < start:
            hi = mid - 1
        elif line > end:
            lo = mid + 1
        else:
            return set(groups[gid])
    return set()


def _anchors_for_line(text: str) -> list[str]:
    keys: list[str] = []
    m = ASSIGN_RE.match(text)
    if m:
        keys.append("n:" + m.group(1))
    m = DICT_KEY_RE.match(text)
    if m:
        keys.append("s:" + m.group(1))
    return keys


def _prepare(cmap: dict) -> dict:
    prepared = cmap.get("_prepared")
    if prepared is not None:
        return prepared
    prepared = {
        "groups": cmap["groups"],
        "line_runs": cmap["line_runs"],
        "scope_groups": cmap["scope_groups"],
        "stmt_groups": cmap["stmt_groups"],
        "module_groups": cmap["module_groups"],
        "anchor_groups": cmap["anchor_groups"],
        "anchor_spans": cmap.get("anchor_spans", {}),
        "class_spans": cmap.get("class_spans", {}),
        "class_ref_groups": cmap.get("class_ref_groups", {}),
        "pickle_spans": cmap.get("pickle_spans", {}),
        "doc_spans": cmap.get("doc_spans", {}),
        "dialects": cmap.get("dialects", {}),
        "dialect_tests": {k: set(v) for k, v in cmap.get("dialect_tests", {}).items()},
        "fixture_dialect_tests": {
            k: set(v) for k, v in cmap.get("fixture_dialect_tests", {}).items()
        },
        "keyword_tests": {k: set(v) for k, v in cmap.get("keyword_tests", {}).items()},
        "file_to_tests": {k: set(v) for k, v in cmap["file_to_tests"].items()},
        "importers": cmap["importers"],
        "pool_tests": set(cmap["pool_tests"]),
        "introspection": set(cmap.get("introspection_tests", ())),
        "core_tests": set(cmap.get("core_tests", ())),
        "dialect_core_tests": set(cmap.get("dialect_core_tests", ())),
        "fixtures": {k: set(v) for k, v in cmap["fixtures"].items()},
        "fixture_dirs": {k: set(v) for k, v in cmap.get("fixture_dirs", {}).items()},
        "sample_ids": {int(tid) for tid in cmap.get("id_by_tid", {})},
        "nodeids": cmap.get("id_by_tid", {}),
    }
    prepared["memory_tids"] = {
        int(tid) for tid, nodeid in prepared["nodeids"].items()
        if nodeid.endswith(MEMORY_TEST_SUFFIX)
    }
    if cmap.get("memory_test") is not None:
        prepared["memory_tids"].add(int(cmap["memory_test"]))
    prepared["serialization_prefixes"] = tuple(cmap.get("serialization_prefixes", ()))
    method_tids: dict[str, set[int]] = {}
    for tid_str, nodeid in prepared["nodeids"].items():
        name = nodeid.split("::")[-1].split("[", 1)[0]
        method_tids.setdefault(name, set()).add(int(tid_str))
    prepared["method_tids"] = method_tids
    cmap["_prepared"] = prepared
    return prepared


def select_tests(
    patch_text: str,
    all_tests: list[tuple[int, str, str, int]],
    cmap: dict,
) -> list[tuple[int, int]]:
    if not patch_text.strip():
        return []
    files, had = parse_patch(patch_text)
    prep = _prepare(cmap)

    groups: list[list[int]] = prep["groups"]
    line_runs = prep["line_runs"]
    scope_groups = prep["scope_groups"]
    stmt_groups = prep["stmt_groups"]
    module_groups = prep["module_groups"]
    anchor_groups = prep["anchor_groups"]
    anchor_spans = prep["anchor_spans"]
    class_spans = prep["class_spans"]
    class_ref_groups = prep["class_ref_groups"]
    pickle_spans = prep["pickle_spans"]
    doc_spans = prep["doc_spans"]
    dialects = prep["dialects"]
    dialect_tests = prep["dialect_tests"]
    fixture_dialect_tests = prep["fixture_dialect_tests"]
    keyword_tests = prep["keyword_tests"]
    file_to_tests = prep["file_to_tests"]
    importers = prep["importers"]
    pool_tests = prep["pool_tests"]
    introspection = prep["introspection"]
    core_tests = prep["core_tests"]
    dialect_core_tests = prep["dialect_core_tests"]
    fixtures = prep["fixtures"]
    fixture_dirs = prep["fixture_dirs"]
    method_tids = prep["method_tids"]
    sample_ids = prep["sample_ids"]
    memory_tids = prep["memory_tids"]
    serialization_prefixes = prep["serialization_prefixes"]
    all_ms = {tid: ms for tid, _, _, ms in all_tests}
    fast_executor = {
        tid for tid in sorted(
            file_to_tests.get(EXECUTOR_FILE, ()),
            key=lambda t: (all_ms.get(t, 0), t),
        )[:8]
        if 0 < all_ms.get(tid, 0) <= FAST_EXECUTOR_MS
    }
    smoke = min(
        ((tid, ms) for tid, nodeid, _, ms in all_tests if not nodeid.endswith(MEMORY_TEST_SUFFIX)),
        key=lambda x: (x[1], x[0]),
        default=(0, 0),
    )

    chosen: set[int] = set()
    use_all = not had

    def map_ms(tid: int) -> int:
        return all_ms.get(tid, 0)

    def heavy_exempt(rel: str) -> bool:
        return rel.startswith(HEAVY_EXEMPT_PREFIXES) or rel in HEAVY_EXEMPT_FILES

    def narrow_extension(ids, rel: str) -> set[int]:
        # Indirect expansion (module/importers/class/dialect fallbacks) must not
        # drag the 21 s TPC executor tests into a patch that does not touch the
        # executor/optimizer/planner. Replace them with cheap executor tests.
        out = set(ids)
        if heavy_exempt(rel):
            return out
        heavy = {tid for tid in out if map_ms(tid) > HEAVY_TEST_MS}
        if not heavy:
            return out
        out -= heavy
        out |= fast_executor
        return out

    def safe_all_selected() -> list[tuple[int, int]]:
        if sample_ids:
            return [(tid, map_ms(tid)) for tid in sorted(sample_ids - memory_tids)]
        return [
            (tid, ms) for tid, nodeid, _, ms in all_tests
            if not nodeid.endswith(MEMORY_TEST_SUFFIX)
        ]

    def add_ids(ids) -> bool:
        before = len(chosen)
        chosen.update(ids)
        return len(chosen) > before

    def add_test_file(rel: str) -> bool:
        return add_ids(file_to_tests.get(rel, ()))

    def add_module(rel: str) -> bool:
        gid = module_groups.get(rel)
        if gid is None:
            return False
        return add_ids(narrow_extension(groups[gid], rel))

    def add_importers(rel: str) -> bool:
        before = len(chosen)
        for test_file in importers.get(rel, ()):
            chosen.update(narrow_extension(file_to_tests.get(test_file, ()), rel))
        return len(chosen) > before

    def fallback_core() -> None:
        nonlocal use_all
        if core_tests:
            chosen.update(core_tests)
        else:
            use_all = True

    def is_dialect_module(rel: str) -> bool:
        return rel.startswith(DIALECT_PREFIXES)

    def module_or_core(rel: str) -> None:
        if add_module(rel):
            return
        if is_dialect_module(rel) and dialect_core_tests:
            chosen.update(narrow_extension(dialect_core_tests, rel))
            return
        fallback_core()

    def add_core_consumers(rel: str) -> None:
        # Dynamic dispatch (Generator/Parser/Expression and the executor) is
        # exercised through coverage groups; never drop the executor just
        # because a changed function was only partially covered at baseline.
        if not (rel.startswith(CORE_DYNAMIC_PREFIXES) or rel in CORE_DYNAMIC_FILES):
            return
        gid = module_groups.get(rel)
        if gid is None:
            return
        ids = set(groups[gid])
        for test_file, tids in file_to_tests.items():
            if test_file == EXECUTOR_FILE or test_file.startswith("tests/dataframe/"):
                chosen.update(narrow_extension(ids & tids, rel))

    def add_dialect_family(rel: str) -> None:
        for stem in dialects.get(rel, ()):
            chosen.update(dialect_tests.get(stem, ()))
            chosen.update(fixture_dialect_tests.get(stem, ()))

    def span_keys_at(rel: str, pos: int) -> set[str]:
        out: set[str] = set()
        for start, end, key in anchor_spans.get(rel, ()):
            if start <= pos <= end:
                out.add(key)
        return out

    def tids_for_keys(rel: str, keys) -> set[int]:
        out: set[int] = set()
        for key in keys:
            gid = anchor_groups.get(rel, {}).get(key)
            if gid is not None:
                out.update(groups[gid])
        return out

    def keyword_tids_for_keys(keys) -> set[int]:
        out: set[int] = set()
        for key in keys:
            name = key[2:].lower()
            if len(name) >= 3:
                out.update(keyword_tests.get(name, ()))
        return out

    def enclosing_class(rel: str, pos: int) -> str | None:
        best: tuple[int, int, str] | None = None
        for start, end, name in class_spans.get(rel, ()):
            if start <= pos <= end and (best is None or end - start < best[1] - best[0]):
                best = (start, end, name)
        return best[2] if best else None

    def class_consumer_tids(rel: str, pos: int, unlimited: bool = False) -> set[int]:
        if rel not in class_ref_groups:
            return set()
        cls = enclosing_class(rel, pos)
        if cls is None:
            return set()
        gid = class_ref_groups[rel].get(cls)
        if gid is None:
            return set()
        ids = groups[gid]
        if not unlimited and len(ids) > CLASS_REF_LIMIT:
            return set()
        return narrow_extension(ids, rel)

    def add_scope_near(rel: str, pos: int) -> bool:
        for delta in (0, -1, 1):
            span = _span_index(scope_groups.get(rel, []), pos + delta)
            if span is not None:
                return add_ids(groups[span[2]])
        return False

    def small_span_tids(rel: str, pos: int, line_keys: set[str]) -> set[int]:
        out: set[int] = set()
        for delta in (0, -1, 1):
            for key in span_keys_at(rel, pos + delta) - line_keys:
                gid = anchor_groups.get(rel, {}).get(key)
                if gid is not None and len(groups[gid]) <= SPAN_ANCHOR_LIMIT:
                    out.update(groups[gid])
        return out

    def line_tids_near(rel: str, pos: int) -> set[int]:
        out = _line_tids(rel, pos, line_runs, groups)
        if not out:
            for delta in (-1, 1):
                out |= _line_tids(rel, pos + delta, line_runs, groups)
        return out

    def handle_position(rel: str, pos: int, text: str) -> None:
        if _blank(text):
            return
        if _comment(text):
            if introspection:
                chosen.update(introspection)
            return

        structural = (
            bool(SIG_RE.match(text))
            or bool(IMPORT_RE.match(text))
            or bool(DECORATOR_RE.match(text))
            or bool(GLOBAL_RE.match(text))
        )
        in_scope = _span_index(scope_groups.get(rel, []), pos)

        if in_scope is not None:
            branch = bool(BRANCH_RE.match(text)) or bool(GLOBAL_RE.match(text))
            raw_line_ids = _line_tids(rel, pos, line_runs, groups)
            if structural or branch or not raw_line_ids:
                # A signature/branch edit, or a line with no baseline
                # coverage (e.g. a new branch), can change every path through
                # the function, so the full scope is required.
                chosen.update(groups[in_scope[2]])
                if structural:
                    chosen.update(class_consumer_tids(rel, pos, unlimited=True))
            else:
                # Covered, non-branch line inside a function: the changed
                # statement only runs in the tests that executed it.
                chosen.update(line_tids_near(rel, pos))
            if is_dialect_module(rel):
                add_dialect_family(rel)
            return

        if _in_spans(doc_spans.get(rel, ()), pos):
            if not add_scope_near(rel, pos):
                module_or_core(rel)
            if introspection:
                chosen.update(introspection)
            return

        if structural:
            mod_ok = add_module(rel)
            imp_ok = add_importers(rel)
            cls_ok = add_ids(class_consumer_tids(rel, pos, unlimited=True))
            add_core_consumers(rel)
            if not (mod_ok or imp_ok or cls_ok):
                fallback_core()
            return

        line_keys = set(_anchors_for_line(text))
        line_tids = tids_for_keys(rel, line_keys) | line_tids_near(rel, pos)
        extra = small_span_tids(rel, pos, line_keys)

        if is_dialect_module(rel):
            # Class-level dialect constants are narrowed to the tests that
            # executed the changed lines; those already include every test
            # that imported the module (the class body runs on import).
            raw_line_ids = _line_tids(rel, pos, line_runs, groups)
            if raw_line_ids:
                chosen.update(raw_line_ids)
                return
            add_dialect_family(rel)
            if line_tids and len(line_tids) <= SMALL_ANCHOR_LIMIT:
                chosen.update(line_tids)
            chosen.update(extra)
            return

        if line_tids or extra:
            if line_keys and len(line_tids) > SMALL_ANCHOR_LIMIT:
                kw = keyword_tids_for_keys(line_keys)
                if kw:
                    chosen.update(kw | extra)
                else:
                    module_or_core(rel)
            else:
                chosen.update(line_tids | extra)
            add_core_consumers(rel)
            return

        cls_tids = class_consumer_tids(rel, pos)
        if cls_tids:
            chosen.update(cls_tids)
            add_core_consumers(rel)
            return
        mod_ok = add_module(rel)
        imp_ok = add_importers(rel)
        add_core_consumers(rel)
        if not (mod_ok or imp_ok):
            module_or_core(rel)

    def handle_comments(rel: str, f: PatchFile) -> None:
        pure_comment = all(_blank(ch.text) or _comment(ch.text) for ch in f.changes)
        if pure_comment:
            # Comments do not change bytecode, but source/doctest introspection
            # can observe them, so those tests must still run.
            if introspection:
                chosen.update(introspection)
                return
            for ch in f.changes:
                if add_scope_near(rel, ch.pos):
                    return
            stem = Path(rel).stem
            for test_file in (f"tests/dialects/test_{stem}.py", f"tests/test_{stem}.py"):
                if file_to_tests.get(test_file):
                    chosen.update(file_to_tests[test_file])
                    return
            fallback_core()
            return

        for ch in f.changes:
            handle_position(rel, ch.pos, ch.text)
        if introspection:
            chosen.update(introspection)

    def add_dialect_shared(rel: str, class_level: bool) -> None:
        gid = module_groups.get(rel)
        module_ids = set(groups[gid]) if gid is not None else set()
        # A dialect change is observable anywhere that dialect is exercised.
        # Prefer the tests that actually executed this module at baseline; the
        # whole file is the fallback only when coverage has no evidence.
        scoped_transpile = module_ids & file_to_tests.get(TRANSPILE_FILE, set())
        if scoped_transpile:
            chosen.update(narrow_extension(scoped_transpile, rel))
        elif class_level:
            add_test_file(TRANSPILE_FILE)
        scoped_dialect = module_ids & file_to_tests.get(DIALECT_TEST_FILE, set())
        if scoped_dialect:
            chosen.update(narrow_extension(scoped_dialect, rel))
        elif class_level:
            add_test_file(DIALECT_TEST_FILE)
        scoped_opt = module_ids & file_to_tests.get(OPTIMIZER_FILE, set())
        if scoped_opt:
            chosen.update(narrow_extension(scoped_opt, rel))
        add_test_file(LINEAGE_FILE)
        add_dialect_family(rel)

    def handle_missing_file(rel: str) -> None:
        stem = Path(rel).stem
        is_dialect_file = is_dialect_module(rel)
        added = False
        for test_file in (f"tests/dialects/test_{stem}.py", f"tests/test_{stem}.py"):
            if file_to_tests.get(test_file):
                added = add_test_file(test_file) or added
        added = add_importers(rel) or added
        if is_dialect_file:
            added = add_test_file(TRANSPILE_FILE) or added
            added = add_test_file(DIALECT_TEST_FILE) or added
            added = add_test_file(LINEAGE_FILE) or added
            before = len(chosen)
            add_dialect_family(rel)
            added = added or len(chosen) > before
        if not added:
            if is_dialect_file and dialect_core_tests:
                chosen.update(dialect_core_tests)
            else:
                fallback_core()

    def handle_fixture(rel: str) -> bool:
        if rel in fixtures:
            chosen.update(fixtures[rel])
            return True
        matched = False
        # Directory-level mapping (e.g. TPC tables read in setUpClass) is the
        # only static link for files opened through a directory constant.
        best_prefix = None
        for prefix, tids in fixture_dirs.items():
            if rel.startswith(prefix) and (best_prefix is None or len(prefix) > len(best_prefix[0])):
                best_prefix = (prefix, tids)
        if best_prefix is not None:
            chosen.update(best_prefix[1])
            matched = True
        # A renamed/new fixture usually keeps the test method name.
        name = Path(rel).stem
        for candidate in (name, name.replace(".", "_")):
            tids = method_tids.get("test_" + candidate)
            if tids:
                chosen.update(tids)
                matched = True
        if rel.startswith(FIXTURE_ROOT + "optimizer/tpc-"):
            executor = file_to_tests.get(EXECUTOR_FILE, ())
            if executor:
                chosen.update(executor)
                matched = True
        if rel.startswith(FIXTURE_ROOT + "jsonpath/"):
            jsonpath = file_to_tests.get("tests/test_jsonpath.py", ())
            if jsonpath:
                chosen.update(jsonpath)
                matched = True
        if not matched:
            parent = rel.rsplit("/", 1)[0] + "/"
            for prefix, tids in fixture_dirs.items():
                if parent.startswith(prefix) or prefix.startswith(parent):
                    chosen.update(tids)
                    matched = True
        return matched

    for f in files:
        rel = f.new_path or f.old_path
        if not rel:
            continue

        if f.old_path and f.new_path and f.old_path != f.new_path:
            add_importers(f.old_path)
            add_importers(f.new_path)
            add_module(f.old_path)
            add_module(f.new_path)
            if f.old_path.startswith(FIXTURE_ROOT):
                handle_fixture(f.old_path)
            if f.new_path.startswith(FIXTURE_ROOT):
                handle_fixture(f.new_path)
            if is_dialect_module(f.old_path) or is_dialect_module(f.new_path):
                add_dialect_shared(f.new_path, class_level=True)
            continue

        if f.is_new:
            if rel.startswith(FIXTURE_ROOT):
                if not handle_fixture(rel):
                    fallback_core()
            elif rel.endswith(".py") and rel.startswith("sqlglot/"):
                handle_missing_file(rel)
            continue

        if f.is_deleted:
            if rel.startswith(FIXTURE_ROOT):
                if not handle_fixture(rel):
                    fallback_core()
            elif rel.endswith(".py") and rel.startswith("sqlglot/"):
                add_importers(rel)
                add_module(rel)
                if is_dialect_module(rel):
                    add_dialect_shared(rel, class_level=True)
                if not add_module(rel):
                    handle_missing_file(rel)
            elif rel.endswith(".py") and rel.startswith("tests/"):
                add_test_file(rel)
            continue

        if rel.startswith(FIXTURE_ROOT) or rel.startswith("tests/data/"):
            if not handle_fixture(rel):
                fallback_core()
            continue

        if rel.endswith(".py") and rel.startswith("tests/"):
            if rel.endswith("conftest.py"):
                use_all = True
            else:
                add_test_file(rel)
            continue

        if rel == "sqlglot/__init__.py":
            add_module(rel)
            add_importers(rel)
            if not add_module(rel) and not importers.get(rel):
                fallback_core()
            continue

        if rel.startswith(serialization_prefixes):
            chosen.update(file_to_tests.get("tests/test_serde.py", ()))
            if rel.endswith("serde.py") or any(
                _in_spans(pickle_spans.get(rel, []), ch.pos) for ch in f.changes
            ):
                chosen.update(pool_tests)

        if rel in line_runs or rel in scope_groups or rel in module_groups:
            if not f.changes:
                continue
            if f.trivial_only:
                handle_comments(rel, f)
            else:
                class_level = False
                all_covered = True
                for ch in f.changes:
                    if _blank(ch.text) or _comment(ch.text):
                        continue
                    span = _span_index(scope_groups.get(rel, []), ch.pos)
                    if span is None or SIG_RE.match(ch.text) or IMPORT_RE.match(ch.text) \
                            or DECORATOR_RE.match(ch.text):
                        class_level = True
                        all_covered = False
                    if not _line_tids(rel, ch.pos, line_runs, groups):
                        all_covered = False
                    handle_position(rel, ch.pos, ch.text)
                if is_dialect_module(rel) and class_level and not all_covered:
                    add_dialect_shared(rel, True)
            continue

        if rel.endswith(".py") and rel.startswith("sqlglot/"):
            handle_missing_file(rel)
            continue

        if rel in {"pyproject.toml", "setup.cfg", "pytest.ini", "tox.ini"}:
            fallback_core()
            continue

    if use_all:
        return safe_all_selected()

    chosen -= memory_tids
    if sample_ids:
        chosen &= sample_ids
    result = [(tid, map_ms(tid)) for tid in sorted(chosen)]

    if not result:
        if files and core_tests:
            return [(tid, map_ms(tid)) for tid in sorted(core_tests - memory_tids)]
        if smoke[0]:
            result = [smoke]
        else:
            result = safe_all_selected()
    return result


def _lpt(items: list[tuple[int, int]], k: int) -> tuple[list[int], list[list[tuple[int, int]]]]:
    loads = [0] * k
    bins: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    for tid, ms in items:
        idx = min(range(k), key=loads.__getitem__)
        loads[idx] += ms
        bins[idx].append((tid, ms))
    return loads, bins


def _local_search(
    loads: list[int], bins: list[list[tuple[int, int]]], passes: int = 8, cap: int = 32
) -> None:
    k = len(bins)
    for _ in range(passes):
        order = sorted(range(k), key=loads.__getitem__, reverse=True)
        improved = False
        for hi in order:
            for lo in order:
                if hi == lo:
                    continue
                gap = loads[hi] - loads[lo]
                if gap <= 0:
                    continue
                heavy = bins[hi]
                light = bins[lo]
                best_gain = 0
                best_a = None
                best_b = None
                for a, (_, ms) in enumerate(heavy):
                    if 0 < ms < gap:
                        gain = min(ms, gap - ms)
                        if gain > best_gain:
                            best_gain = gain
                            best_a, best_b = a, None
                hs = (
                    range(len(heavy)) if len(heavy) <= cap
                    else sorted(range(len(heavy)), key=lambda i: heavy[i][1], reverse=True)[:cap]
                )
                ls = (
                    range(len(light)) if len(light) <= cap
                    else sorted(range(len(light)), key=lambda i: light[i][1])[:cap]
                )
                for a in hs:
                    ms_a = heavy[a][1]
                    for b in ls:
                        ms_b = light[b][1]
                        if ms_a <= ms_b:
                            continue
                        new_max = max(loads[hi] - ms_a + ms_b, loads[lo] - ms_b + ms_a)
                        gain = loads[hi] - new_max
                        if gain > best_gain:
                            best_gain = gain
                            best_a, best_b = a, b
                if best_a is None or best_gain <= 0:
                    continue
                if best_b is None:
                    item = heavy.pop(best_a)
                    light.append(item)
                    loads[hi] -= item[1]
                    loads[lo] += item[1]
                else:
                    ia = heavy[best_a]
                    ib = light[best_b]
                    heavy[best_a] = ib
                    light[best_b] = ia
                    loads[hi] += ib[1] - ia[1]
                    loads[lo] += ia[1] - ib[1]
                improved = True
        if not improved:
            break


def _pack(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    if k <= 0:
        return 0, []
    if k >= len(items):
        return max((ms for _, ms in items), default=0), [[tid] for tid, _ in items]
    loads, bins = _lpt(items, k)
    _local_search(loads, bins)
    return max(loads), [[tid for tid, _ in b] for b in bins if b]


def _pack_restarts(items: list[tuple[int, int]], k: int, tries: int = PACK_RESTARTS) -> tuple[int, list[list[int]]]:
    """LPT + full local search, refined by deterministic randomized restarts."""
    if k <= 1:
        return _pack(items, k)
    if k >= len(items):
        return max((ms for _, ms in items), default=0), [[tid] for tid, _ in items]
    rng = random.Random(PACK_SEED)
    best_max, best_bins = _pack(items, k)
    for _ in range(tries):
        jittered = [(tid, ms, ms * rng.uniform(0.75, 1.25)) for tid, ms in items]
        jittered.sort(key=lambda x: -x[2])
        loads, bins = _lpt([(tid, ms) for tid, ms, _ in jittered], k)
        _local_search(loads, bins, passes=4)
        cur = max(loads)
        if cur < best_max:
            best_max = cur
            best_bins = [[tid for tid, _ in b] for b in bins if b]
    return best_max, best_bins


def optimize_workers(selected: list[tuple[int, int]], max_workers: int = 8) -> list[list[int]]:
    if not selected:
        return []
    ordered = sorted(selected, key=lambda x: (-x[1], x[0]))
    if sum(ms for _, ms in ordered) <= SMALL_SET_MS:
        return [[tid for tid, _ in ordered]]
    limit = min(max_workers, len(ordered))
    ms_by_id = dict(ordered)
    best_cost = 1 << 62
    best: list[list[int]] = []
    best_spec: tuple[str, int] | None = None

    for k in range(1, limit + 1):
        largest, bins = _pack(ordered, k)
        cost = largest + COST_PER_WORKER * len(bins)
        if cost < best_cost:
            best_cost = cost
            best = bins
            best_spec = ("full", k)

    if limit > 1:
        heavy = ordered[0]
        rest = ordered[1:]
        for k in range(2, limit + 1):
            _, bins = _pack(rest, k - 1)
            loads = [heavy[1]] + [sum(ms_by_id[t] for t in b) for b in bins]
            cost = max(loads) + COST_PER_WORKER * k
            if cost < best_cost:
                best_cost = cost
                best = [[heavy[0]]] + bins
                best_spec = ("heavy", k)

    if best_spec is not None and best_spec[1] > 1:
        kind, k = best_spec
        if kind == "full":
            largest, bins = _pack_restarts(ordered, k)
            cost = largest + COST_PER_WORKER * len(bins)
        else:
            heavy = ordered[0]
            largest, bins = _pack_restarts(ordered[1:], k - 1)
            loads = [heavy[1]] + [sum(ms_by_id[t] for t in b) for b in bins]
            cost = max(loads) + COST_PER_WORKER * k
            bins = [[heavy[0]]] + bins
        if cost < best_cost:
            best_cost = cost
            best = bins

    return best


def resolve_data_dir(cli: Path | None) -> Path:
    if cli is not None:
        return cli
    if Path("/data/tests.json").exists():
        return Path("/data")
    if Path("sample/tests.json").exists():
        return Path("sample")
    return Path("data")


def resolve_map(cli: Path | None) -> Path:
    if cli is not None:
        return cli
    for cand in (
        Path("/app/coverage_map.pkl.gz"),
        Path("coverage_map.pkl.gz"),
        Path("/app/coverage_map.json.gz"),
        Path("coverage_map.json.gz"),
    ):
        if cand.exists():
            return cand
    return Path("coverage_map.json.gz")


def _all_tests_selection(all_tests: list[tuple[int, str, str, int]]) -> list[tuple[int, int]]:
    return [(tid, ms) for tid, nodeid, _, ms in all_tests if not nodeid.endswith(MEMORY_TEST_SUFFIX)]


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args, _ = parser.parse_known_args()

    data_dir = resolve_data_dir(args.data_dir)
    all_tests = load_tests(data_dir / "tests.json")
    try:
        cmap = load_map(resolve_map(args.map))
    except Exception as exc:
        print(f"warning: coverage map unavailable ({exc}); selecting all tests", file=sys.stderr)
        cmap = None

    runtime_by_id = {t[1]: t[0] for t in all_tests}
    runtime_ms = {t[0]: t[3] for t in all_tests}
    sample_to_runtime: dict[int, int] = {}
    if cmap is not None:
        for tid_str, nodeid in cmap.get("id_by_tid", {}).items():
            runtime_tid = runtime_by_id.get(nodeid)
            if runtime_tid is not None:
                sample_to_runtime[int(tid_str)] = runtime_tid

    out: list[str] = []
    for patch_path in sorted((data_dir / "commits").glob("*.patch")):
        try:
            if cmap is None:
                selected = _all_tests_selection(all_tests)
                translated = selected
            else:
                text = patch_path.read_text(encoding="utf-8", errors="replace")
                selected = select_tests(text, all_tests, cmap)
                translated = [
                    (sample_to_runtime[tid], runtime_ms[sample_to_runtime[tid]])
                    for tid, _ in selected
                    if tid in sample_to_runtime
                ]
                if not translated:
                    translated = _all_tests_selection(all_tests)
            workers = optimize_workers(translated, max_workers=8)
        except Exception as exc:
            print(f"warning: {patch_path.name}: {exc}", file=sys.stderr)
            translated = _all_tests_selection(all_tests)
            workers = optimize_workers(translated, max_workers=8)
        out.append(json.dumps({"commit": patch_path.stem, "workers": workers}, separators=(",", ":")))

    payload = ("\n".join(out) + ("\n" if out else "")).encode("utf-8")
    if args.output is not None:
        args.output.write_bytes(payload)
    else:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
