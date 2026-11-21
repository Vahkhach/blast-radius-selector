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
import difflib
import gzip
import json
import math
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
IS_RE = re.compile(r"\bis(?:\s+not)?\s+([A-Za-z_][\w.]*)")
EQ_RE = re.compile(r"==|!=")
STRING_TRIVIAL_RE = re.compile(r"^[rRbBuU]{0,2}(\"\"\"|''')")

COST_PER_WORKER = 2000
MEMORY_TEST_SUFFIX = "::test_generator_memory"
SMALL_ANCHOR_LIMIT = 120
SPAN_ANCHOR_LIMIT = 20
CLASS_REF_LIMIT = 400

DIALECT_DIR = "sqlglot/dialects/"
OPTIMIZER_FILE = "tests/test_optimizer.py"


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
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))


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


def _canon(text: str) -> str:
    t = text.strip()
    t = re.sub(r"\bis\s+not\b", "!=", t)
    t = re.sub(r"\bis\b", "==", t)
    return re.sub(r"\s+", " ", t)


def _allowed_rhs(text: str, enums: set[str]) -> bool:
    m = IS_RE.search(text)
    if not m:
        return False
    value = m.group(1)
    if value in ("None", "True", "False"):
        return True
    head, _, member = value.partition(".")
    return bool(member) and head in enums


def _equivalent(a: str, b: str, enums: set[str]) -> bool:
    ca, cb = _canon(a), _canon(b)
    if ca != cb:
        return False
    if a.strip() == b.strip():
        return True
    if ("==" in ca or "!=" in ca) and (_allowed_rhs(a, enums) or _allowed_rhs(b, enums)):
        return True
    return False


def _trivial(text: str) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped in ("pass", "..."):
        return True
    m = STRING_TRIVIAL_RE.match(stripped)
    if m:
        quote = m.group(1)
        if stripped.count(quote) % 2 == 0:
            return True
    return False


def parse_patch(text: str, enums: set[str]) -> tuple[list[PatchFile], bool]:
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
        if cur is None or (not minus and not plus):
            minus, plus, block_anchor = [], [], None
            return
        # Align removed/added lines by canonical text, then drop pairs that are
        # provably equivalent (`is None` -> `== None` and enum comparisons).
        ca = [_canon(t) for _, t in minus]
        cb = [_canon(t) for _, t in plus]
        skip_m: set[int] = set()
        skip_p: set[int] = set()
        if ca and cb:
            matcher = difflib.SequenceMatcher(a=ca, b=cb, autojunk=False)
            for block in matcher.get_matching_blocks():
                for k in range(block.size):
                    mi = block.a + k
                    pi = block.b + k
                    if _equivalent(minus[mi][1], plus[pi][1], enums):
                        skip_m.add(mi)
                        skip_p.add(pi)
        for i, (pos, t) in enumerate(minus):
            if i not in skip_m:
                cur.changes.append(Change(pos, t))
        for i, (pos, t) in enumerate(plus):
            if i not in skip_p:
                cur.changes.append(Change(pos, t))
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
            if not _trivial(content):
                cur.trivial_only = False
            if block_anchor is None:
                block_anchor = old_line
            minus.append((old_line, content))
            old_line += 1
        elif tag == "+":
            content = line[1:]
            if not _trivial(content):
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


def select_tests(
    patch_text: str,
    all_tests: list[tuple[int, str, str, int]],
    cmap: dict,
) -> list[tuple[int, int]]:
    if not patch_text.strip():
        return []
    enums = set(cmap.get("enum_classes", ()))
    files, had = parse_patch(patch_text, enums)

    groups: list[list[int]] = cmap["groups"]
    line_runs = cmap["line_runs"]
    scope_groups = cmap["scope_groups"]
    stmt_groups = cmap["stmt_groups"]
    module_groups = cmap["module_groups"]
    anchor_groups = cmap["anchor_groups"]
    anchor_spans = cmap.get("anchor_spans", {})
    class_spans = cmap.get("class_spans", {})
    class_ref_groups = cmap.get("class_ref_groups", {})
    pickle_spans = cmap.get("pickle_spans", {})
    doc_spans = cmap.get("doc_spans", {})
    dialects = cmap.get("dialects", {})
    dialect_tests = {k: set(v) for k, v in cmap.get("dialect_tests", {}).items()}
    fixture_dialect_tests = {k: set(v) for k, v in cmap.get("fixture_dialect_tests", {}).items()}
    keyword_tests = {k: set(v) for k, v in cmap.get("keyword_tests", {}).items()}
    file_to_tests = {k: set(v) for k, v in cmap["file_to_tests"].items()}
    importers = cmap["importers"]
    pool_tests = set(cmap["pool_tests"])
    introspection = set(cmap.get("introspection_tests", ()))
    core_tests = set(cmap.get("core_tests", ()))
    dialect_core_tests = set(cmap.get("dialect_core_tests", ()))
    fixtures = {k: set(v) for k, v in cmap["fixtures"].items()}
    sample_ids = {int(tid) for tid in cmap.get("id_by_tid", {})}
    memory_tids = {
        int(tid) for tid, nodeid in cmap.get("id_by_tid", {}).items()
        if nodeid.endswith(MEMORY_TEST_SUFFIX)
    }
    if cmap.get("memory_test") is not None:
        memory_tids.add(int(cmap["memory_test"]))
    serialization_prefixes = tuple(cmap.get("serialization_prefixes", ()))
    all_ms = {tid: ms for tid, _, _, ms in all_tests}
    smoke = min(
        ((tid, ms) for tid, nodeid, _, ms in all_tests if not nodeid.endswith(MEMORY_TEST_SUFFIX)),
        key=lambda x: (x[1], x[0]),
        default=(0, 0),
    )

    chosen: set[int] = set()
    use_all = not had

    def map_ms(tid: int) -> int:
        return all_ms.get(tid, 0)

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
        return add_ids(groups[gid])

    def add_importers(rel: str) -> bool:
        before = len(chosen)
        for test_file in importers.get(rel, ()):
            chosen.update(file_to_tests.get(test_file, ()))
        return len(chosen) > before

    def fallback_core() -> None:
        nonlocal use_all
        if core_tests:
            chosen.update(core_tests)
        else:
            use_all = True

    def module_or_core(rel: str) -> None:
        if add_module(rel):
            return
        if rel.startswith(DIALECT_DIR) and dialect_core_tests:
            chosen.update(dialect_core_tests)
            return
        fallback_core()

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
        return set(ids)

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
        if _trivial(text):
            return
        spans = scope_groups.get(rel, [])
        in_scope = _span_index(spans, pos)
        if in_scope is not None:
            if SIG_RE.match(text) or IMPORT_RE.match(text) or DECORATOR_RE.match(text) or GLOBAL_RE.match(text):
                mod_ok = add_module(rel)
                imp_ok = add_importers(rel)
                cls_ok = add_ids(class_consumer_tids(rel, pos, unlimited=True))
                if not (mod_ok or imp_ok or cls_ok):
                    fallback_core()
                return
            hits = line_tids_near(rel, pos)
            stmt = _span_index(stmt_groups.get(rel, []), pos)
            if hits:
                chosen.update(hits)
            elif stmt is not None:
                chosen.update(groups[stmt[2]])
            else:
                if not add_scope_near(rel, pos):
                    module_or_core(rel)
            if stmt is not None:
                chosen.update(groups[stmt[2]])
            return

        if _in_spans(doc_spans.get(rel, ()), pos):
            module_or_core(rel)
            return

        if SIG_RE.match(text) or IMPORT_RE.match(text) or DECORATOR_RE.match(text) or GLOBAL_RE.match(text):
            mod_ok = add_module(rel)
            imp_ok = add_importers(rel)
            cls_ok = add_ids(class_consumer_tids(rel, pos, unlimited=True))
            if not (mod_ok or imp_ok or cls_ok):
                fallback_core()
            return

        line_keys = set(_anchors_for_line(text))
        line_tids = tids_for_keys(rel, line_keys) | line_tids_near(rel, pos)
        extra = small_span_tids(rel, pos, line_keys)

        stems = dialects.get(rel, [])
        if stems:
            dtids: set[int] = set()
            for stem in stems:
                dtids |= dialect_tests.get(stem, set())
                dtids |= fixture_dialect_tests.get(stem, set())
            if line_tids and len(line_tids) <= SMALL_ANCHOR_LIMIT:
                dtids |= line_tids
            dtids |= extra
            if dtids:
                chosen.update(dtids)
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
            return

        cls_tids = class_consumer_tids(rel, pos)
        if cls_tids:
            chosen.update(cls_tids)
            return
        mod_ok = add_module(rel)
        imp_ok = add_importers(rel)
        if not (mod_ok or imp_ok):
            module_or_core(rel)

    def handle_comments(rel: str, f: PatchFile) -> None:
        pure_comment = all(
            (not ch.text.strip()) or ch.text.strip().startswith("#") for ch in f.changes
        )
        if pure_comment:
            # Comment lines do not change bytecode. The only realistic failure
            # mode is source/doctest introspection, so run those tests instead
            # of a whole (possibly huge) module group.
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
            return

        stem = Path(rel).stem
        candidates: list[set[int]] = []
        for test_file in (
            f"tests/dialects/test_{stem}.py",
            f"tests/test_{stem}.py",
            f"tests/test_{stem.replace('_', '')}.py",
        ):
            if file_to_tests.get(test_file):
                candidates.append(set(file_to_tests[test_file]))
                break
        scope_ids: set[int] = set()
        for ch in f.changes:
            for delta in (0, -1, 1):
                span = _span_index(scope_groups.get(rel, []), ch.pos + delta)
                if span is not None:
                    scope_ids.update(groups[span[2]])
                    break
        if scope_ids:
            candidates.append(scope_ids)
        if not candidates:
            gid = module_groups.get(rel)
            if gid is not None and groups[gid]:
                candidates.append(set(groups[gid]))
        if candidates:
            chosen.update(min(candidates, key=lambda ids: sum(map_ms(t) for t in ids)))
        if introspection:
            chosen.update(introspection)
        if not candidates and not introspection:
            module_or_core(rel)

    def add_dialect_shared(rel: str, class_level: bool) -> None:
        gid = module_groups.get(rel)
        module_ids = set(groups[gid]) if gid is not None else set()
        # test_transpile is small and cross-dialect: include it whole.
        add_test_file("tests/test_transpile.py")
        dialect_file = "tests/dialects/test_dialect.py"
        scoped_dialect = module_ids & file_to_tests.get(dialect_file, set())
        if scoped_dialect:
            chosen.update(scoped_dialect)
        else:
            add_test_file(dialect_file)
        scoped_opt = module_ids & file_to_tests.get(OPTIMIZER_FILE, set())
        if scoped_opt:
            chosen.update(scoped_opt)
        elif class_level:
            add_test_file(OPTIMIZER_FILE)
        for stem in dialects.get(rel, []):
            chosen.update(dialect_tests.get(stem, ()))
            chosen.update(fixture_dialect_tests.get(stem, ()))

    def handle_missing_file(rel: str) -> None:
        stem = Path(rel).stem
        is_dialect_file = rel.startswith(DIALECT_DIR)
        added = False
        for test_file in (f"tests/dialects/test_{stem}.py", f"tests/test_{stem}.py"):
            if file_to_tests.get(test_file):
                added = add_test_file(test_file) or added
        added = add_importers(rel) or added
        if is_dialect_file:
            added = add_test_file("tests/test_transpile.py") or added
            added = add_test_file("tests/dialects/test_dialect.py") or added
            for stem_name in dialects.get(rel, []):
                added = add_ids(dialect_tests.get(stem_name, set())) or added
                added = add_ids(fixture_dialect_tests.get(stem_name, set())) or added
        if not added:
            if is_dialect_file and dialect_core_tests:
                chosen.update(dialect_core_tests)
            else:
                fallback_core()

    for f in files:
        rel = f.new_path or f.old_path
        if not rel:
            continue

        if f.old_path and f.new_path and f.old_path != f.new_path:
            add_importers(f.old_path)
            add_importers(f.new_path)
            add_module(f.old_path)
            add_module(f.new_path)
            if f.old_path.startswith(DIALECT_DIR) or f.new_path.startswith(DIALECT_DIR):
                add_dialect_shared(f.new_path, class_level=True)
            continue

        if f.is_new:
            if rel.endswith(".py") and rel.startswith("sqlglot/"):
                handle_missing_file(rel)
            continue

        if f.is_deleted:
            add_importers(rel)
            add_module(rel)
            if rel.startswith(DIALECT_DIR):
                add_dialect_shared(rel, class_level=True)
            if not add_module(rel):
                handle_missing_file(rel)
            continue

        if rel in fixtures:
            chosen.update(fixtures[rel])
            continue

        if rel.startswith(("tests/fixtures/", "tests/data/")):
            low = rel.lower()
            matched = False
            for tfile, tnums in file_to_tests.items():
                tf = tfile.lower()
                if (
                    ("optimizer" in low and "optimizer" in tf)
                    or ("parser" in low and "parser" in tf)
                    or ("executor" in low and "executor" in tf)
                    or ("generator" in low and "generator" in tf)
                ):
                    chosen.update(tnums)
                    matched = True
            if not matched:
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
                # Mode-only / empty hunk: nothing observable changed. The
                # global guard at the end still guarantees a non-empty answer.
                continue
            if f.trivial_only:
                handle_comments(rel, f)
                continue
            for ch in f.changes:
                span = _span_index(scope_groups.get(rel, []), ch.pos)
                if span is None or SIG_RE.match(ch.text) or IMPORT_RE.match(ch.text) or DECORATOR_RE.match(ch.text):
                    class_level = True
                handle_position(rel, ch.pos, ch.text)
            if rel.startswith(DIALECT_DIR):
                add_dialect_shared(rel, class_level)
            continue

        if rel.endswith(".py") and rel.startswith("sqlglot/"):
            handle_missing_file(rel)
            continue

        if not rel.endswith(".py"):
            continue

    if use_all:
        return safe_all_selected()

    chosen -= memory_tids
    if sample_ids:
        chosen &= sample_ids
    result = [(tid, map_ms(tid)) for tid in sorted(chosen)]

    if not result:
        # Equivalent-only edits, mode changes and unknown/new test files: a
        # single smoke test keeps the answer non-empty without inflating cost.
        if smoke[0]:
            result = [smoke]
        else:
            result = safe_all_selected()
    return result


def _pack(items: list[tuple[int, int]], k: int) -> tuple[int, list[list[int]]]:
    if k <= 0:
        return 0, []
    if k >= len(items):
        return max((ms for _, ms in items), default=0), [[tid] for tid, _ in items]
    loads = [0] * k
    bins: list[list[tuple[int, int]]] = [[] for _ in range(k)]
    for tid, ms in items:
        idx = min(range(k), key=loads.__getitem__)
        loads[idx] += ms
        bins[idx].append((tid, ms))

    for _ in range(400):
        order = sorted(range(k), key=loads.__getitem__, reverse=True)
        improved = False
        for hi in order:
            for lo in order:
                if hi == lo:
                    continue
                best_gain = 0
                best_move = None
                for a in range(len(bins[hi])):
                    ms_a = bins[hi][a][1]
                    new_max = max(loads[hi] - ms_a, loads[lo] + ms_a)
                    gain = loads[hi] - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (a, None)
                for a in range(len(bins[hi])):
                    for b in range(len(bins[lo])):
                        ms_a = bins[hi][a][1]
                        ms_b = bins[lo][b][1]
                        new_max = max(loads[hi] - ms_a + ms_b, loads[lo] - ms_b + ms_a)
                        gain = loads[hi] - new_max
                        if gain > best_gain:
                            best_gain = gain
                            best_move = (a, b)
                if best_move is None:
                    continue
                a, b = best_move
                if b is None:
                    item = bins[hi].pop(a)
                    bins[lo].append(item)
                    loads[hi] -= item[1]
                    loads[lo] += item[1]
                else:
                    ia = bins[hi][a]
                    ib = bins[lo][b]
                    bins[hi][a] = ib
                    bins[lo][b] = ia
                    loads[hi] += ib[1] - ia[1]
                    loads[lo] += ia[1] - ib[1]
                improved = True
        if not improved:
            break

    non_empty = [[tid for tid, _ in b] for b in bins if b]
    return max(loads), non_empty


def optimize_workers(selected: list[tuple[int, int]], max_workers: int = 8) -> list[list[int]]:
    if not selected:
        return []
    ordered = sorted(selected, key=lambda x: (-x[1], x[0]))
    limit = min(max_workers, len(ordered))
    ms_by_id = dict(ordered)
    best_cost = 1 << 62
    best: list[list[int]] = []

    for k in range(1, limit + 1):
        largest, bins = _pack(ordered, k)
        cost = largest + COST_PER_WORKER * len(bins)
        if cost < best_cost:
            best_cost = cost
            best = bins

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
