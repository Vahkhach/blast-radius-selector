"""Blast Radius selector (v4).

Maps are built from the pinned sqlglot commit only: per-test line coverage,
AST statement spans, a consumer index for class/module constants, dialect
inheritance, fixture usage and static importers. No patch-hash lookup tables.

Runtime test numbers are translated through node ids, so a re-numbered
tests.json still selects the right tests.
"""
from __future__ import annotations

import argparse
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
IMPORT_RE = re.compile(r"^\s*(?:from\s+\S+\s+import|import\s+)")
GLOBAL_RE = re.compile(r"^\s*(?:global|nonlocal)\s+")
IS_RE = re.compile(r"\bis(?:\s+not)?\s+([A-Za-z_][\w.]*)")
EQ_RE = re.compile(r"==|!=")

COST_PER_WORKER = 2000
MEMORY_TEST_SUFFIX = "::test_generator_memory"
SMALL_ANCHOR_LIMIT = 120
SPAN_ANCHOR_LIMIT = 20
CLASS_REF_LIMIT = 400


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


def _equivalent(a: str, b: str, enums: set[str]) -> bool:
    ca, cb = _canon(a), _canon(b)
    if ca != cb:
        return False
    if ca == a.strip() and cb == b.strip():
        return True
    if ca == cb and ("==" in ca or "!=" in ca):
        rhs = IS_RE.search(a)
        if rhs:
            value = rhs.group(1)
            if value == "None":
                return True
            head = value.split(".")[0]
            if "." in value and head in enums:
                return True
        return False
    return True


def _trivial(text: str, enums: set[str]) -> bool:
    stripped = text.strip()
    if not stripped or stripped.startswith("#"):
        return True
    if stripped in ("pass", "..."):
        return True
    if stripped.startswith(('"""', "'''", 'r"""', "r'''")) and stripped.count('"""') % 2 == 0:
        return True
    return False


def parse_patch(text: str, enums: set[str]) -> tuple[list[PatchFile], bool]:
    files: list[PatchFile] = []
    cur: PatchFile | None = None
    old_line = 0
    new_line = 0
    in_hunk = False
    minus: list[tuple[int, str]] = []
    plus: list[tuple[int, int, str]] = []

    def flush_hunk() -> None:
        nonlocal minus, plus, cur
        if cur is None:
            return
        n = max(len(minus), len(plus))
        for i in range(n):
            m = minus[i] if i < len(minus) else None
            p = plus[i] if i < len(plus) else None
            if m and p and _equivalent(m[1], p[2], enums):
                continue
            if m:
                cur.changes.append(Change(m[0], m[1]))
            if p:
                cur.changes.append(Change(p[1] if p[1] > 0 else p[0], p[2]))
            if m or p:
                cur.trivial_only = False
        for _, t in minus:
            if not _trivial(t, enums):
                cur.trivial_only = False
        for _, _, t in plus:
            if not _trivial(t, enums):
                cur.trivial_only = False
        minus, plus = [], []

    for line in text.splitlines():
        if line.startswith("diff --git "):
            flush_hunk()
            if cur is not None:
                files.append(cur)
            cur = PatchFile()
            in_hunk = False
            continue
        if cur is None:
            continue
        if line.startswith("--- "):
            flush_hunk()
            cur.old_path = _norm_path(line[4:])
            cur.is_new = cur.old_path is None
            in_hunk = False
            continue
        if line.startswith("+++ "):
            flush_hunk()
            cur.new_path = _norm_path(line[4:])
            cur.is_deleted = cur.new_path is None
            in_hunk = False
            continue
        m = HUNK_RE.match(line)
        if m:
            flush_hunk()
            old_line = int(m.group(1))
            new_line = int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk or not line:
            continue
        tag = line[0]
        if tag == " ":
            old_line += 1
            new_line += 1
        elif tag == "-":
            if not line.startswith("--- "):
                minus.append((old_line, line[1:]))
            old_line += 1
        elif tag == "+":
            if not line.startswith("+++ "):
                plus.append((new_line, old_line - 1, line[1:]))
            new_line += 1

    flush_hunk()
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


def _in_spans(spans: list[list[int]], line: int) -> bool:
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
    if not had:
        return [(tid, ms) for tid, _, _, ms in all_tests]

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
    fixtures = {k: set(v) for k, v in cmap["fixtures"].items()}
    sample_ids = {int(tid) for tid in cmap.get("id_by_tid", {})}
    memory_tids = {
        int(tid) for tid, nodeid in cmap.get("id_by_tid", {}).items()
        if nodeid.endswith(MEMORY_TEST_SUFFIX)
    }
    serialization_prefixes = tuple(cmap.get("serialization_prefixes", ()))

    chosen: set[int] = set()
    fallback_all = False
    explicit_noop = True

    def add_module(rel: str, broad: bool = False) -> None:
        gid = module_groups.get(rel)
        if gid is None or not groups[gid]:
            if broad:
                nonlocal fallback_all
                fallback_all = True
            return
        chosen.update(groups[gid])

    def add_importers(rel: str) -> None:
        for test_file in importers.get(rel, ()):
            chosen.update(file_to_tests.get(test_file, ()))

    def line_keys_at(text: str) -> list[str]:
        return _anchors_for_line(text)

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

    def small_span_tids(rel: str, pos: int, line_keys: set[str]) -> set[int]:
        out: set[int] = set()
        for key in span_keys_at(rel, pos) - line_keys:
            gid = anchor_groups.get(rel, {}).get(key)
            if gid is not None and len(groups[gid]) <= SPAN_ANCHOR_LIMIT:
                out.update(groups[gid])
        return out

    def handle_position(rel: str, pos: int, text: str) -> None:
        nonlocal explicit_noop
        if _trivial(text, enums):
            return
        explicit_noop = False
        spans = scope_groups.get(rel, [])
        in_scope = _span_index(spans, pos)
        if in_scope is not None:
            if SIG_RE.match(text):
                add_module(rel, broad=True)
                add_importers(rel)
                return
            if GLOBAL_RE.match(text):
                add_module(rel, broad=True)
                return
            line_hits = _line_tids(rel, pos, line_runs, groups)
            stmt = _span_index(stmt_groups.get(rel, []), pos)
            if line_hits:
                chosen.update(line_hits)
            elif stmt is not None:
                chosen.update(groups[stmt[2]])
            else:
                chosen.update(groups[in_scope[2]])
            if stmt is not None:
                chosen.update(groups[stmt[2]])
            return

        if _in_spans(doc_spans.get(rel, []), pos):
            return

        if rel.startswith("sqlglot/expressions/") and class_ref_groups.get(rel):
            cls = enclosing_class(rel, pos)
            gid = class_ref_groups[rel].get(cls) if cls else None
            if gid is not None and len(groups[gid]) <= CLASS_REF_LIMIT:
                chosen.update(groups[gid])
                return

        line_keys = set(line_keys_at(text))
        line_tids = tids_for_keys(rel, line_keys)
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
                    add_module(rel, broad=True)
            else:
                chosen.update(line_tids | extra)
            return
        if SIG_RE.match(text) or IMPORT_RE.match(text):
            add_module(rel, broad=True)
            add_importers(rel)
            return
        add_module(rel, broad=True)

    for f in files:
        rel = f.new_path or f.old_path
        if not rel:
            continue

        if f.is_new:
            stem = Path(rel).stem
            dialect_test = f"tests/dialects/test_{stem}.py"
            if dialect_test in file_to_tests:
                chosen.update(file_to_tests[dialect_test])
            add_importers(rel)
            if not f.trivial_only:
                explicit_noop = False
            continue

        if f.is_deleted:
            explicit_noop = False
            add_importers(rel)
            add_module(rel, broad=True)
            continue

        if rel in fixtures:
            if f.changes or not f.trivial_only:
                explicit_noop = False
            chosen.update(fixtures[rel])
            continue

        if rel.startswith(("tests/fixtures/", "tests/data/")):
            explicit_noop = False
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
                fallback_all = True
            continue

        if rel.endswith(".py") and rel.startswith("tests/"):
            explicit_noop = False
            if rel.endswith("conftest.py"):
                fallback_all = True
            else:
                chosen.update(file_to_tests.get(rel, ()))
            continue

        if rel == "sqlglot/__init__.py":
            explicit_noop = False
            add_module(rel, broad=True)
            continue

        if rel.startswith(serialization_prefixes):
            explicit_noop = False
            chosen.update(file_to_tests.get("tests/test_serde.py", ()))
            if rel.endswith("serde.py") or any(
                _in_spans(pickle_spans.get(rel, []), ch.pos) for ch in f.changes
            ):
                chosen.update(pool_tests)

        if rel in line_runs or rel in scope_groups or rel in module_groups:
            if f.changes or not f.trivial_only:
                explicit_noop = False
            for ch in f.changes:
                handle_position(rel, ch.pos, ch.text)
            continue

        if rel.endswith(".py") and rel.startswith("sqlglot/"):
            explicit_noop = False
            fallback_all = True
            continue
        if f.changes:
            explicit_noop = False

    if fallback_all:
        return [(tid, ms) for tid, _, _, ms in all_tests]

    chosen -= memory_tids
    if sample_ids:
        chosen &= sample_ids
    result: list[tuple[int, int]] = []
    ms_by_id = {tid: ms for tid, _, _, ms in all_tests}
    for tid in sorted(chosen):
        if tid in ms_by_id:
            result.append((tid, ms_by_id[tid]))

    if not result and not explicit_noop:
        fastest = min(all_tests, key=lambda t: (t[3], t[0]))
        result.append((fastest[0], fastest[3]))
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
        Path("/app/coverage_map_v3.json.gz"),
        Path("coverage_map_v3.json.gz"),
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
                    (sample_to_runtime[tid], ms) for tid, ms in selected if tid in sample_to_runtime
                ]
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
