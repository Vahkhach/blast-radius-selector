"""Build the selection map from per-test line coverage (ctrace backend).

Inputs (defaults): _exp/cov/.coverage.*, sample/tests.json, .sqlglot_repo.
Output: coverage_map.json.gz

The map contains only static facts derived from the pinned commit: per-test
line coverage, AST spans, consumer indexes for module/class constants, dialect
inheritance, fixture usage, importer graphs and safe fallback sets. No patch
hash or patch-shape lookup tables.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from coverage import CoverageData

ROOT = Path(__file__).resolve().parent
REPO = ROOT / ".sqlglot_repo"
TESTS = ROOT / "sample" / "tests.json"
COVDIR = ROOT / "_exp" / "cov"
OUT = ROOT / "coverage_map.json.gz"
PINNED_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"

POOL_TEST_MARKERS = ("ProcessPoolExecutor", "multiprocessing", "get_context(")
SERIALIZATION_PREFIXES = ("sqlglot/serde.py", "sqlglot/expressions/")
PICKLE_HOOKS = {"__reduce__", "__reduce_ex__", "__getstate__", "__setstate__", "__deepcopy__", "__copy__"}
DIALECT_DIRS = ("sqlglot/dialects/", "sqlglot/parsers/", "sqlglot/generators/", "sqlglot/typing/")
BASE_MODULE_STEMS = {
    "dialect", "parser", "generator", "tokens", "tokenizer_core", "base", "__init__",
    "helpers", "serde", "typing", "expressions",
}
STMT_NODE_TYPES = (
    ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try,
    ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise, ast.Assert,
    ast.Expr, ast.Delete, ast.Match,
)
COMPOUND_NODE_TYPES = (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try, ast.Match)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
INTROSPECTION_RE = re.compile(r"\b(?:inspect\.|doctest\b|getsource|ast\.parse|tokenize\b)")


def load_tests(tests_path: Path) -> tuple[
    list[dict], dict[str, int], dict[str, list[int]], dict[str, list[int]]
]:
    tests = json.loads(tests_path.read_text(encoding="utf-8"))["tests"]
    nodeid_to_num: dict[str, int] = {}
    base_to_nums: dict[str, list[int]] = defaultdict(list)
    file_to_tests: dict[str, list[int]] = defaultdict(list)
    for t in tests:
        num = int(t["number"])
        nid = str(t["id"]).replace("\\", "/")
        nodeid_to_num[nid] = num
        base_to_nums[nid.split("[", 1)[0]].append(num)
        file_to_tests[str(t["file"]).replace("\\", "/")].append(num)
    return (
        tests,
        nodeid_to_num,
        {k: sorted(v) for k, v in file_to_tests.items()},
        {k: sorted(v) for k, v in base_to_nums.items()},
    )


def context_to_num(context: str, nodeid_to_num: dict[str, int]) -> int | None:
    ctx = context.split("|", 1)[0]
    if not ctx:
        return None
    return nodeid_to_num.get(ctx) or nodeid_to_num.get(ctx.split("[", 1)[0])


def merge_coverage(nodeid_to_num: dict[str, int], covdir: Path) -> dict[str, dict[int, set[int]]]:
    master = CoverageData()
    files = sorted(covdir.glob(".coverage.*"))
    if not files:
        raise SystemExit(f"no coverage data in {covdir}")
    for path in files:
        data = CoverageData(str(path))
        data.read()
        master.update(data)
    repo_prefix = os.path.normcase(str(REPO.resolve())) + os.sep
    hits: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for filename in master.measured_files():
        norm = os.path.normcase(os.path.abspath(filename))
        if not norm.startswith(repo_prefix):
            continue
        rel = os.path.relpath(os.path.abspath(filename), REPO).replace("\\", "/")
        if not rel.startswith("sqlglot/"):
            continue
        for line, contexts in master.contexts_by_lineno(filename).items():
            tids = {
                num for num in (context_to_num(c, nodeid_to_num) for c in contexts)
                if num is not None
            }
            if tids:
                hits[rel][int(line)].update(tids)
    return {rel: dict(lines) for rel, lines in hits.items()}


def ast_indexes() -> dict:
    scopes: dict[str, list[tuple[int, int]]] = {}
    classes: dict[str, list[tuple[int, int, str]]] = {}
    anchors: dict[str, list[tuple[str, int]]] = defaultdict(list)
    anchor_spans: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    refs: dict[str, list[tuple[str, int]]] = defaultdict(list)
    doc_spans: dict[str, list[tuple[int, int]]] = {}
    stmt_spans: dict[str, list[tuple[int, int]]] = {}
    pickle_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    edge_map: dict[str, set[str]] = defaultdict(set)
    classes_by_module: dict[str, set[str]] = defaultdict(set)
    enums: set[str] = set()
    class_bases: dict[str, set[str]] = defaultdict(set)
    ENUM_ROOTS = {"Enum", "StrEnum", "IntEnum", "Flag", "IntFlag", "AutoName"}
    func_defs: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    handler_nodes: dict[tuple[str, int], ast.AST] = {}

    py_files = sorted((REPO / "sqlglot").rglob("*.py"))
    for path in py_files:
        rel = path.relative_to(REPO).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            scopes[rel] = []
            stmt_spans[rel] = []
            continue

        spans: list[tuple[int, int]] = []
        stmts: list[tuple[int, int]] = []
        cls_spans: list[tuple[int, int, str]] = []
        docs: list[tuple[int, int]] = []
        local_class_names: set[str] = set()
        imports: dict[str, str] = {}
        import_symbols: dict[str, str] = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                local_class_names.add(node.name)
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                base_rel = "sqlglot/" + base.replace(".", "/") + ".py"
                if not (REPO / base_rel).exists():
                    base_rel = "sqlglot/" + base.replace(".", "/") + "/__init__.py"
                for alias in node.names:
                    imports[alias.asname or alias.name] = base_rel
                    import_symbols[alias.asname or alias.name] = alias.name
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("sqlglot"):
                        imports[alias.asname or alias.name.split(".")[0]] = (
                            "sqlglot/" + alias.name[len("sqlglot."):].replace(".", "/") + ".py"
                        )

        def add_doc(node: ast.AST) -> None:
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                stmt = body[0]
                docs.append((stmt.lineno, getattr(stmt, "end_lineno", None) or stmt.lineno))

        def collect_assign(stmt: ast.AST) -> None:
            targets = []
            if isinstance(stmt, ast.Assign):
                targets = stmt.targets
            elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                targets = [stmt.target]
            stmt_end = getattr(stmt, "end_lineno", None) or stmt.lineno
            for target in targets:
                if isinstance(target, ast.Name):
                    key = f"n:{target.id}"
                    anchors[rel].append((key, stmt.lineno))
                    anchor_spans[rel].append((stmt.lineno, stmt_end, key))
            value = getattr(stmt, "value", None)
            if isinstance(value, ast.Dict):
                for key_node, value_node in zip(value.keys, value.values):
                    kln = getattr(key_node, "lineno", None)
                    if kln is not None:
                        handler_nodes[(rel, kln)] = value_node
                    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                        key = f"s:{key_node.value}"
                        anchors[rel].append((key, kln if kln is not None else stmt.lineno))
                        v_end = getattr(value_node, "end_lineno", None) or getattr(key_node, "lineno", stmt.lineno)
                        anchor_spans[rel].append((kln if kln is not None else stmt.lineno, v_end, key))

        def walk(node: ast.AST) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    start = child.lineno
                    if not isinstance(child, ast.Lambda):
                        decs = [d.lineno for d in child.decorator_list if hasattr(d, "lineno")]
                        start = min(decs + [start])
                        add_doc(child)
                    end = getattr(child, "end_lineno", None) or start
                    spans.append((start, end))
                    if not isinstance(child, ast.Lambda):
                        func_defs[rel].append((child.name, start, end))
                        if child.name in PICKLE_HOOKS:
                            pickle_spans[rel].append((start, end))
                elif isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    spans.append((child.lineno, getattr(child, "end_lineno", None) or child.lineno))
                if isinstance(child, STMT_NODE_TYPES):
                    end = getattr(child, "end_lineno", None) or child.lineno
                    if isinstance(child, COMPOUND_NODE_TYPES):
                        body = getattr(child, "body", None) or []
                        if body:
                            end = min(end, body[0].lineno - 1)
                    if end > child.lineno:
                        stmts.append((child.lineno, end))
                if isinstance(child, ast.ClassDef):
                    end = getattr(child, "end_lineno", None) or child.lineno
                    cls_spans.append((child.lineno, end, child.name))
                    classes_by_module[rel].add(child.name)
                    for base in child.bases:
                        base_name = None
                        if isinstance(base, ast.Name):
                            base_name = base.id
                        elif isinstance(base, ast.Attribute):
                            base_name = base.attr
                        if base_name:
                            class_bases[child.name].add(import_symbols.get(base_name, base_name))
                    add_doc(child)
                    for stmt in child.body:
                        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                            collect_assign(stmt)
                    for base in child.bases:
                        base_name = None
                        if isinstance(base, ast.Name):
                            base_name = base.id
                        elif isinstance(base, ast.Attribute):
                            base_name = base.attr
                        if base_name and base_name in imports:
                            edge_map[rel].add(imports[base_name])
                        elif base_name and base_name in local_class_names:
                            edge_map[rel].add(rel)
                    walk(child)
                    continue
                walk(child)
                if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node is tree:
                    collect_assign(child)

        add_doc(tree)
        walk(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                refs[f"n:{node.attr}"].append((rel, node.lineno))
            elif isinstance(node, ast.Name):
                refs[f"n:{node.id}"].append((rel, node.lineno))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                refs[f"s:{node.value}"].append((rel, node.lineno))

        scopes[rel] = sorted(set(spans))
        stmt_spans[rel] = sorted(set(stmts))
        classes[rel] = sorted(cls_spans)
        doc_spans[rel] = sorted(set(docs))

    for _ in range(50):
        changed = False
        for name, bases in class_bases.items():
            if name in enums:
                continue
            if any(b in ENUM_ROOTS or b in enums for b in bases):
                enums.add(name)
                changed = True
        if not changed:
            break
    enums -= ENUM_ROOTS

    handler_table: list[tuple[str, int, str, int, int]] = []
    for (rel, line), node in handler_nodes.items():
        if isinstance(node, ast.Lambda):
            hs = getattr(node, "lineno", None)
            if hs is not None:
                he = getattr(node, "end_lineno", None) or hs
                handler_table.append((rel, line, rel, hs, he))
            continue
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        if not name:
            continue
        for fname, hs, he in func_defs.get(rel, ()):
            if fname == name:
                handler_table.append((rel, line, rel, hs, he))

    return {
        "scopes": scopes,
        "stmt_spans": stmt_spans,
        "pickle_spans": {k: sorted(set(v)) for k, v in pickle_spans.items()},
        "classes": classes,
        "anchors": {k: sorted(set(v)) for k, v in anchors.items()},
        "anchor_spans": {k: sorted(set(v)) for k, v in anchor_spans.items()},
        "refs": refs,
        "doc_spans": doc_spans,
        "edges": edge_map,
        "enums": sorted(enums),
        "dict_handlers": handler_table,
    }


def _test_method_spans(rel: str, path: Path) -> list[tuple[int, int, str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    spans: list[tuple[int, int, str]] = []

    def walk(node: ast.AST, cls: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, child.name)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name.startswith("test"):
                    prefix = f"{rel}::{cls}::{child.name}" if cls else f"{rel}::{child.name}"
                    end = getattr(child, "end_lineno", None) or child.lineno
                    spans.append((child.lineno, end, prefix))
                walk(child, cls)
            else:
                walk(child, cls)

    walk(tree, None)
    return spans


def build_dialect_indexes(
    idx: dict,
    file_to_tests: dict[str, list[int]],
    base_to_nums: dict[str, list[int]],
    fixture_map: dict[str, list[int]],
) -> dict:
    edges = idx["edges"]
    children: dict[str, set[str]] = defaultdict(set)
    for child, bases in edges.items():
        for base in bases:
            children[base].add(child)

    dialects: dict[str, set[str]] = {}

    def stems(rel: str) -> set[str]:
        for prefix in DIALECT_DIRS:
            if rel.startswith(prefix):
                stem = Path(rel).stem
                if stem not in BASE_MODULE_STEMS:
                    return {stem}
                return set()
        return set()

    for rel in set(edges) | set(children):
        seen = {rel}
        queue = [rel]
        out: set[str] = set()
        while queue:
            cur = queue.pop()
            out |= stems(cur)
            for nxt in children.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if out:
            dialects[rel] = sorted(out)

    dialect_tests: dict[str, list[int]] = {}
    known_stems = {s for ss in dialects.values() for s in ss}
    known_stems |= {Path(r).stem[len("test_"):] for r in file_to_tests if r.startswith("tests/dialects/test_")}
    mentioned: dict[str, set[int]] = defaultdict(set)
    for path in sorted((REPO / "tests").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel not in file_to_tests:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        spans = _test_method_spans(rel, path)
        for m in re.finditer(r"""(?:read|write|dialect)\s*=\s*["']([A-Za-z0-9_]+)["']""", text):
            stem = m.group(1).lower()
            if stem not in known_stems:
                continue
            line = text.count("\n", 0, m.start()) + 1
            owner = None
            for start, end, prefix in spans:
                if start <= line <= end and (owner is None or end - start < owner[1] - owner[0]):
                    owner = (start, end, prefix)
            if owner is not None:
                nums = base_to_nums.get(owner[2])
                if nums:
                    mentioned[stem].update(nums)
                    continue
            mentioned[stem].update(file_to_tests[rel])
    for stem in known_stems | set(mentioned):
        tids = set(file_to_tests.get(f"tests/dialects/test_{stem}.py", ()))
        tids |= mentioned.get(stem, set())
        if tids:
            dialect_tests[stem] = sorted(tids)

    fixture_dialects: dict[str, set[str]] = defaultdict(set)
    for rel in fixture_map:
        path = REPO / rel
        if not path.exists() or path.suffix.lower() not in (".sql", ".json", ".txt", ".csv"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        markers = {
            m.group(1).lower()
            for m in re.finditer(r"^#\s*dialect:\s*([A-Za-z0-9_]+)", text, re.MULTILINE)
        }
        tokens = set(re.split(r"[^a-z0-9_]+", rel.lower()))
        for stem in markers:
            if len(markers) == 1 or stem in tokens:
                fixture_dialects[stem].add(rel)
        for stem in known_stems & tokens:
            fixture_dialects[stem].add(rel)
    fixture_dialect_tests: dict[str, list[int]] = {}
    for stem, rels in fixture_dialects.items():
        tids: set[int] = set()
        for rel in rels:
            tids.update(fixture_map.get(rel, ()))
        if tids:
            fixture_dialect_tests[stem] = sorted(tids)

    return {"dialects": dialects, "dialect_tests": dialect_tests, "fixture_dialect_tests": fixture_dialect_tests}


def build_keyword_tests(anchor_keys: set[str], file_to_tests: dict[str, list[int]], fixture_map: dict[str, list[int]]) -> dict[str, list[int]]:
    wanted = {k[2:].lower() for k in anchor_keys if k.startswith("s:") and len(k[2:]) >= 3}
    wanted |= {k[2:].lower() for k in anchor_keys if k.startswith("n:") and len(k[2:]) >= 4}
    hits: dict[str, set[int]] = defaultdict(set)
    for rel in fixture_map:
        path = REPO / rel
        if not path.exists() or path.suffix.lower() not in (".sql", ".json", ".txt", ".csv"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        words = {w.lower() for w in WORD_RE.findall(text)}
        for word in words & wanted:
            hits[word].update(fixture_map[rel])
    return {w: sorted(t) for w, t in hits.items()}


def build_importers(file_to_tests: dict[str, list[int]]) -> dict[str, list[str]]:
    module_to_rel: dict[str, str] = {}
    graph_out: dict[str, set[str]] = defaultdict(set)
    py_files = sorted((REPO / "sqlglot").rglob("*.py")) + sorted((REPO / "tests").rglob("*.py"))
    trees: dict[str, ast.AST] = {}
    for path in py_files:
        rel = path.relative_to(REPO).as_posix()
        if rel.endswith("/__init__.py"):
            mod = rel[: -len("/__init__.py")].replace("/", ".")
        else:
            mod = rel[:-3].replace("/", ".")
        module_to_rel[mod] = rel
        try:
            trees[rel] = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

    for rel, tree in trees.items():
        pkg = rel.split("/")[:-1]
        targets: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                level = node.level or 0
                base = pkg[: max(0, len(pkg) - (level - 1))] if level else []
                if node.module:
                    base = base + node.module.split(".")
                base_mod = ".".join(base)
                if base_mod:
                    targets.add(base_mod)
                for alias in node.names:
                    if alias.name != "*" and base_mod:
                        targets.add(f"{base_mod}.{alias.name}")
        graph_out[rel].update(targets)

    reverse: dict[str, set[str]] = defaultdict(set)
    rel_to_module = {r: m for m, r in module_to_rel.items()}
    for src_rel, targets in graph_out.items():
        if src_rel not in rel_to_module:
            continue
        for tgt in targets:
            parts = tgt.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix in module_to_rel:
                    reverse[module_to_rel[prefix]].add(src_rel)

    importer_tests: dict[str, list[str]] = {}
    for rel in module_to_rel.values():
        seen: set[str] = set()
        queue = [rel]
        while queue:
            cur = queue.pop()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(reverse.get(cur, ()))
        importer_tests[rel] = sorted(f for f in seen if f in file_to_tests)
    return importer_tests


def pool_tests(file_to_tests: dict[str, list[int]]) -> list[int]:
    out: set[int] = set()
    for path in sorted((REPO / "tests").glob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel not in file_to_tests:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in POOL_TEST_MARKERS):
            out.update(file_to_tests[rel])
    return sorted(out)


def introspection_tests(
    file_to_tests: dict[str, list[int]], base_to_nums: dict[str, list[int]]
) -> list[int]:
    out: set[int] = set()
    for path in sorted((REPO / "tests").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if rel not in file_to_tests:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        spans = _test_method_spans(rel, path)
        for m in INTROSPECTION_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            owner = None
            for start, end, prefix in spans:
                if start <= line <= end and (owner is None or end - start < owner[1] - owner[0]):
                    owner = (start, end, prefix)
            if owner is not None:
                nums = base_to_nums.get(owner[2])
                if nums:
                    out.update(nums)
                    continue
            out.update(file_to_tests[rel])
    return sorted(out)


def main() -> None:
    global REPO
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--cov-dir", type=Path, default=COVDIR)
    parser.add_argument("--tests", type=Path, default=TESTS)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()

    REPO = args.repo.resolve()
    out_path = args.out if args.out.is_absolute() else (ROOT / args.out)
    covdir = args.cov_dir if args.cov_dir.is_absolute() else (ROOT / args.cov_dir)

    tests, nodeid_to_num, file_to_tests, base_to_nums = load_tests(args.tests)
    hits = merge_coverage(nodeid_to_num, covdir)
    idx = ast_indexes()
    scopes = idx["scopes"]
    stmt_spans = idx["stmt_spans"]
    classes = idx["classes"]
    anchors = idx["anchors"]
    anchor_spans = idx["anchor_spans"]
    refs = idx["refs"]
    doc_spans = idx["doc_spans"]

    memory_tid = next(
        (int(t["number"]) for t in tests if str(t["id"]).endswith("::test_generator_memory")),
        None,
    )

    groups: list[list[int]] = []
    group_index: dict[tuple[int, ...], int] = {}

    def gid(tids: set[int] | list[int]) -> int:
        key = tuple(sorted(set(tids)))
        found = group_index.get(key)
        if found is None:
            found = len(groups)
            groups.append(list(key))
            group_index[key] = found
        return found

    def union_lines(rel: str, start: int, end: int) -> set[int]:
        out: set[int] = set()
        line_map = hits.get(rel, {})
        for line in range(start, end + 1):
            out |= line_map.get(line, set())
        return out

    scope_groups: dict[str, list[list[int]]] = {}
    for rel, spans in scopes.items():
        entries = []
        for start, end in spans:
            tids = union_lines(rel, start, end)
            if tids:
                entries.append([start, end, gid(tids)])
        scope_groups[rel] = entries

    stmt_groups: dict[str, list[list[int]]] = {}
    for rel, spans in stmt_spans.items():
        entries = []
        for start, end in spans:
            tids = union_lines(rel, start, end)
            if tids:
                entries.append([start, end, gid(tids)])
        stmt_groups[rel] = entries

    line_runs: dict[str, list[list[int]]] = {}
    for rel, line_map in hits.items():
        runs: list[list[int]] = []
        for line in sorted(line_map):
            g = gid(line_map[line])
            if runs and runs[-1][1] == line - 1 and runs[-1][2] == g:
                runs[-1][1] = line
            else:
                runs.append([line, line, g])
        line_runs[rel] = runs

    module_groups: dict[str, int] = {}
    for rel in sorted(set(hits) | set(scopes)):
        tids: set[int] = set()
        for line_tids in hits.get(rel, {}).values():
            tids |= line_tids
        module_groups[rel] = gid(tids)

    ref_cache: dict[tuple[str, int], set[int]] = {}

    def tests_for_ref(rel: str, line: int) -> set[int]:
        key = (rel, line)
        cached = ref_cache.get(key)
        if cached is not None:
            return cached
        direct = hits.get(rel, {}).get(line, set())
        if direct:
            out = set(direct)
        else:
            out = set()
            for start, end in scopes.get(rel, ()):
                if start <= line <= end:
                    out |= union_lines(rel, start, end)
                    break
        ref_cache[key] = out
        return out

    handler_index: dict[tuple[str, int], list[tuple[str, int, int]]] = defaultdict(list)
    for rel, line, hrel, hs, he in idx["dict_handlers"]:
        handler_index[(rel, line)].append((hrel, hs, he))

    scope_membership: dict[tuple[str, int], bool] = {}

    def in_function(rel: str, line: int) -> bool:
        key = (rel, line)
        cached = scope_membership.get(key)
        if cached is not None:
            return cached
        found = any(start <= line <= end for start, end in scopes.get(rel, ()))
        scope_membership[key] = found
        return found

    def consumer_tids(ref_rel: str, ref_line: int) -> set[int]:
        if in_function(ref_rel, ref_line):
            return tests_for_ref(ref_rel, ref_line)
        out: set[int] = set()
        for hrel, hs, he in handler_index.get((ref_rel, ref_line), ()):
            out |= union_lines(hrel, hs, he)
        return out

    anchor_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for rel, entries in anchors.items():
        for key, line in entries:
            tids: set[int] = set()
            keys = [key]
            if key.startswith("n:"):
                keys.append("s:" + key[2:])
            for ref_key in keys:
                for ref_rel, ref_line in refs.get(ref_key, ()):
                    if ref_rel == rel and abs(ref_line - line) <= 1:
                        continue
                    tids |= consumer_tids(ref_rel, ref_line)
            cls = None
            for start, end, name in classes.get(rel, ()):
                if start <= line <= end:
                    cls = name
                    break
            if cls:
                for ref_rel, ref_line in refs.get(f"n:{cls}", ()):
                    if ref_rel == rel and abs(ref_line - line) <= 1:
                        continue
                    tids |= consumer_tids(ref_rel, ref_line)
            if tids:
                anchor_groups[rel][key] = gid(tids)

    class_ref_groups: dict[str, dict[str, int]] = defaultdict(dict)
    for rel, cls_spans in classes.items():
        for _, _, name in cls_spans:
            tids: set[int] = set()
            for ref_rel, ref_line in refs.get(f"n:{name}", ()):
                tids |= consumer_tids(ref_rel, ref_line)
            if tids:
                class_ref_groups[rel][name] = gid(tids)

    fixtures: dict[str, list[int]] = {}
    for candidate in (
        ROOT / "_exp" / "coverage_map_v1.json.gz",
        ROOT / "_exp" / "coverage_map_v2.json.gz",
        out_path,
    ):
        if not candidate.exists():
            continue
        try:
            old = json.loads(gzip.decompress(candidate.read_bytes()).decode("utf-8"))
        except Exception:
            continue
        for key, tids in old.get("fixtures", {}).items():
            if not tids or not (REPO / key).exists():
                continue
            fixtures[key] = sorted(set(tids))
        if fixtures:
            break

    dialect_idx = build_dialect_indexes(idx, file_to_tests, base_to_nums, fixtures)
    all_anchor_keys = {k for per_file in anchors.values() for k, _ in per_file}
    keyword_tests = build_keyword_tests(all_anchor_keys, file_to_tests, fixtures)
    importers = build_importers(file_to_tests)
    pools = pool_tests(file_to_tests)
    intro = introspection_tests(file_to_tests, base_to_nums)

    def module_union(rels: tuple[str, ...]) -> set[int]:
        out: set[int] = set()
        for rel in rels:
            for line_tids in hits.get(rel, {}).values():
                out |= line_tids
        return out

    core_rels = (
        "sqlglot/parser.py",
        "sqlglot/generator.py",
        "sqlglot/tokens.py",
        "sqlglot/__init__.py",
        "sqlglot/dialects/dialect.py",
        "sqlglot/optimizer/optimizer.py",
    )
    core = module_union(core_rels)
    for rel in hits:
        if rel.startswith("sqlglot/expressions/"):
            core |= module_union((rel,))
    executor_tids = set(file_to_tests.get("tests/test_executor.py", ()))
    core -= executor_tids
    if memory_tid is not None:
        core.discard(memory_tid)
    dialect_core = module_union(("sqlglot/dialects/dialect.py", "sqlglot/dialects/__init__.py"))
    if memory_tid is not None:
        dialect_core.discard(memory_tid)

    payload = {
        "pinned_commit": PINNED_COMMIT,
        "groups": groups,
        "line_runs": line_runs,
        "scope_groups": scope_groups,
        "stmt_groups": stmt_groups,
        "module_groups": module_groups,
        "anchor_groups": dict(anchor_groups),
        "class_ref_groups": {k: dict(v) for k, v in class_ref_groups.items()},
        "anchor_spans": anchor_spans,
        "class_spans": classes,
        "doc_spans": doc_spans,
        "pickle_spans": idx["pickle_spans"],
        "file_to_tests": file_to_tests,
        "id_by_tid": {str(t["number"]): str(t["id"]).replace("\\", "/") for t in tests},
        "importers": importers,
        "pool_tests": pools,
        "introspection_tests": intro,
        "fixtures": fixtures,
        "dialects": dialect_idx["dialects"],
        "dialect_tests": dialect_idx["dialect_tests"],
        "fixture_dialect_tests": dialect_idx["fixture_dialect_tests"],
        "keyword_tests": keyword_tests,
        "enum_classes": idx["enums"],
        "memory_test": memory_tid,
        "core_tests": sorted(core),
        "dialect_core_tests": sorted(dialect_core),
        "serialization_prefixes": list(SERIALIZATION_PREFIXES),
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    out_path.write_bytes(gzip.compress(raw, compresslevel=9))

    covered = sum(len(v) for v in hits.values())
    print(
        f"files={len(hits)} covered_lines={covered} groups={len(groups)} "
        f"stmt_spans={sum(len(v) for v in stmt_groups.values())} "
        f"anchors={sum(len(v) for v in anchor_groups.values())} "
        f"keywords={len(keyword_tests)} dialects={len(dialect_idx['dialect_tests'])} "
        f"fixtures={len(fixtures)} pool_tests={len(pools)} intro={len(intro)} "
        f"core={len(core)} dialect_core={len(dialect_core)} "
        f"raw={len(raw) / 1e6:.1f}MB gz={out_path.stat().st_size / 1e6:.2f}MB"
    )


if __name__ == "__main__":
    main()
