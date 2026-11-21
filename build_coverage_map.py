from __future__ import annotations

import argparse
import ast
import concurrent.futures
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

SQLGLOT_REPO = "https://github.com/tobymao/sqlglot.git"
PINNED_COMMIT = "eb11aec4be8cb2122afe713a4816662c3f0fb2fa"

SPLIT_FILES = {
    "tests/test_executor.py": 10,
    "tests/test_optimizer.py": 4,
    "tests/test_parser.py": 3,
}

EXTRA_TEST_FILES_FOR_MODULE: dict[str, tuple[str, ...]] = {
    "sqlglot/serde.py": (
        "tests/test_serde.py",
        "tests/test_optimizer.py",
        "tests/test_executor.py",
        "tests/test_expressions.py",
        "tests/test_lineage.py",
        "tests/dialects/test_databricks.py",
        "tests/dialects/test_teradata.py",
    ),
}


def ensure_sqlglot(work_dir: Path) -> Path:
    repo_dir = (work_dir / ".sqlglot_repo").resolve()
    if not (repo_dir / ".git").exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", SQLGLOT_REPO, str(repo_dir)],
            check=True,
        )
    res = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.stdout.strip() != PINNED_COMMIT:
        subprocess.run(
            ["git", "-C", str(repo_dir), "fetch", "--depth=1", "origin", PINNED_COMMIT],
            check=False,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--detach", PINNED_COMMIT],
            check=True,
        )
    return repo_dir


def path_to_module(rel_path: str) -> str | None:
    p = rel_path.replace("\\", "/")
    if not p.endswith(".py"):
        return None
    if p.endswith("/__init__.py"):
        p = p[: -len("/__init__.py")]
    else:
        p = p[:-3]
    return p.replace("/", ".")


def extract_imports(py_file: Path, current_module: str) -> set[str]:
    try:
        source = py_file.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(py_file))
    except SyntaxError:
        return set()

    imported: set[str] = set()
    pkg_parts = current_module.split(".")[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level > 0:
                base_parts = pkg_parts[: max(0, len(pkg_parts) - (level - 1))]
                if node.module:
                    base_parts = base_parts + node.module.split(".")
                base_mod = ".".join(base_parts)
            else:
                base_mod = node.module or ""

            if base_mod:
                imported.add(base_mod)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if base_mod:
                    imported.add(f"{base_mod}.{alias.name}")
                else:
                    imported.add(alias.name)

    return imported


def build_static_reverse_imports(repo_dir: Path) -> dict[str, set[str]]:
    py_files = sorted(
        list((repo_dir / "sqlglot").rglob("*.py")) + list((repo_dir / "tests").rglob("*.py"))
    )
    all_modules: set[str] = set()
    module_to_rel: dict[str, str] = {}
    imports_out: dict[str, set[str]] = defaultdict(set)

    for fpath in py_files:
        rel = fpath.relative_to(repo_dir).as_posix()
        mod = path_to_module(rel)
        if not mod:
            continue
        all_modules.add(mod)
        module_to_rel[mod] = rel
        imports_out[mod] = extract_imports(fpath, mod)

    reverse_graph: dict[str, set[str]] = defaultdict(set)
    for src_mod, targets in imports_out.items():
        for tgt in targets:
            if tgt in all_modules:
                reverse_graph[tgt].add(src_mod)
            parts = tgt.split(".")
            for i in range(1, len(parts)):
                prefix = ".".join(parts[:i])
                if prefix in all_modules:
                    reverse_graph[prefix].add(src_mod)

    rel_to_dependents: dict[str, set[str]] = {}
    for mod, rel in module_to_rel.items():
        visited: set[str] = set()
        queue: deque[str] = deque([mod])
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            for dep in reverse_graph.get(cur, ()):
                if dep not in visited:
                    queue.append(dep)
        rel_to_dependents[rel] = {module_to_rel[m] for m in visited if m in module_to_rel}

    return rel_to_dependents


def extract_function_spans(repo_dir: Path) -> dict[str, list[tuple[int, int]]]:
    spans_by_rel: dict[str, list[tuple[int, int]]] = {}
    for root_name in ("sqlglot", "tests"):
        root = repo_dir / root_name
        if not root.exists():
            continue
        for fpath in sorted(root.rglob("*.py")):
            rel = fpath.relative_to(repo_dir).as_posix()
            try:
                source = fpath.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(fpath))
            except SyntaxError:
                spans_by_rel[rel] = []
                continue

            spans: list[tuple[int, int]] = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    dec_lines = [d.lineno for d in node.decorator_list if hasattr(d, "lineno")]
                    start_ln = min(dec_lines + [node.lineno])
                    end_ln = getattr(node, "end_lineno", None) or node.lineno
                    spans.append((start_ln, end_ln))
            spans.sort()
            spans_by_rel[rel] = spans
    return spans_by_rel


WORKER_SCRIPT = """
import builtins
import concurrent.futures
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
import pytest

class InlineProcessPoolExecutor:
    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=()):
        if initializer is not None:
            initializer(*initargs)

    def submit(self, fn, *args, **kwargs):
        fut = concurrent.futures.Future()
        try:
            res = fn(*args, **kwargs)
        except BaseException as exc:
            fut.set_exception(exc)
        else:
            fut.set_result(res)
        return fut

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return map(fn, *iterables)

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

concurrent.futures.ProcessPoolExecutor = InlineProcessPoolExecutor

cfg = json.loads(sys.stdin.read())
repo_dir = os.path.abspath(cfg["repo_dir"])
repo_prefix = repo_dir.rstrip("\\\\/") + os.sep
repo_prefix_low = os.path.normcase(repo_prefix)
target_file = cfg["target_file"]
pytest_targets = cfg.get("pytest_targets") or [target_file]
tests_list = cfg["tests"]

nodeid_to_num = {}
base_to_nums = defaultdict(list)
file_to_nums = defaultdict(list)
for t in tests_list:
    num = int(t["number"])
    nid = str(t["id"]).replace("\\\\", "/")
    tfile = str(t["file"]).replace("\\\\", "/")
    nodeid_to_num[nid] = num
    base_to_nums[nid.split("[", 1)[0]].append(num)
    file_to_nums[tfile].append(num)

cov_by_rel = defaultdict(lambda: defaultdict(set))
fixture_to_tids = defaultdict(set)
traced_tids = set()
state = {"current_tids": [], "collect_file": target_file}

def resolve_rel(raw_path):
    if not isinstance(raw_path, (str, os.PathLike)):
        return None
    try:
        p = os.fspath(raw_path)
        if isinstance(p, bytes):
            p = p.decode("utf-8", errors="ignore")
        ap = os.path.abspath(p)
        if os.path.normcase(ap).startswith(repo_prefix_low):
            rel = os.path.relpath(ap, repo_dir).replace("\\\\", "/")
            if not rel.startswith("../") and rel != "..":
                return rel
    except Exception:
        return None
    return None

def record_file(f):
    rel = resolve_rel(f)
    if not rel or rel.endswith(".py") or rel.startswith(".git/"):
        return
    tids = state["current_tids"]
    if tids:
        fixture_to_tids[rel].update(tids)
    elif state["collect_file"] in file_to_nums:
        fixture_to_tids[rel].update(file_to_nums[state["collect_file"]])

orig_open = builtins.open
orig_io_open = io.open
orig_path_open = Path.open

def h_open(f, *a, **kw):
    record_file(f)
    return orig_open(f, *a, **kw)

def h_io_open(f, *a, **kw):
    record_file(f)
    return orig_io_open(f, *a, **kw)

def h_path_open(self, *a, **kw):
    record_file(self)
    return orig_path_open(self, *a, **kw)

code_to_info = {}
tool_id = sys.monitoring.PROFILER_ID
disable = sys.monitoring.DISABLE

def on_py_start(code, instruction_offset):
    info = code_to_info.get(code)
    if info is None:
        rel = resolve_rel(code.co_filename)
        if not rel:
            code_to_info[code] = False
            return disable
        info = (cov_by_rel[rel], code.co_firstlineno)
        code_to_info[code] = info
    elif info is False:
        return disable
    tids = state["current_tids"]
    if tids:
        d, first_ln = info
        d[first_ln].update(tids)
    return None

class Plugin:
    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):
        norm = item.nodeid.replace("\\\\", "/")
        if norm in nodeid_to_num:
            tids = [nodeid_to_num[norm]]
        else:
            tids = base_to_nums.get(norm.split("[", 1)[0], [])
        state["current_tids"] = tids
        if tids:
            traced_tids.update(tids)
        try:
            yield
        finally:
            state["current_tids"] = []

sys.path.insert(0, repo_dir)
os.chdir(repo_dir)
builtins.open = h_open
io.open = h_io_open
Path.open = h_path_open

sys.monitoring.use_tool_id(tool_id, "br_inline_func_tracer")
sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, on_py_start)
sys.monitoring.set_events(tool_id, sys.monitoring.events.PY_START)

try:
    pytest.main(
        ["-q", "--no-header", "--disable-warnings", "--continue-on-collection-errors", "-p", "no:cacheprovider", *pytest_targets],
        plugins=[Plugin()],
    )
finally:
    sys.monitoring.set_events(tool_id, 0)
    sys.monitoring.register_callback(tool_id, sys.monitoring.events.PY_START, None)
    sys.monitoring.free_tool_id(tool_id)
    builtins.open = orig_open
    io.open = orig_io_open
    Path.open = orig_path_open

out = {
    "cov": {r: {str(k): sorted(v) for k, v in d.items()} for r, d in cov_by_rel.items()},
    "fixtures": {r: sorted(v) for r, v in fixture_to_tids.items()},
    "traced": sorted(traced_tids),
}
sys.stdout.write(json.dumps(out))
"""


def trace_job(
    repo_dir: Path,
    job_key: str,
    target_rel_file: str,
    pytest_targets: list[str],
    tests_list: list[dict[str, Any]],
    cache_dir: Path,
    timeout_s: float,
    force: bool = False,
) -> tuple[str, str, dict[str, dict[int, list[int]]], dict[str, list[int]], list[int], bool, float, str]:
    safe_name = job_key.replace("/", "__").replace("\\", "__").replace(":", "_") + ".json.gz"
    cache_file = cache_dir / safe_name

    if not force and cache_file.exists():
        parsed = json.loads(gzip.decompress(cache_file.read_bytes()).decode("utf-8"))
        cov_out = {
            rel: {int(ln): [int(x) for x in tids] for ln, tids in lmap.items()}
            for rel, lmap in parsed.get("cov", {}).items()
        }
        fix_out = {rel: [int(x) for x in tids] for rel, tids in parsed.get("fixtures", {}).items()}
        traced_out = [int(x) for x in parsed.get("traced", [])]
        if traced_out:
            return job_key, target_rel_file, cov_out, fix_out, traced_out, False, 0.0, "CACHE"

    t0 = time.perf_counter()
    payload = {
        "repo_dir": str(repo_dir),
        "target_file": target_rel_file,
        "pytest_targets": pytest_targets,
        "tests": tests_list,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(WORKER_SCRIPT)
        worker_path = tf.name

    try:
        proc = subprocess.run(
            [sys.executable, worker_path],
            input=json.dumps(payload),
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        dt = time.perf_counter() - t0
        return job_key, target_rel_file, {}, {}, [], True, dt, "TIMEOUT"
    finally:
        try:
            os.remove(worker_path)
        except OSError:
            pass

    dt = time.perf_counter() - t0
    if not proc.stdout.strip():
        return job_key, target_rel_file, {}, {}, [], False, dt, "EMPTY"

    last_line = proc.stdout.strip().splitlines()[-1]
    try:
        parsed = json.loads(last_line)
    except json.JSONDecodeError:
        return job_key, target_rel_file, {}, {}, [], False, dt, "PARSE_ERR"

    traced_out = [int(x) for x in parsed.get("traced", [])]
    if traced_out:
        cache_file.write_bytes(gzip.compress(last_line.encode("utf-8"), compresslevel=6))

    cov_out = {
        rel: {int(ln): [int(x) for x in tids] for ln, tids in lmap.items()}
        for rel, lmap in parsed.get("cov", {}).items()
    }
    fix_out = {rel: [int(x) for x in tids] for rel, tids in parsed.get("fixtures", {}).items()}
    return job_key, target_rel_file, cov_out, fix_out, traced_out, False, dt, f"ok in {dt:.1f}s"


def build_compressed_map(
    repo_dir: Path,
    tests_json_path: Path,
    timeout_per_file: float = 240.0,
    workers: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    raw_tests = json.loads(tests_json_path.read_text(encoding="utf-8"))
    tests_list = raw_tests["tests"] if isinstance(raw_tests, dict) and "tests" in raw_tests else raw_tests

    file_to_tests: dict[str, list[int]] = defaultdict(list)
    file_to_nodeids: dict[str, list[str]] = defaultdict(list)
    tid_to_file: dict[int, str] = {}
    for t in tests_list:
        num = int(t["number"])
        tfile = str(t["file"]).replace("\\", "/")
        nid = str(t["id"]).replace("\\", "/")
        file_to_tests[tfile].append(num)
        file_to_nodeids[tfile].append(nid)
        tid_to_file[num] = tfile
    for k in file_to_tests:
        file_to_tests[k].sort()

    static_rev = build_static_reverse_imports(repo_dir)
    func_spans = extract_function_spans(repo_dir)

    cache_dir = Path(".trace_cache")
    if force and cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    jobs: list[tuple[str, str, list[str]]] = []
    for tfile in sorted(file_to_tests.keys()):
        n_parts = SPLIT_FILES.get(tfile, 1)
        if n_parts > 1:
            nids = sorted(set(file_to_nodeids[tfile]))
            chunk_size = max(1, (len(nids) + n_parts - 1) // n_parts)
            for idx in range(0, len(nids), chunk_size):
                chunk = nids[idx : idx + chunk_size]
                jobs.append((f"{tfile}#part{idx // chunk_size + 1}", tfile, chunk))
        else:
            jobs.append((tfile, tfile, [tfile]))

    cov_by_rel: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    fixture_to_tids: dict[str, set[int]] = defaultdict(set)
    traced_tids: set[int] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                trace_job,
                repo_dir,
                job_key,
                tfile,
                ptargets,
                tests_list,
                cache_dir,
                timeout_per_file,
                False,
            )
            for job_key, tfile, ptargets in jobs
        ]
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futures):
            done_cnt += 1
            job_key, tfile, cov_part, fix_part, traced_part, is_to, dt, tag = fut.result()
            for rel, lmap in cov_part.items():
                for ln, tids in lmap.items():
                    cov_by_rel[rel][ln].update(tids)
            for rel, tids in fix_part.items():
                fixture_to_tids[rel].update(tids)
            traced_tids.update(traced_part)

            sys.stderr.write(
                f"[{done_cnt}/{len(jobs)}] {job_key}: {len(traced_part)} tests ({tag})\n"
            )
            sys.stderr.flush()

    completely_untraced_files = [
        tfile
        for tfile, tnums in sorted(file_to_tests.items())
        if not any(t in traced_tids for t in tnums)
    ]
    if completely_untraced_files:
        sys.stderr.write(f"[WARNING] Completely untraced test files: {completely_untraced_files}\n")
        sys.stderr.flush()

    global_fallback_tids: set[int] = set()
    for uf in completely_untraced_files:
        global_fallback_tids.update(file_to_tests[uf])

    untraced_by_file: dict[str, set[int]] = defaultdict(set)
    for num, tfile in tid_to_file.items():
        if num not in traced_tids:
            untraced_by_file[tfile].add(num)

    def augment_with_untraced(tids: set[int]) -> set[int]:
        if not tids:
            return tids
        out = set(tids) | global_fallback_tids
        if untraced_by_file:
            touched_files = {tid_to_file[t] for t in tids if t in tid_to_file}
            for tf in touched_files:
                if tf in untraced_by_file:
                    out.update(untraced_by_file[tf])
        return out

    groups: list[list[int]] = []
    group_index: dict[tuple[int, ...], int] = {}

    def get_gid(tids: set[int] | list[int]) -> int:
        key = tuple(sorted(set(tids)))
        gid = group_index.get(key)
        if gid is None:
            gid = len(groups)
            groups.append(list(key))
            group_index[key] = gid
        return gid

    get_gid(())

    all_tracked_files = sorted(set(cov_by_rel.keys()) | set(func_spans.keys()) | set(static_rev.keys()))
    module_full_group: dict[str, int] = {}
    rle_lines: dict[str, list[list[int]]] = {}
    func_groups: dict[str, list[list[int]]] = {}

    for rel in all_tracked_files:
        line_map = cov_by_rel.get(rel, {})
        dynamic_mod_tests: set[int] = set()
        for tids in line_map.values():
            dynamic_mod_tests.update(tids)

        extra_mod_tests: set[int] = set()
        for extra_tf in EXTRA_TEST_FILES_FOR_MODULE.get(rel, ()):
            extra_mod_tests.update(file_to_tests.get(extra_tf, ()))

        if dynamic_mod_tests and not extra_mod_tests:
            full_tests = augment_with_untraced(dynamic_mod_tests)
        elif extra_mod_tests:
            full_tests = augment_with_untraced(dynamic_mod_tests | extra_mod_tests)
        else:
            static_mod_tests: set[int] = set()
            for dep_rel in static_rev.get(rel, ()):
                if dep_rel in file_to_tests:
                    static_mod_tests.update(file_to_tests[dep_rel])
            full_tests = augment_with_untraced(static_mod_tests)

        full_gid = get_gid(full_tests)
        module_full_group[rel] = full_gid

        runs: list[list[int]] = []
        for ln in sorted(line_map.keys()):
            base_line_tids = line_map[ln] | extra_mod_tests
            gid = get_gid(augment_with_untraced(base_line_tids))
            if runs and runs[-1][1] == ln - 1 and runs[-1][2] == gid:
                runs[-1][1] = ln
            else:
                runs.append([ln, ln, gid])
        rle_lines[rel] = runs

        f_entries: list[list[int]] = []
        for fs, fe in func_spans.get(rel, []):
            fn_tests: set[int] = set()
            for ln in range(fs, fe + 1):
                if ln in line_map:
                    fn_tests.update(line_map[ln])
            fn_tests.update(extra_mod_tests)
            fgid = get_gid(augment_with_untraced(fn_tests)) if fn_tests else full_gid
            f_entries.append([fs, fe, fgid])
        func_groups[rel] = f_entries

    fixtures_clean: dict[str, list[int]] = {
        k: sorted(augment_with_untraced(v)) for k, v in sorted(fixture_to_tids.items()) if v
    }

    return {
        "pinned_commit": PINNED_COMMIT,
        "groups": groups,
        "rle_lines": rle_lines,
        "func_groups": func_groups,
        "module_full_group": module_full_group,
        "file_to_tests": dict(sorted(file_to_tests.items())),
        "fixtures": fixtures_clean,
    }


def main() -> None:
    t0 = time.perf_counter()
    parser = argparse.ArgumentParser(description="Сборка полной функциональной карты покрытия")
    parser.add_argument("--tests", type=Path, default=Path("sample/tests.json"))
    parser.add_argument("--out", type=Path, default=Path("coverage_map.json.gz"))
    parser.add_argument("--timeout-per-file", type=float, default=240.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_dir = ensure_sqlglot(Path("."))
    cov_map = build_compressed_map(
        repo_dir,
        args.tests,
        timeout_per_file=args.timeout_per_file,
        workers=args.workers,
        force=args.force,
    )
    raw = json.dumps(cov_map, separators=(",", ":")).encode("utf-8")
    compressed = gzip.compress(raw, compresslevel=9)
    args.out.write_bytes(compressed)

    elapsed = time.perf_counter() - t0
    sys.stderr.write(
        f"Saved {args.out} ({len(compressed)} bytes) in {elapsed:.1f}s | "
        f"groups={len(cov_map['groups'])} files={len(cov_map['rle_lines'])} fixtures={len(cov_map['fixtures'])}\n"
    )
    sys.stderr.flush()


if __name__ == "__main__":
    main()