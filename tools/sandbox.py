"""Sandboxed code runner.

Generated code is NEVER run in the project directory. Each run gets a fresh temp
directory, a subprocess with a wall-clock timeout, a scrubbed environment and
(on Linux) CPU / memory limits. If Docker is enabled, the run happens in a
container with networking disabled.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from state import TestResult

_SAFE_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "SYSTEMROOT", "TEMP", "TMP")


def _limits():  # pragma: no cover - executes in the child process
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))   # 1 GiB
        resource.setrlimit(resource.RLIMIT_FSIZE, (50 << 20, 50 << 20))
    except Exception:
        pass


def _scrubbed_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"})
    # make obvious network use fail fast (best effort outside Docker)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env[k] = "http://127.0.0.1:9"
    return env


def safe_relpath(rel: str) -> str:
    """Reject absolute paths, drive letters and '..' under BOTH POSIX and Windows rules,
    so the check behaves the same on every OS (on Windows '/etc/x' is not 'absolute')."""
    from pathlib import PurePosixPath, PureWindowsPath
    bad = (not rel or rel.startswith(("/", "\\")) or ":" in rel
           or PurePosixPath(rel).is_absolute() or PureWindowsPath(rel).is_absolute()
           or ".." in PurePosixPath(rel.replace("\\", "/")).parts)
    if bad:
        raise ValueError(f"unsafe path from model output: {rel!r}")
    return rel


def write_tree(root: Path, *trees: dict[str, str]) -> None:
    for tree in trees:
        for rel, content in tree.items():
            safe_relpath(rel)
            dest = root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)


_SUMMARY = re.compile(r"(\d+) (passed|failed|error|errors)")


def parse_pytest(output: str) -> TestResult:
    counts = {"passed": 0, "failed": 0, "errors": 0}
    # Look at the final summary line only
    lines = [l for l in output.strip().splitlines() if l.strip()]
    summary = lines[-1] if lines else ""
    for n, kind in _SUMMARY.findall(summary):
        key = "errors" if kind.startswith("error") else kind
        counts[key] += int(n)
    no_tests = "no tests ran" in summary or (not any(counts.values()))
    tr = TestResult(passed=counts["passed"], failed=counts["failed"], errors=counts["errors"],
                    raw=output[-6000:])
    tr.all_passed = (not no_tests) and tr.failed == 0 and tr.errors == 0 and tr.passed > 0
    if not tr.all_passed:
        tr.traceback = output[-3500:]
    return tr


def run_pytest(files: dict[str, str], tests: dict[str, str], timeout_s: int = 20,
               use_docker: bool = False) -> TestResult:
    with tempfile.TemporaryDirectory(prefix="agentic_run_") as tmp:
        root = Path(tmp)
        write_tree(root, files, tests)
        cmd = [sys.executable, "-I", "-m", "pytest", "-q", "--tb=short", "-x", "--no-header",
               "-p", "no:cacheprovider", "--rootdir", str(root), str(root)]
        env = _scrubbed_env()
        kwargs = {}
        if use_docker:
            cmd = ["docker", "run", "--rm", "--network", "none", "--memory", "1g",
                   "-v", f"{root}:/work", "-w", "/work", "python:3.11-slim",
                   "sh", "-c", "pip install -q pytest >/dev/null 2>&1 && python -m pytest -q --tb=short -x"]
            env = os.environ.copy()
        elif os.name == "posix":
            kwargs["preexec_fn"] = _limits
        # NOTE: `-I` ignores cwd/PYTHONPATH; pytest's rootdir import mode puts the test
        # file's directory on sys.path, which is how tests import the generated modules.
        try:
            p = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True,
                               timeout=timeout_s, **kwargs)
            return parse_pytest(p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            out = out.decode() if isinstance(out, bytes) else out
            return TestResult(errors=1, all_passed=False,
                              traceback=f"TIMEOUT after {timeout_s}s (possible infinite loop)\n{out[-1500:]}",
                              raw="timeout")


def run_script(code: str, timeout_s: int = 20) -> tuple[int, str]:
    """Run a snippet of Python in the sandbox; returns (returncode, combined output)."""
    with tempfile.TemporaryDirectory(prefix="agentic_script_") as tmp:
        f = Path(tmp) / "snippet.py"
        f.write_text(code, encoding="utf-8")
        kwargs = {"preexec_fn": _limits} if os.name == "posix" else {}
        try:
            p = subprocess.run([sys.executable, "-I", str(f)], cwd=tmp, env=_scrubbed_env(),
                               capture_output=True, text=True, timeout=timeout_s, **kwargs)
            return p.returncode, p.stdout + p.stderr
        except subprocess.TimeoutExpired:
            return 124, "TIMEOUT"
