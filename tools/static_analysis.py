"""ruff (quality) and bandit (security) wrappers."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.sandbox import write_tree


def _find(tool: str) -> list[str] | None:
    exe = shutil.which(tool)
    if exe:
        return [exe]
    # fall back to module execution
    try:
        __import__(tool)
        return [sys.executable, "-m", tool]
    except ImportError:
        return None


def run_ruff(files: dict[str, str]) -> dict:
    cmd = _find("ruff")
    if not cmd:
        return {"available": False, "issues": []}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_tree(root, files)
        p = subprocess.run(cmd + ["check", "--output-format", "json", "--no-cache", str(root)],
                           capture_output=True, text=True, timeout=60)
        try:
            raw = json.loads(p.stdout or "[]")
        except json.JSONDecodeError:
            raw = []
        issues = [{"file": Path(i["filename"]).name, "line": i["location"]["row"],
                   "code": i["code"], "message": i["message"]} for i in raw]
    return {"available": True, "issues": issues}


def run_bandit(files: dict[str, str]) -> dict:
    cmd = _find("bandit")
    if not cmd:
        return {"available": False, "issues": []}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_tree(root, files)
        p = subprocess.run(cmd + ["-r", str(root), "-f", "json", "-q"],
                           capture_output=True, text=True, timeout=60)
        try:
            raw = json.loads(p.stdout or "{}").get("results", [])
        except json.JSONDecodeError:
            raw = []
        issues = [{"file": Path(i["filename"]).name, "line": i["line_number"],
                   "severity": i["issue_severity"], "confidence": i["issue_confidence"],
                   "test_id": i["test_id"], "message": i["issue_text"]} for i in raw]
    return {"available": True, "issues": issues}


def analyze(files: dict[str, str]) -> dict:
    """Run both tools on non-test source files and summarise a 0-100 standards score."""
    src = {k: v for k, v in files.items() if k.endswith(".py")}
    ruff, bandit = run_ruff(src), run_bandit(src)
    high = [i for i in bandit["issues"] if i["severity"] == "HIGH"]
    med = [i for i in bandit["issues"] if i["severity"] == "MEDIUM"]
    score = 100 - 3 * len(ruff["issues"]) - 25 * len(high) - 10 * len(med) \
        - 3 * (len(bandit["issues"]) - len(high) - len(med))
    return {"ruff": ruff, "bandit": bandit, "standards_score": max(0, score),
            "blocking_security": len(high) + len(med)}
