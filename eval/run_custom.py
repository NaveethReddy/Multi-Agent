"""Custom end-to-end tasks: success = the task's HIDDEN tests (written independently of the
model) pass; correctness = fraction of hidden tests passing.

    python -m eval.run_custom --mode multi            # needs Ollama
    python -m eval.run_custom --mode single --backend mock   # plumbing smoke test only
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config
from eval.baselines import run_single, run_single_retry
from llm import LLMClient
from supervisor import Supervisor
from tools.sandbox import run_pytest

TASKS_DIR = pathlib.Path(__file__).parent / "custom_tasks"


def load_tasks() -> list[dict]:
    out = []
    for d in sorted(TASKS_DIR.iterdir()):
        if (d / "task.json").exists():
            t = json.loads((d / "task.json").read_text())
            t["hidden_test"] = (d / "hidden_test.py").read_text()
            out.append(t)
    return out


def run_task(task: dict, mode: str, cfg, backend=None) -> dict:
    llm = LLMClient(cfg, backend)
    t0 = time.time()
    request = task["request"]
    if mode == "multi":
        state = Supervisor(cfg, llm).run(request)
        files, status, iters = state.files, state.status, state.iteration
    else:
        r = (run_single if mode == "single" else lambda p, l: run_single_retry(p, l, cfg))(
            request + f"\n\nPut everything in a single module named {task['module']}.", llm)
        files, status, iters = {task["module"]: r.code}, "n/a", 0
    res = run_pytest(files, {"test_hidden.py": task["hidden_test"]}, cfg.sandbox_timeout_s)
    total = res.passed + res.failed + res.errors
    return {"task": task["id"], "mode": mode, "success": res.all_passed,
            "correctness": round(res.passed / total, 3) if total else 0.0,
            "tests_passed": res.passed, "tests_total": total, "status": status,
            "debug_iterations": iters, "tokens": llm.total_tokens,
            "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "single_retry", "multi"], default="multi")
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["ollama", "mock"])
    ap.add_argument("--out")
    a = ap.parse_args()
    cfg = default_config()
    cfg.model = a.model or cfg.model
    cfg.backend = a.backend or cfg.backend

    rows = []
    for t in load_tasks():
        backend = None
        if cfg.backend == "mock":
            from eval.mock_llm import custom_oracle_backend
            backend = custom_oracle_backend(t["id"])
        row = run_task(t, a.mode, cfg, backend)
        rows.append(row)
        print(f"{row['task']:>14}  success={row['success']}  {row['tests_passed']}/{row['tests_total']}"
              f"  tokens={row['tokens']}  {row['seconds']}s")
    n = len(rows)
    summary = {"mode": a.mode, "model": cfg.model, "tasks": n,
               "task_success_rate": round(sum(r["success"] for r in rows) / n, 3),
               "mean_correctness": round(sum(r["correctness"] for r in rows) / n, 3),
               "mean_tokens": round(sum(r["tokens"] for r in rows) / n),
               "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2)}
    print(json.dumps(summary, indent=2))
    out = a.out or f"{cfg.runs_dir}/custom_{a.mode}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)


if __name__ == "__main__":
    main()
