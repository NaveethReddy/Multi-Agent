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
    t0 = time.monotonic()
    llm = None
    status, iters, error, res = "n/a", 0, None, None
    try:
        llm = LLMClient(cfg, backend)
        request = task["request"]
        if mode == "multi":
            state = Supervisor(cfg, llm).run(request)
            files, status = state.files, state.status
            iters = sum(h["action"] == "patched" for h in state.history)
            if status == "failed":
                error = state.stop_reason
        else:
            prompt = request + f"\n\nPut everything in a single module named {task['module']}."
            result = run_single(prompt, llm) if mode == "single" else run_single_retry(prompt, llm, cfg)
            files = {task["module"]: result.code}
        if error is None:
            res = run_pytest(files, {"test_hidden.py": task["hidden_test"]}, cfg.sandbox_timeout_s)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    passed = res.passed if res is not None else 0
    total = res.passed + res.failed + res.errors if res is not None else 0
    return {"task": task["id"], "mode": mode, "success": res is not None and res.all_passed,
            "correctness": round(passed / total, 3) if total else 0.0,
            "tests_passed": passed, "tests_total": total, "status": status, "error": error,
            "debug_iterations": iters, "tokens": llm.total_tokens if llm is not None else 0,
            "seconds": round(time.monotonic() - t0, 2)}


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
        if row["error"] is not None:
            print(f"  [{row['task']}] error: {row['error']}", file=sys.stderr)
        print(f"{row['task']:>14}  success={row['success']}  {row['tests_passed']}/{row['tests_total']}"
              f"  tokens={row['tokens']}  {row['seconds']}s")
    n = len(rows)
    summary = {"mode": a.mode, "model": cfg.model, "tasks": n,
               "backend": cfg.backend, "temperature": cfg.temperature, "seed": cfg.seed,
               "error_tasks": sum(r["error"] is not None for r in rows),
               "task_success_rate": round(sum(r["success"] for r in rows) / n, 3),
               "mean_correctness": round(sum(r["correctness"] for r in rows) / n, 3),
               "mean_tokens": round(sum(r["tokens"] for r in rows) / n),
               "mean_seconds": round(sum(r["seconds"] for r in rows) / n, 2)}
    print(json.dumps(summary, indent=2))
    out = a.out or f"{cfg.runs_dir}/custom_{a.mode}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as report:
        json.dump({"summary": summary, "rows": rows}, report, indent=2)


if __name__ == "__main__":
    main()
