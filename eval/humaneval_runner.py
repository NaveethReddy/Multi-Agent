"""HumanEval harness with unbiased Pass@k (Chen et al., 2021).

Usage (real run, needs Ollama + the dataset file):
    python -m eval.humaneval_runner --dataset HumanEval.jsonl --mode multi --n 1 --limit 20
Dataset: https://github.com/openai/human-eval (data/HumanEval.jsonl.gz, gunzip it).
Smoke test without a model:
    python -m eval.humaneval_runner --dataset eval/data/mini_humaneval.jsonl --backend mock
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, default_config
from eval.baselines import run_single, run_single_retry
from llm import LLMClient
from supervisor import Supervisor
from tools.sandbox import run_script


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def load_problems(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_solution(problem: dict, code: str, timeout_s: int) -> bool:
    """Official-style execution: solution, then the benchmark's check(candidate)."""
    program = f"{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})\n"
    rc, _ = run_script(program, timeout_s)
    return rc == 0


def build_prompt(problem: dict) -> str:
    return ("Complete this Python function. Return the full function including the "
            "signature and docstring.\n\n" + problem["prompt"])


def solve(problem: dict, mode: str, cfg: Config, llm: LLMClient) -> tuple[str, int, float]:
    prompt = build_prompt(problem)
    t0 = time.time()
    if mode == "single":
        r = run_single(prompt, llm)
        return r.code, llm.total_tokens, time.time() - t0
    if mode == "single_retry":
        r = run_single_retry(prompt, llm, cfg)
        return r.code, llm.total_tokens, time.time() - t0
    if mode == "multi":
        req = (prompt + f"\n\nThe solution file must define `{problem['entry_point']}` "
               "with exactly this signature.")
        state = Supervisor(cfg, llm).run(req)
        code = "\n\n".join(c for p, c in state.files.items())
        return code, llm.total_tokens, time.time() - t0
    raise ValueError(mode)


def evaluate(problems, mode, cfg, n=1, ks=(1,), backend_factory=None, out_path=None):
    rows = []
    for prob in problems:
        correct = 0
        toks = secs = 0
        for i in range(n):
            c = replace(cfg, seed=cfg.seed + i)
            llm = LLMClient(c, backend_factory(prob) if backend_factory else None)
            try:
                code, tok, sec = solve(prob, mode, c, llm)
                ok = check_solution(prob, code, c.sandbox_timeout_s)
            except Exception as e:  # network/model error counts as a failure, logged
                code, tok, sec, ok = "", llm.total_tokens, 0.0, False
                print(f"  [{prob['task_id']}] error: {e}", file=sys.stderr)
            correct += ok
            toks += tok
            secs += sec
        rows.append({"task_id": prob["task_id"], "n": n, "correct": correct,
                     "tokens": toks, "seconds": round(secs, 2)})
        print(f"{prob['task_id']:>16}  {correct}/{n}  tokens={toks}  {secs:.1f}s")
    summary = {"mode": mode, "model": cfg.model, "problems": len(rows), "n": n,
               **{f"pass@{k}": round(sum(pass_at_k(r["n"], r["correct"], k) for r in rows)
                                     / max(len(rows), 1), 4) for k in ks if k <= n},
               "avg_tokens_per_problem": round(sum(r["tokens"] for r in rows) / max(len(rows), 1) / n),
               "avg_seconds_per_problem": round(sum(r["seconds"] for r in rows) / max(len(rows), 1) / n, 2)}
    print(json.dumps(summary, indent=2))
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "rows": rows}, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["single", "single_retry", "multi"], default="single")
    ap.add_argument("--n", type=int, default=1, help="samples per problem")
    ap.add_argument("--k", type=int, nargs="+", default=[1])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model")
    ap.add_argument("--backend", choices=["ollama", "mock"])
    ap.add_argument("--out")
    a = ap.parse_args()

    cfg = default_config()
    if a.model:
        cfg.model = a.model
    if a.backend:
        cfg.backend = a.backend
    problems = load_problems(a.dataset)[: a.limit]
    factory = None
    if cfg.backend == "mock":
        from eval.mock_llm import humaneval_oracle_backend
        factory = humaneval_oracle_backend
    evaluate(problems, a.mode, cfg, a.n, a.k, factory,
             a.out or f"{cfg.runs_dir}/humaneval_{a.mode}.json")


if __name__ == "__main__":
    main()
