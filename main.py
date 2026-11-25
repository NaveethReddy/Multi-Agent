"""CLI entry point.

    python main.py "Create a Python student management system with add, delete, update and search"
    python main.py --demo          # offline scripted demo (no Ollama needed)
"""
from __future__ import annotations

import argparse

from config import default_config
from llm import LLMClient
from supervisor import Supervisor, new_run_dir


def main():
    ap = argparse.ArgumentParser(description="Agentic AI software engineering assistant (Phase-I)")
    ap.add_argument("request", nargs="?", help="plain-English requirement")
    ap.add_argument("--model")
    ap.add_argument("--demo", action="store_true", help="scripted offline demo, no LLM needed")
    ap.add_argument("--max-debug", type=int)
    ap.add_argument("--token-budget", type=int)
    a = ap.parse_args()

    cfg = default_config()
    backend = None
    if a.demo:
        from eval.mock_llm import demo_backend
        cfg.backend, backend = "mock", demo_backend()
        a.request = a.request or ("Create a Python student management system with add, delete, "
                                  "update and search functions.")
    if not a.request:
        ap.error("give a request or use --demo")
    cfg.model = a.model or cfg.model
    cfg.max_debug_iterations = a.max_debug or cfg.max_debug_iterations
    cfg.token_budget = a.token_budget or cfg.token_budget

    llm = LLMClient(cfg, backend)
    run_dir = new_run_dir(cfg, "demo" if a.demo else "run")
    state = Supervisor(cfg, llm).run(a.request, run_dir)

    print(f"\nstatus      : {state.status}  ({state.stop_reason})")
    flow: list[str] = []
    for h in state.history:
        if h["action"] not in ("start", "stop", "change_strategy") and (not flow or flow[-1] != h["agent"]):
            flow.append(h["agent"])
    print("workflow    : " + " -> ".join(flow))
    if state.test_result:
        t = state.test_result
        print(f"tests       : {t.passed} passed, {t.failed} failed, {t.errors} errors")
    if state.review:
        print(f"review      : approved={state.review['approved']}  "
              f"standards_score={state.review['standards_score']}  "
              f"findings={len(state.review['findings'])}")
    per_agent: dict[str, list] = {}
    for m in state.metrics:
        per_agent.setdefault(m["agent"], [0, 0, 0.0])
        per_agent[m["agent"]][0] += m["tokens_in"] + m["tokens_out"]
        per_agent[m["agent"]][1] += 1
        per_agent[m["agent"]][2] += m["seconds"]
    print("metrics     : agent  calls  tokens  seconds")
    for k, (tok, n, sec) in per_agent.items():
        print(f"              {k:<12}{n:<7}{tok:<8}{sec:.2f}")
    print(f"artifacts   : {run_dir}/")


if __name__ == "__main__":
    main()
