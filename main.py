"""CLI entry point.

    python main.py "Create a Python student management system with add, delete, update and search"
    python main.py --demo          # offline scripted demo (no Ollama needed)
"""
from __future__ import annotations

import argparse

from config import default_config
from llm import LLMClient
from supervisor import Supervisor, new_run_dir


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def main():
    ap = argparse.ArgumentParser(description="Agentic AI software engineering assistant (Phase-I)")
    ap.add_argument("request", nargs="?", help="plain-English requirement")
    ap.add_argument("--model")
    ap.add_argument("--demo", action="store_true", help="scripted offline demo, no LLM needed")
    ap.add_argument("--plan-only", action="store_true", help="stop after saving the requirements and plan")
    ap.add_argument("--llm-timeout", type=positive_int, help="Ollama read timeout in seconds")
    ap.add_argument("--max-output-tokens", type=positive_int, help="cap output tokens per Ollama call")
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
    if a.llm_timeout is not None:
        cfg.llm_timeout_s = a.llm_timeout
    if a.max_output_tokens is not None:
        cfg.max_output_tokens = a.max_output_tokens
    cfg.max_debug_iterations = a.max_debug or cfg.max_debug_iterations
    cfg.token_budget = a.token_budget or cfg.token_budget

    llm = LLMClient(cfg, backend)
    tag = "demo" if a.demo else ("plan" if a.plan_only else "run")
    run_dir = new_run_dir(cfg, tag)
    state = Supervisor(cfg, llm).run(a.request, run_dir, plan_only=a.plan_only)

    print(f"\nstatus      : {state.status}  ({state.stop_reason})")
    print(f"backend     : {cfg.backend} ({'scripted demo' if a.demo else cfg.model})")
    flow = [h["agent"] for h in state.history if h["action"] == "stage_started"]
    for h in state.history:
        if h["action"] == "stage_failed":
            print(f"stopped at  : {h['agent']}")
    print("workflow    : " + " -> ".join(flow))
    print(f"debug cycles: {sum(h['action'] == 'patched' for h in state.history)}")
    print(f"total tokens: {llm.total_tokens}")
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
    return 0 if state.status in ("success", "planned") else 1


if __name__ == "__main__":
    raise SystemExit(main())
