"""Supervisor: workflow state machine with termination rules (design doc, section 5).

`next_action` is a pure function of the shared state, so the routing logic can be
unit-tested without any LLM. The supervisor owns the state; agents only read and
write it.
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from agents.coder import CoderAgent
from agents.debugger import DebuggerAgent
from agents.planner import PlannerAgent
from agents.requirement import RequirementAgent
from agents.reviewer import ReviewerAgent
from agents.tester import TesterAgent
from config import Config
from llm import BudgetExceeded, LLMClient
from state import SharedState
from tools.sandbox import write_tree

STOP = "stop"


def failure_signature(traceback: str) -> str:
    """Stable fingerprint of a failure so repeats can be detected."""
    errs = re.findall(r"^E\s+(.*)$", traceback, re.M)
    if errs:
        return " | ".join(errs[:2])[:200]
    lines = [l for l in traceback.strip().splitlines() if l.strip()]
    return lines[-1][:200] if lines else "unknown"


def next_action(state: SharedState, cfg: Config) -> str:
    """Decide which agent runs next, or STOP (state.status / stop_reason are set)."""
    if state.srs is None:
        return "requirement"
    if state.plan is None:
        return "planner"
    if not state.files:
        return "coder"
    if not state.tests or state.last_agent in ("coder", "debugger"):
        return "tester"

    tr = state.test_result
    if not tr.all_passed:
        if state.iteration >= cfg.max_debug_iterations:
            state.status, state.stop_reason = "partial", \
                f"debug limit reached ({cfg.max_debug_iterations}); tests still failing"
            return STOP
        return "debugger"

    # tests pass
    if state.needs_review:
        return "reviewer"
    if state.review["approved"]:
        state.status, state.stop_reason = "success", "tests pass and reviewer approved"
        return STOP
    if state.review_cycles > cfg.max_review_cycles:
        state.status, state.stop_reason = "partial", \
            f"review rejected after {cfg.max_review_cycles} fix cycles"
        return STOP
    return "coder"   # fixable review findings go back to the coder


class Supervisor:
    def __init__(self, cfg: Config, llm: LLMClient):
        self.cfg = cfg
        self.llm = llm
        self.agents = {
            "requirement": RequirementAgent(llm),
            "planner": PlannerAgent(llm),
            "coder": CoderAgent(llm),
            "tester": TesterAgent(llm, cfg.sandbox_timeout_s, cfg.use_docker),
            "debugger": DebuggerAgent(llm),
            "reviewer": ReviewerAgent(llm),
        }

    def run(self, request: str, run_dir: Optional[str] = None, *,
            plan_only: bool = False) -> SharedState:
        state = SharedState(request=request)
        t_start = time.time()
        state.log("supervisor", "start", {
            "model": self.cfg.model,
            "backend": self.cfg.backend,
            "plan_only": plan_only,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "llm_timeout_s": self.cfg.llm_timeout_s,
            "max_output_tokens": self.cfg.max_output_tokens,
        })
        repeats = 0
        action = None
        max_steps = 60                       # absolute safety net against routing bugs
        try:
            for _ in range(max_steps):
                if time.time() - t_start > self.cfg.wall_clock_timeout_s:
                    state.status, state.stop_reason = "partial", "wall-clock timeout"
                    break
                action = next_action(state, self.cfg)
                if action == STOP:
                    break
                state.log(action, "stage_started")
                self._step(state, action)
                if plan_only and action == "planner":
                    state.status, state.stop_reason = "planned", "requirements and plan generated"
                    break
                repeats = self._track_failures(state, action, repeats)
                if repeats >= 3:             # still the same failure after changing strategy
                    state.status, state.stop_reason = "partial", \
                        "same failure repeated after strategy change; escalate to user"
                    break
            else:
                state.status, state.stop_reason = "failed", "step limit reached"
        except BudgetExceeded as e:
            state.status, state.stop_reason = "partial", f"budget: {e}"
            state.log(action or "supervisor", "stage_failed", state.stop_reason)
        except Exception as e:               # malformed output after repair, unsafe path, ...
            state.status, state.stop_reason = "failed", f"{type(e).__name__}: {e}"
            state.log(action or "supervisor", "stage_failed", state.stop_reason)

        state.metrics = list(self.llm.calls)
        state.log("supervisor", "stop", {"status": state.status, "reason": state.stop_reason,
                                         "tokens": self.llm.total_tokens,
                                         "seconds": round(time.time() - t_start, 2)})
        if run_dir:
            self.save(state, run_dir)
        return state

    # ------------------------------------------------------------------ helpers
    def _step(self, state: SharedState, action: str) -> None:
        if action == "coder" and state.review is not None:
            state.iteration = 0              # fresh debug budget for each review-fix cycle
        self.agents[action].run(state)
        state.last_agent = action
        if action in ("coder", "debugger"):
            state.needs_review = True
        if action == "reviewer":
            state.needs_review = False

    def _track_failures(self, state: SharedState, action: str, repeats: int) -> int:
        if action != "tester" or state.test_result is None:
            return repeats
        if state.test_result.all_passed:
            state.failure_signatures.clear()
            state.debug_hint = ""
            return 0
        sig = failure_signature(state.test_result.traceback)
        if state.failure_signatures and state.failure_signatures[-1] == sig:
            repeats += 1
            if repeats == 1:                 # failed twice in a row: change strategy
                state.debug_hint = ("The same failure occurred again after your last patch. "
                                    "Try a different approach; the test may be wrong.")
                state.log("supervisor", "change_strategy", {"signature": sig})
            elif repeats >= 2:
                repeats = 3                  # third identical failure -> escalate
        else:
            repeats = 0
            state.debug_hint = ""
        state.failure_signatures.append(sig)
        return repeats

    def save(self, state: SharedState, run_dir: str) -> None:
        output = Path(run_dir) / "output"
        output.mkdir(parents=True, exist_ok=True)
        write_tree(output, state.files, state.tests)
        state.dump(os.path.join(run_dir, "state.json"))


def new_run_dir(cfg: Config, tag: str = "run") -> str:
    d = os.path.join(cfg.runs_dir, f"{datetime.now():%Y%m%d-%H%M%S}-{tag}")
    os.makedirs(d, exist_ok=True)
    return d
