"""Single-agent baselines (design doc, section 10).

Same model, temperature and token budget as the multi-agent system, so any
difference comes from the architecture.

  single        one prompt, one model, no tools
  single_retry  same, plus a test-retry loop: the model writes its own asserts, runs
                them in the sandbox, and is shown the traceback on failure.
                It never sees hidden/benchmark tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from config import Config
from llm import BudgetExceeded, LLMClient, extract_code
from tools.sandbox import run_script

SYSTEM_CODE = ("You are an expert Python developer. Reply with ONLY one fenced python code "
               "block containing the complete solution. Standard library only.")
SYSTEM_TESTS = ("You write short Python assert-based self-tests. Reply with ONLY one fenced "
                "python code block of top-level assert statements that call the solution "
                "(it is already defined above your code). Cover edge cases.")


@dataclass
class BaselineResult:
    code: str
    mode: str
    tokens: int = 0
    seconds: float = 0.0
    attempts: int = 1
    notes: list = field(default_factory=list)


def run_single(task_prompt: str, llm: LLMClient) -> BaselineResult:
    t0 = time.time()
    resp = llm.call("single", SYSTEM_CODE, task_prompt)
    return BaselineResult(extract_code(resp.text), "single", llm.total_tokens, time.time() - t0)


def run_single_retry(task_prompt: str, llm: LLMClient, cfg: Config,
                     max_attempts: int = 3) -> BaselineResult:
    t0 = time.time()
    code = extract_code(llm.call("single", SYSTEM_CODE, task_prompt).text)
    tests = extract_code(llm.call("single", SYSTEM_TESTS,
                                  f"Task:\n{task_prompt}\n\nSolution:\n```python\n{code}```").text)
    attempts, notes = 1, []
    try:
        while True:
            rc, out = run_script(code + "\n\n" + tests, cfg.sandbox_timeout_s)
            notes.append("pass" if rc == 0 else "fail")
            if rc == 0 or attempts >= max_attempts:
                break
            attempts += 1
            code = extract_code(llm.call(
                "single", SYSTEM_CODE,
                f"{task_prompt}\n\nYour previous solution:\n```python\n{code}```\n"
                f"It failed your self-tests with:\n{out[-1500:]}\nReturn the corrected full solution."
            ).text)
    except BudgetExceeded:
        notes.append("budget")
    return BaselineResult(code, "single_retry", llm.total_tokens, time.time() - t0,
                          attempts, notes)
