"""Deterministic scripted LLM backend for OFFLINE testing and demos.

This is NOT a model and its outputs are NOT results. It lets us verify that the
supervisor, sandbox, parsers, static analysis and evaluation harness work
end-to-end without a GPU. Real experiments use the Ollama backend.
"""
from __future__ import annotations

import json
import pathlib

from llm import MockBackend

CUSTOM_DIR = pathlib.Path(__file__).parent / "custom_tasks"


def _agent_of(system: str) -> str:
    s = system.lower()
    for marker, name in (("requirements analyst", "requirement"), ("software architect", "planner"),
                         ("qa engineer", "tester"), ("debugging expert", "debugger"),
                         ("senior code reviewer", "reviewer"),
                         ("assert-based self-tests", "single_tests"),
                         ("reply with only one fenced python code block", "single_code"),
                         ("expert python developer. write clean", "coder")):
        if marker in s:
            return name
    return "unknown"


def file_block(path: str, code: str) -> str:
    return f"### FILE: {path}\n```python\n{code}\n```"


class ScriptedBackend(MockBackend):
    """responses[agent] is a list; each call pops the next item (the last one repeats).
    An item may be a str or a callable(prompt) -> str."""

    def __init__(self, responses: dict[str, list]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.counts: dict[str, int] = {}
        super().__init__(self._handle)

    def _handle(self, system: str, prompt: str, json_mode: bool) -> str:
        agent = _agent_of(system)
        seq = self.responses.get(agent)
        if not seq:
            raise RuntimeError(f"no scripted response for agent '{agent}'")
        i = self.counts.get(agent, 0)
        self.counts[agent] = i + 1
        item = seq[min(i, len(seq) - 1)]
        return item(prompt) if callable(item) else item


def srs_json(title: str, reqs: list[str], accept: list[str]) -> str:
    return json.dumps({"title": title, "functional_requirements": reqs,
                       "constraints": ["standard library only"], "acceptance_criteria": accept})


def plan_json(module: str, purpose: str, funcs: list[str]) -> str:
    return json.dumps({"modules": [{"file": module, "purpose": purpose,
                                    "functions": [{"signature": f, "description": ""} for f in funcs]}],
                       "tasks": [{"id": 1, "file": module, "description": purpose}]})


def review_json(approved: bool = True, findings=None) -> str:
    return json.dumps({"approved": approved, "findings": findings or [], "summary": "mock review"})


def oracle_backend(module: str, code: str, test_code: str, title: str = "task") -> ScriptedBackend:
    """Everything correct on the first try (checks plumbing, not intelligence)."""
    return ScriptedBackend({
        "requirement": [srs_json(title, ["implement the task"], ["tests pass"])],
        "planner": [plan_json(module, title, [])],
        "coder": [file_block(module, code)],
        "tester": [file_block("test_solution.py", test_code)],
        "debugger": [file_block(module, code)],
        "reviewer": [review_json(True)],
        "single_code": [f"```python\n{code}\n```"],
        "single_tests": ["```python\nassert True\n```"],
    })


def humaneval_oracle_backend(problem: dict) -> ScriptedBackend:
    code = problem["prompt"] + problem["canonical_solution"]
    test = (f"from solution import {problem['entry_point']}\n{problem['test']}\n\n"
            f"def test_check():\n    check({problem['entry_point']})\n")
    return oracle_backend("solution.py", code, test, problem["task_id"])


def custom_oracle_backend(task_id: str) -> ScriptedBackend:
    d = CUSTOM_DIR / task_id
    meta = json.loads((d / "task.json").read_text())
    return oracle_backend(meta["module"], (d / "reference.py").read_text(),
                          (d / "hidden_test.py").read_text(), task_id)


BUGGY_STUDENTS = '''class StudentManager:
    def __init__(self):
        self._s = {}

    def add(self, student_id, name, grade):
        self._s[student_id] = {"id": student_id, "name": name, "grade": grade}

    def delete(self, student_id):
        self._s.pop(student_id, None)

    def update(self, student_id, name=None, grade=None):
        rec = self._s[student_id]
        rec["name"] = name or rec["name"]
        rec["grade"] = grade or rec["grade"]

    def search(self, query):
        return [r for r in self._s.values() if query in r["name"]]
'''


def demo_backend() -> ScriptedBackend:
    """Student management demo: first draft has real bugs, the debugger fixes them."""
    d = CUSTOM_DIR / "student_mgmt"
    fixed = (d / "reference.py").read_text()
    return ScriptedBackend({
        "requirement": [srs_json("Student management system",
                                 ["add", "delete", "update", "search students"],
                                 ["duplicate add raises ValueError", "search is case-insensitive",
                                  "unknown id raises KeyError"])],
        "planner": [plan_json("students.py", "StudentManager class",
                              ["def add(self, student_id, name, grade)", "def delete(self, student_id)",
                               "def update(self, student_id, name=None, grade=None)",
                               "def search(self, query) -> list"])],
        "coder": [file_block("students.py", BUGGY_STUDENTS)],
        "tester": [file_block("test_solution.py", (d / "hidden_test.py").read_text())],
        "debugger": ["CAUSE: add() accepts duplicates, search is case-sensitive, delete swallows "
                     "unknown ids and update ignores falsy values.\n" + file_block("students.py", fixed)],
        "reviewer": [review_json(True)],
    })
