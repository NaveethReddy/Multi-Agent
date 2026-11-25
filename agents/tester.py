from __future__ import annotations

import json

from agents.base import Agent, parse_files_or_single
from state import SharedState
from tools.sandbox import run_pytest

SYSTEM = (
    "You are a QA engineer. Write a pytest test file for the given code, derived from the SRS "
    "acceptance criteria (not by copying the implementation's behaviour). Cover specified normal, "
    "edge and error cases. Import every module used from the module files shown. Tests must be "
    "deterministic and must not touch the network or filesystem outside tmp_path. "
    "Use pytest fixtures (tmp_path, monkeypatch) ONLY as parameters of test functions, never at "
    "module level. If the code uses relative file paths, include this autouse fixture after the "
    "imports so each test runs in tmp_path:\n"
    "@pytest.fixture(autouse=True)\n"
    "def _isolate(tmp_path, monkeypatch):\n"
    "    monkeypatch.chdir(tmp_path)\n"
    "If the code has module-level mutable state, import its module (for example, `import student`, "
    "not `from student import students`) and reset the actual store to a fresh value in the autouse "
    "fixture (for example, `monkeypatch.setattr(student, 'students', [])`). Adapt the module, store "
    "name and empty value to the code. Changing the working directory does not reset Python globals. "
    "Call the public functions in tests; do not assert on internal stores. "
    "Assert behavior through the program's functions; never open or inspect its data files "
    "directly. Do not test behavior the SRS does not state. "
    "Reply with ONLY:\n### FILE: test_solution.py\n```python\n<code>\n```"
)
TEST_PATH = "test_solution.py"


class TesterAgent(Agent):
    name = "tester"

    def __init__(self, llm, sandbox_timeout_s: int = 20, use_docker: bool = False):
        super().__init__(llm)
        self.timeout = sandbox_timeout_s
        self.use_docker = use_docker

    def write_tests(self, state: SharedState) -> None:
        code_listing = "\n\n".join(f"# {p}\n{c}" for p, c in state.files.items())
        prompt = ("SRS:\n" + json.dumps(state.srs, indent=2) + "\n\nPlan:\n"
                  + json.dumps(state.plan, indent=2) + "\n\nCode:\n" + code_listing)
        resp = self.llm.call(self.name, SYSTEM, prompt)
        files = parse_files_or_single(resp.text, TEST_PATH)
        state.tests = {TEST_PATH: next(iter(files.values()))}
        state.log(self.name, "wrote_tests", {"files": list(state.tests)})

    def run_tests(self, state: SharedState) -> None:
        res = run_pytest(state.files, state.tests, self.timeout, self.use_docker)
        state.test_result = res
        state.log(self.name, "ran_tests", {"passed": res.passed, "failed": res.failed,
                                           "errors": res.errors, "all_passed": res.all_passed})

    def run(self, state: SharedState) -> None:
        if not state.tests:
            self.write_tests(state)
        self.run_tests(state)
