from __future__ import annotations

import re

from agents.base import Agent, parse_file_blocks
from state import SharedState
from tools.sandbox import safe_relpath

SYSTEM = (
    "You are a debugging expert. You get Python source, a failing pytest run and its traceback. "
    "First write one line starting with 'CAUSE:' explaining the root cause. Then output the "
    "COMPLETE corrected file(s) in this format, one block per changed file:\n"
    "### FILE: <path>\n```python\n<full file>\n```\n"
    "Fix the implementation, not the symptom. Keep the public function signatures. "
    "If NO test could run (collection error: the test file itself fails to import or "
    "execute, e.g. pytest fixtures used outside a test function), the test file is the "
    "problem: output the corrected test file. For a test that directly reads or writes data "
    "files instead of using the program API, correct its setup or API usage without weakening "
    "the expected behavior. Preserve valid acceptance assertions. Make sure every import the "
    "code uses exists."
)
SYSTEM_TESTS_MAY_BE_WRONG = (
    SYSTEM + " The same failure has repeated, so CHANGE STRATEGY: first decide whether the test "
    "itself contradicts the SRS or has invalid setup. If so, output the corrected test file without "
    "weakening valid acceptance assertions; otherwise fix the code in a different way than before."
)


class DebuggerAgent(Agent):
    name = "debugger"

    def run(self, state: SharedState) -> None:
        tr = state.test_result
        sys_prompt = SYSTEM_TESTS_MAY_BE_WRONG if state.debug_hint else SYSTEM
        code = "\n\n".join(f"### FILE: {p}\n```python\n{c}```" for p, c in state.files.items())
        tests = "\n\n".join(f"### FILE: {p}\n```python\n{c}```" for p, c in state.tests.items())
        prompt = (f"Acceptance criteria:\n{state.srs.get('acceptance_criteria')}\n\n"
                  f"Source:\n{code}\n\nTests:\n{tests}\n\nFailing run:\n{tr.traceback}")
        if state.debug_hint:
            prompt += f"\n\nSupervisor note: {state.debug_hint}"
        no_test_ran = tr.passed == 0 and tr.failed == 0     # collection error: suspect the tests
        behaviour_mismatch = bool(
            re.search(r"^E\s+(AssertionError|assert\b)|DID NOT RAISE", tr.traceback, re.M)
        )
        tests_read_files_directly = any(
            re.search(r"\bopen\s*\(|\.(?:open|read_text|read_bytes|write_text|write_bytes)\s*\(", test)
            for test in state.tests.values()
        )
        tests_editable = no_test_ran or not behaviour_mismatch or tests_read_files_directly
        resp = self.llm.call(self.name, sys_prompt, prompt)
        cause = (re.search(r"CAUSE:\s*(.+)", resp.text) or [None, ""])[1].strip()
        patched = parse_file_blocks(resp.text)
        changed = []
        for path, content in patched.items():
            safe_relpath(path)
            if path in state.tests and tests_editable:
                state.tests[path] = content
                changed.append(path)
            elif path in state.files:
                state.files[path] = content
                changed.append(path)
        state.iteration += 1
        state.log(self.name, "patched", {"changed": changed, "cause": cause[:300],
                                         "iteration": state.iteration})
