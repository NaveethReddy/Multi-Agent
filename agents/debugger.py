from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath

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
    "the expected behavior. A test that asserts on a module-level mutable store instead of "
    "public behavior is invalid: rewrite that assertion to exercise the public API without "
    "changing correct implementation code. Preserve valid acceptance assertions. Make sure "
    "every import the code uses exists."
)
SYSTEM_TESTS_MAY_BE_WRONG = (
    SYSTEM + " The same failure has repeated, so CHANGE STRATEGY: first decide whether the test "
    "itself contradicts the SRS or has invalid setup. If so, output the corrected test file without "
    "weakening valid acceptance assertions; otherwise fix the code in a different way than before."
)


def _repair_missing_module_import(state: SharedState) -> tuple[str, str] | None:
    """Add an import when a generated test references a generated module by qualifier."""
    if state.test_result is None:
        return None
    missing = re.search(
        r"NameError: name ['\"]([A-Za-z_]\w*)['\"] is not defined",
        state.test_result.traceback,
    )
    if not missing:
        return None

    module_name = missing.group(1)
    if not any(
        path.endswith(".py") and PurePosixPath(path).parent == PurePosixPath(".")
        and PurePosixPath(path).stem == module_name
        for path in state.files
    ):
        return None

    for path, source in state.tests.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        if not any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == module_name
            for node in ast.walk(tree)
        ):
            continue

        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.asname or alias.name.split(".")[0]
                                      for alias in node.names)
        if module_name in imported_names:
            continue

        insert_after = 0
        body = tree.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            insert_after = body[0].end_lineno or body[0].lineno
        for node in body:
            if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                insert_after = max(insert_after, node.end_lineno or node.lineno)

        lines = source.splitlines(keepends=True)
        lines.insert(insert_after, f"import {module_name}\n")
        repaired = "".join(lines)
        try:
            ast.parse(repaired)
        except SyntaxError:
            continue
        state.tests[path] = repaired
        return path, module_name
    return None


def _tests_access_internal_state(state: SharedState) -> bool:
    """Detect tests that inspect mutable module globals instead of using the public API."""
    module_globals: dict[str, set[str]] = {}
    for path, source in state.files.items():
        module_path = PurePosixPath(path)
        if not path.endswith(".py") or module_path.parent != PurePosixPath("."):
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        names = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign):
                targets, value = [node.target], node.value
            else:
                continue
            mutable_value = isinstance(
                value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)
            ) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in {"dict", "list", "set"}
            )
            if mutable_value:
                names.update(target.id for target in targets if isinstance(target, ast.Name))
        if names:
            module_globals[module_path.stem] = names

    for source in state.tests.values():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                aliases.update((alias.asname or alias.name.split(".")[0], alias.name.split(".")[0])
                               for alias in node.names)
        if any(
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.attr in module_globals.get(aliases.get(node.value.id, ""), set())
            for node in ast.walk(tree)
        ):
            return True
    return False


class DebuggerAgent(Agent):
    name = "debugger"

    def run(self, state: SharedState) -> None:
        tr = state.test_result
        repaired = _repair_missing_module_import(state)
        if repaired:
            path, module_name = repaired
            state.iteration += 1
            state.log(self.name, "patched", {
                "changed": [path],
                "cause": f"Added missing import for generated module {module_name!r}.",
                "iteration": state.iteration,
            })
            return

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
        tests_editable = (
            no_test_ran or not behaviour_mismatch or tests_read_files_directly
            or _tests_access_internal_state(state)
        )
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
