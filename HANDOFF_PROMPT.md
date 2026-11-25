# Handoff prompt: finish Phase-I of the Agentic SE Assistant (paste this to the next agent)

You are continuing an M.Tech. CSE final-year project (Phase-I, Review-1). Work from the attached
project ZIP, which may be an older snapshot. Do not assume it contains any fix listed in section 3;
apply them if absent. Be honest: never invent benchmark numbers, and never present output from the
scripted mock backend as model results.

## 1. The project

**Title:** Agentic AI-Based Autonomous Software Engineering Assistant using Multi-Agent LLMs.
**Student:** [Name and SRN omitted; request separately only if necessary.] Individual project. Guide and
Project ID are still blank.
**Idea:** a supervisor-controlled multi-agent pipeline that turns a plain-English requirement into
tested, reviewed Python code, then compares it fairly against single-agent and human-assisted modes.

**Pipeline (all in the zip):** Requirement -> Planner -> Coder -> Tester -> (tests pass? no: Debugger -> Tester;
yes: Reviewer -> approve, or reject back to Coder). The Supervisor (`supervisor.py`, pure `next_action()` state
machine) owns the shared state (`state.py`). Agents exchange validated JSON or `### FILE:` blocks, not free chat.
Tools: `tools/sandbox.py` (temp dir, subprocess timeout, rlimits, scrubbed env, path guard), `tools/static_analysis.py`
(ruff + bandit). LLM: `llm.py`, Ollama at `OLLAMA_URL` (default `http://localhost:11434`), model from
`AGENTIC_MODEL` (default `qwen2.5-coder:7b`), with token and latency accounting and a one-shot JSON repair retry.
Limits: 5 debug iterations, 2 review cycles, token budget, wall-clock timeout; a repeated failure makes the
supervisor change strategy, and a third identical failure escalates (status `partial`).
Evaluation (`eval/`): `humaneval_runner.py` (unbiased Pass@k), `run_custom.py` (5 custom tasks, hidden tests),
`baselines.py` (`single`, `single_retry`), `mock_llm.py` (SCRIPTED backend, plumbing only).

**Phase-I scope (what "50%" means):** literature survey, architecture, core-pipeline prototype, evaluation
plan and harness, baseline evaluation, Phase-I report. Phase-II (NOT now): RAG memory, documentation agent,
explainability, dashboard, SWE-bench subset.

## 2. Where things stand

- Ollama connectivity and qwen2.5-coder:7b inference work, but five real runs of
  `python main.py "Create a Python student management system with add, delete, update and search functions."`
  ended `partial` (escalated), not success. From their `state.json` logs:
  1. Run `runs/20261125-042755-run`: 6 model calls, 8,076 tokens, 114.22 seconds. The tester used
     `monkeypatch.setattr(...)` at module level, causing collection to fail. The generated `student.py`
     also used `json` without importing it, but collection failed before that defect was exercised.
  2. Run `runs/20261125-045510-run`: 8 model calls, 11,480 tokens, 170.16 seconds. The test read a
     `students.json` under `tmp_path`, while the implementation wrote a relative `students.json` in
     the sandbox working directory. The debugger repeatedly changed code instead of correcting the
     test's file isolation/access.
  3. Run `runs/20261125-053718-run`: 6 model calls, 6,684 tokens, 117.64 seconds. The implementation
     used a module-level in-memory dictionary, but the tests did not reset it between functions.
     Changing to `tmp_path` did not isolate Python module state, so a later test hit a duplicate-ID
     `ValueError`; the debugger's patches did not resolve the repeated failure.
  4. Run `runs/20261125-054048-run`: 9 model calls, 11,753 tokens, 136.39 seconds. The implementation
     used a module-level `students` list, but the test imported that list directly and asserted on
     internal state. Its fixture changed directories but did not reset the module's list, so
     `test_delete_student` failed after an earlier test populated it; the debugger repeated an
     incorrect diagnosis until the debug limit.
  5. Run `runs/20261125-054417-run`: 6 model calls, 7,480 tokens, 99.09 seconds. The tester generated a
     fixture that patched `student.students` but did not import the `student` module, then referenced
     `student.students`; the resulting `NameError` persisted through two debugger attempts.
- A separate `qwen2.5-coder:14b` smoke attempt returned an Ollama HTTP 500 because its Windows CUDA
  initialization failed; it did not produce a workflow result.
- No HumanEval or custom-task numbers exist yet. The report says so under "Not yet run".
- A Windows path-validation bug was found in the older snapshot: `safe_relpath` could treat
  `/etc/passwd` as safe on Windows. Apply the cross-platform guard in section 3a if absent.

## 3. Fixes to apply to the attached snapshot if absent

**3a. `tools/sandbox.py`: replace `safe_relpath`:**
```python
def safe_relpath(rel: str) -> str:
    """Reject absolute paths, drive letters and '..' under BOTH POSIX and Windows rules."""
    from pathlib import PurePosixPath, PureWindowsPath
    bad = (not rel or rel.startswith(("/", "\\")) or ":" in rel
           or PurePosixPath(rel).is_absolute() or PureWindowsPath(rel).is_absolute()
           or ".." in PurePosixPath(rel.replace("\\", "/")).parts)
    if bad:
        raise ValueError(f"unsafe path from model output: {rel!r}")
    return rel
```
In `tests/test_units.py::test_safe_relpath_blocks_traversal` use the bad list
`("../x.py", "/etc/passwd", "a/../../b.py", "..\\x.py", "\\etc\\x", "C:\\x.py", "C:/x.py", "")`.

**3b. `agents/debugger.py`:** append to `SYSTEM`: "If no test could run or a test itself raises a
non-assertion error (for example `NameError`, `ImportError`, or a fixture used outside a test
function), the test file may be the problem: output its corrected version. If a test directly reads
or writes data files instead of using the program API, correct its setup or API usage without
weakening valid expected behavior. For genuine assertion mismatches in tests that exercise the
program API, fix the implementation and preserve the tests. Make sure every import the code uses
exists."

In `run()`, before deciding which patched files to accept:
```python
no_test_ran = tr.passed == 0 and tr.failed == 0
behaviour_mismatch = bool(
    re.search(r"^E\s+(AssertionError|assert\b)|DID NOT RAISE", tr.traceback, re.M)
)
tests_read_files_directly = any(
    re.search(r"\bopen\s*\(|\.(?:open|read_text|read_bytes|write_text|write_bytes)\s*\(", test)
    for test in state.tests.values()
)
tests_editable = no_test_ran or not behaviour_mismatch or tests_read_files_directly
```
Use `if path in state.tests and tests_editable:` (else the `elif path in state.files` branch).
`state.debug_hint` may change the diagnostic prompt but must never override this edit guard. The
direct-file exception is only for repairing test setup/API use; preserve valid acceptance expectations.

**3c. `agents/tester.py` `SYSTEM`:** require fixtures only as function parameters. For
module-level mutable state, import the module (not `from student import students`) and reset its
actual store to a fresh value in an autouse fixture. Adapt the module, store name and empty value
to the generated code; changing the working directory does not reset Python globals. If the code
uses relative file paths, also change to `tmp_path` in the fixture:
```python
import pytest
import student

@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(student, "students", [])
```
Test through public program functions; do not assert on internal stores or inspect data files
directly. Import every module used, and do not test behavior the SRS does not state.

**3d. `agents/coder.py` `SYSTEM`:** add "Include EVERY import the code uses (json, os, ...)."

**3e. `agents/requirement.py` `SYSTEM`:** add "Do not invent features: unless the request mentions files, a
database or persistence, keep all data in memory."

**3f. Add regression tests to `tests/test_supervisor.py`** (use the file's existing helpers `scripted`, `run`,
`file_block`, `TESTS`, `GOOD`, `BAD`):
1. Tester returns a file that fails at import (module-level `monkeypatch.setattr`); debugger returns the corrected
   test; expect `status == "success"`, `iteration == 1`, no `change_strategy` event.
2. Tester returns a test with a `NameError` inside a test function (uses `json` without import); same expectations.
3. Tester returns an assertion failure caused by a test directly reading a data file from the wrong location;
   debugger corrects the test setup/API use; expect success without changing implementation behavior.
4. Code has a real bug (`BAD`); debugger first returns a weakened test (`assert True`), then returns both a
   weakened test and the fixed code. Even after `debug_hint` is set, expect success and the original `== 5`
   assertion to remain in `state.tests`.

## 4. What to verify (in this order; report each result honestly)

1. `pip install -r requirements.txt` and `python -m pytest -q`: all tests pass offline (about 41; the exact count
   depends on parametrization). If any test fails on a supported OS/Python version, fix the cause.
2. `ruff check main.py config.py llm.py state.py supervisor.py agents eval tests tools --select F,E9`
   clean; `bandit -r main.py config.py llm.py state.py supervisor.py agents eval tools -ll` shows nothing
   medium or above. Explicit paths avoid scanning `.venv/` and generated `runs/` artifacts.
3. `python main.py --demo` succeeds (scripted; proves plumbing only).
4. Real smoke run (user's machine, WSL with Ollama on Windows): get the current Windows host IP with
   `WIN_HOST=$(ip route show | awk '/default/ {print $3; exit}')`, then run
   `OLLAMA_URL="http://${WIN_HOST}:11434" AGENTIC_MODEL=qwen2.5-coder:7b python main.py "Create a Python student management system with add, delete, update and search functions."`.
   Read `runs/<ts>/state.json`: status, `history`, final
   `test_result.traceback`, `review`. If `partial`, diagnose from the log whether the cause is a pipeline
   defect (fix it) or plain model weakness (report it).
5. Dev prompts may be tuned on THIS demo request only. **Then freeze prompts.** Do not tune against HumanEval or
   the five custom tasks; record the model name, temperature (0.2), seed (42), token budget and git/zip hash with the results.

## 5. Evaluation to run once prompts are frozen (user's machine, Ollama running)

```
python -m eval.run_custom --mode single
python -m eval.run_custom --mode single_retry
python -m eval.run_custom --mode multi
# HumanEval: download HumanEval.jsonl.gz from github.com/openai/human-eval and gunzip it
python -m eval.humaneval_runner --dataset HumanEval.jsonl --mode single       --n 1 --limit 20   # time check first
python -m eval.humaneval_runner --dataset HumanEval.jsonl --mode single_retry --n 1 --limit 20
python -m eval.humaneval_runner --dataset HumanEval.jsonl --mode multi        --n 1 --limit 20
```
Then the full 164 problems (use `--n 5 --k 1 5` only if time allows; multi mode is slow, about 2 to 3 minutes
per problem on the user's laptop, so budget hours and consider a stated subset). Human-assisted baseline: the
student times themselves on the 5 custom tasks with a plain chat assistant. Same model, temperature and token
budget for all automated modes.

## 6. Expected outputs

1. Working project zip (fixes in section 3 applied, tests green) plus a short changelog.
2. Result JSON files in `runs/` for each mode (`custom_*.json`, `humaneval_*.json`) and one results table:
   mode x {task success rate, mean correctness, Pass@1 (and Pass@k if run), mean tokens, mean seconds}.
3. Updated Phase-I report (Claude Docs artifact "Phase-I Report: Agentic AI-Based Autonomous Software Engineering
   Assistant", link: https://claude.ai/artifact/E9mQpAANtZxWk4VTB4h4Hb): put the real table into section 5, delete
   the "Not yet run" paragraph, state that a 7B model was used and what the failures were. Report negative or
   mixed findings plainly: the literature (Agentless) predicts multi-agent can cost more without winning.
4. A one-page failure analysis from `state.json` logs: how many runs ended success / partial / failed, top causes
   (bad tests, missing imports, repeated failure), tokens and time per agent.
5. Review-1 slide update: real numbers on the evaluation slide, plus honest limits (7B model, no Docker network
   isolation unless `AGENTIC_DOCKER=1`, interface-specified custom tasks, no SWE-bench yet).

## 7. Rules

- No fabricated or estimated numbers. Mock-backend output is never a result.
- Do not edit hidden tests or benchmark data to make results look better.
- Keep the Windows path guard and sandbox protections; do not weaken them.
- Ask the user for: guide name, project ID, GPU/RAM if model choice matters.
