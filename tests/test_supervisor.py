"""Workflow, termination rules and safety behaviour of the supervisor (design doc, sections 5-6)."""
import dataclasses


from config import Config
from eval.mock_llm import (ScriptedBackend, demo_backend, file_block, plan_json,
                           review_json, srs_json)
from llm import LLMClient
from state import SharedState, TestResult
from supervisor import STOP, Supervisor, next_action

GOOD = "def add(a: int, b: int) -> int:\n    return a + b\n"
BAD = "def add(a, b):\n    return a - b\n"
TESTS = "from m import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"


def cfg(**kw) -> Config:
    return dataclasses.replace(Config(backend="mock"), sandbox_timeout_s=10, **kw)


def scripted(**over) -> ScriptedBackend:
    base = {
        "requirement": [srs_json("t", ["add"], ["add(2,3)==5"])],
        "planner": [plan_json("m.py", "adder", ["def add(a, b)"])],
        "coder": [file_block("m.py", GOOD)],
        "tester": [file_block("test_solution.py", TESTS)],
        "debugger": [file_block("m.py", GOOD)],
        "reviewer": [review_json(True)],
    }
    base.update(over)
    return ScriptedBackend(base)


def run(backend, **cfg_kw):
    c = cfg(**cfg_kw)
    llm = LLMClient(c, backend)
    return Supervisor(c, llm).run("add two numbers"), llm


def agents(state):
    return [h["agent"] for h in state.history if h["action"] in
            ("produced_srs", "produced_plan", "wrote_code", "revised_code", "ran_tests",
             "patched", "reviewed")]


# ---------------------------------------------------------------- routing (pure function)
def test_routing_order_of_first_steps():
    c, s = cfg(), SharedState(request="x")
    assert next_action(s, c) == "requirement"
    s.srs = {}
    assert next_action(s, c) == "planner"
    s.plan = {}
    assert next_action(s, c) == "coder"
    s.files = {"m.py": ""}
    assert next_action(s, c) == "tester"


def test_routing_failure_goes_to_debugger_then_limit_stops():
    c = cfg(max_debug_iterations=2)
    s = SharedState(request="x", srs={}, plan={}, files={"m.py": ""}, tests={"t": ""},
                    test_result=TestResult(failed=1), last_agent="tester")
    assert next_action(s, c) == "debugger"
    s.iteration = 2
    assert next_action(s, c) == STOP and s.status == "partial"


def test_routing_pass_review_and_stop():
    c = cfg()
    s = SharedState(request="x", srs={}, plan={}, files={"m.py": ""}, tests={"t": ""},
                    test_result=TestResult(passed=1, all_passed=True), last_agent="tester")
    assert next_action(s, c) == "reviewer"
    s.needs_review, s.review = False, {"approved": True}
    assert next_action(s, c) == STOP and s.status == "success"


# ---------------------------------------------------------------- full runs
def test_happy_path_no_debugging():
    s, _ = run(scripted())
    assert s.status == "success"
    assert agents(s) == ["requirement", "planner", "coder", "tester", "reviewer"]
    assert "debugger" not in agents(s)


def test_debug_loop_fixes_bug():
    s, _ = run(scripted(coder=[file_block("m.py", BAD)]))
    assert s.status == "success"
    assert agents(s) == ["requirement", "planner", "coder", "tester", "debugger", "tester", "reviewer"]
    assert s.iteration == 1 and s.files["m.py"] == GOOD


def test_demo_scenario_succeeds():
    c = cfg()
    s = Supervisor(c, LLMClient(c, demo_backend())).run("student mgmt")
    assert s.status == "success" and s.test_result.passed == 5


def test_debug_iteration_limit_gives_partial_with_artifacts():
    # debugger keeps returning a different-but-still-wrong fix, so signatures differ
    wrong = [file_block("m.py", f"def add(a, b):\n    return a - b + {i}\n") for i in range(1, 8)]
    s, _ = run(scripted(coder=[file_block("m.py", BAD)], debugger=wrong), max_debug_iterations=3)
    assert s.status == "partial" and "debug limit" in s.stop_reason
    assert agents(s).count("debugger") == 3 and "reviewer" not in agents(s)
    assert s.files  # partial results are still reported


def test_same_failure_repeated_changes_strategy_then_escalates():
    s, _ = run(scripted(coder=[file_block("m.py", BAD)], debugger=[file_block("m.py", BAD)]),
               max_debug_iterations=10)
    assert s.status == "partial" and "escalate" in s.stop_reason
    assert any(h["action"] == "change_strategy" for h in s.history)
    assert agents(s).count("debugger") == 2   # not 10: escalation beat the iteration cap


def test_strategy_change_lets_debugger_fix_a_wrong_test():
    wrong_test = (
        "from m import add\n"
        "def test_add(tmp_path):\n"
        "    result = tmp_path / 'result.txt'\n"
        "    result.write_text('[]')\n"
        "    assert result.read_text() == str(add(2, 3))\n"
    )
    s, _ = run(scripted(tester=[file_block("test_solution.py", wrong_test)],
                        debugger=[file_block("m.py", GOOD),
                                  file_block("test_solution.py", TESTS)]),
               max_debug_iterations=10)
    assert s.status == "success"
    assert any(h["action"] == "change_strategy" for h in s.history)
    assert "== 5" in s.tests["test_solution.py"]


def test_reviewer_cannot_approve_insecure_code_and_coder_revises():
    insecure = "def add(a, b):\n    return eval(f'{a}+{b}')\n"
    s, _ = run(scripted(coder=[file_block("m.py", insecure), file_block("m.py", GOOD)],
                        reviewer=[review_json(True)]))   # LLM says approve; bandit overrides
    assert s.status == "success"
    assert agents(s).count("reviewer") == 2 and s.files["m.py"] == GOOD
    assert "revised_code" in agents(s) or s.review_cycles == 1


def test_review_cycle_limit():
    insecure = file_block("m.py", "def add(a, b):\n    return eval(f'{a}+{b}')\n")
    s, _ = run(scripted(coder=[insecure]), max_review_cycles=2)
    assert s.status == "partial" and "review rejected" in s.stop_reason
    assert agents(s).count("reviewer") == 3     # initial + 2 fix cycles


def test_reviewer_llm_reject_with_findings_triggers_revision():
    s, _ = run(scripted(reviewer=[review_json(False, [{"severity": "low", "file": "m.py",
                                                       "issue": "docstring", "fix": "add one"}]),
                                  review_json(True)]))
    assert s.status == "success" and s.review_cycles == 1


def test_token_budget_stops_run_with_partial_status():
    s, llm = run(scripted(), token_budget=200)
    assert s.status == "partial" and "budget" in s.stop_reason


def test_malformed_json_gets_one_repair_retry():
    s, llm = run(scripted(requirement=["not json at all", srs_json("t", ["a"], ["b"])]))
    assert s.status == "success"
    assert sum(1 for m in llm.calls if m["agent"] == "requirement") == 2


def test_malformed_json_twice_fails_cleanly():
    s, _ = run(scripted(requirement=["garbage"]))
    assert s.status == "failed" and "invalid JSON" in s.stop_reason


def test_unsafe_path_from_planner_is_rejected():
    s, _ = run(scripted(planner=[plan_json("../evil.py", "x", [])]))
    assert s.status == "failed" and "unsafe path" in s.stop_reason


def test_metrics_and_replayable_log(tmp_path):
    c = cfg()
    llm = LLMClient(c, scripted())
    s = Supervisor(c, llm).run("add", str(tmp_path))
    assert (tmp_path / "state.json").exists() and (tmp_path / "output" / "m.py").exists()
    assert all({"agent", "tokens_in", "tokens_out", "seconds"} <= set(m) for m in s.metrics)
    assert s.history[0]["action"] == "start" and s.history[-1]["action"] == "stop"


def test_debugger_may_fix_broken_test_file_when_no_test_ran():
    """Regression from the first real-model run: the tester produced a file that failed at
    import (fixture used at module level). No test ran, so the debugger must be allowed to
    repair the test file on its FIRST attempt instead of wasting an iteration."""
    broken = "from m import add\nmonkeypatch.setattr('m.x', 1)\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    s, _ = run(scripted(tester=[file_block("test_solution.py", broken)],
                        debugger=[file_block("test_solution.py", TESTS)]))
    assert s.status == "success" and s.iteration == 1
    assert not any(h["action"] == "change_strategy" for h in s.history)


def test_debugger_may_fix_nameerror_inside_test():
    broken = ("from m import add\n\n"
              "def test_add():\n"
              "    assert json.loads('{\"value\": 5}')['value'] == add(2, 3)\n")
    s, _ = run(scripted(tester=[file_block("test_solution.py", broken)],
                        debugger=[file_block("test_solution.py", TESTS)]))
    assert s.status == "success" and s.iteration == 1
    assert not any(h["action"] == "change_strategy" for h in s.history)


def test_debugger_may_fix_test_that_reads_wrong_file():
    wrong_test = (
        "from m import add\n"
        "def test_add(tmp_path):\n"
        "    result = tmp_path / 'result.txt'\n"
        "    result.write_text('[]')\n"
        "    assert result.read_text() == str(add(2, 3))\n"
    )
    s, _ = run(scripted(tester=[file_block("test_solution.py", wrong_test)],
                        debugger=[file_block("test_solution.py", TESTS)]))
    assert s.status == "success" and s.iteration == 1
    assert not any(h["action"] == "change_strategy" for h in s.history)


def test_debugger_cannot_edit_tests_when_tests_ran_and_failed():
    """A strategy change must not let the debugger weaken an assertion to hide a code bug."""
    weak = file_block("test_solution.py", "from m import add\n\ndef test_add():\n    assert True\n")
    s, _ = run(scripted(coder=[file_block("m.py", BAD)],
                        debugger=[weak, weak + file_block("m.py", GOOD)]))
    assert s.status == "success"
    assert "== 5" in s.tests["test_solution.py"]      # the original test was kept
