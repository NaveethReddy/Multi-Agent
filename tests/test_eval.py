"""Evaluation harness plumbing, checked with oracle mocks (NOT model results)."""
import dataclasses
import pathlib

import pytest

from config import Config
from eval import mock_llm
from eval.humaneval_runner import check_solution, evaluate, load_problems
from eval.run_custom import load_tasks, run_task

MINI = str(pathlib.Path(__file__).parent.parent / "eval" / "data" / "mini_humaneval.jsonl")
CFG = dataclasses.replace(Config(backend="mock"), sandbox_timeout_s=10)


def test_mini_dataset_canonical_solutions_pass_and_wrong_ones_fail():
    for p in load_problems(MINI):
        assert check_solution(p, p["prompt"] + p["canonical_solution"], 10)
        assert not check_solution(p, p["prompt"] + "    return None\n", 10)


@pytest.mark.parametrize("mode", ["single", "single_retry", "multi"])
def test_humaneval_harness_all_modes(mode):
    s = evaluate(load_problems(MINI), mode, CFG, n=2, ks=(1, 2),
                 backend_factory=mock_llm.humaneval_oracle_backend)
    assert s["pass@1"] == 1.0 and s["pass@2"] == 1.0 and s["problems"] == 5


def test_humaneval_harness_scores_a_wrong_solver_as_zero():
    def bad(problem):
        return mock_llm.oracle_backend("solution.py", "def broken():\n    pass\n",
                                       "def test_x():\n    assert True\n")
    s = evaluate(load_problems(MINI)[:2], "single", CFG, n=1, backend_factory=bad)
    assert s["pass@1"] == 0.0


def test_custom_tasks_reference_solutions_pass_hidden_tests_in_all_modes():
    tasks = load_tasks()
    assert len(tasks) == 5
    for t in tasks:
        for mode in ("single", "single_retry", "multi"):
            r = run_task(t, mode, CFG, mock_llm.custom_oracle_backend(t["id"]))
            assert r["success"] and r["correctness"] == 1.0, (t["id"], mode, r)


def test_custom_hidden_tests_reject_a_naive_solution():
    t = next(t for t in load_tasks() if t["id"] == "student_mgmt")
    r = run_task(t, "single", CFG, mock_llm.oracle_backend(
        t["module"], mock_llm.BUGGY_STUDENTS, "def test_x():\n    assert True\n"))
    assert not r["success"] and r["correctness"] < 1.0
