"""Evaluation harness plumbing, checked with oracle mocks (NOT model results)."""
import dataclasses
import json
import pathlib
import sys

import pytest

from config import Config
from eval import mock_llm
from eval import run_custom
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


def test_failed_reviewer_does_not_earn_benchmark_credit(tmp_path):
    problem = load_problems(MINI)[0]

    def broken_review(backend):
        backend.responses["reviewer"] = ["invalid JSON"]
        return backend

    report = tmp_path / "humaneval.json"
    summary = evaluate([problem], "multi", CFG,
                       backend_factory=lambda p: broken_review(mock_llm.humaneval_oracle_backend(p)),
                       out_path=str(report))
    row = json.loads(report.read_text())["rows"][0]
    assert summary["pass@1"] == 0.0
    assert summary["error_samples"] == 1
    assert summary["backend"] == "mock"
    assert "reviewer: invalid JSON" in row["errors"][0]["reason"]
    assert row["tokens"] > 0

    task = next(t for t in load_tasks() if t["id"] == "student_mgmt")
    result = run_task(task, "multi", CFG, broken_review(mock_llm.custom_oracle_backend(task["id"])))
    assert not result["success"] and result["tests_total"] == 0
    assert "reviewer: invalid JSON" in result["error"]


def test_custom_report_keeps_going_after_a_model_error(tmp_path, monkeypatch):
    tasks = load_tasks()[:2]
    original_backend = mock_llm.custom_oracle_backend

    def fail_generate(*args):
        raise ConnectionError("model unavailable")

    monkeypatch.setattr(run_custom, "load_tasks", lambda: tasks)
    monkeypatch.setattr(mock_llm, "custom_oracle_backend", lambda task_id:
                        mock_llm.MockBackend(fail_generate) if task_id == tasks[0]["id"]
                        else original_backend(task_id))
    report = tmp_path / "custom.json"
    monkeypatch.setattr(sys, "argv", ["run_custom", "--mode", "single", "--backend", "mock",
                                     "--out", str(report)])
    run_custom.main()
    data = json.loads(report.read_text())
    assert data["summary"]["backend"] == "mock"
    assert data["summary"]["error_tasks"] == 1
    assert data["summary"]["task_success_rate"] == 0.5
    assert data["rows"][0]["error"] == "ConnectionError: model unavailable"
    assert data["rows"][1]["success"]
