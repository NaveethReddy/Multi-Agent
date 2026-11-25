from config import Config
from eval.humaneval_runner import evaluate
from eval.mock_llm import ScriptedBackend, plan_json, srs_json
from llm import LLMClient
from supervisor import Supervisor


def make_supervisor(plan):
    config = Config(backend="mock")
    responses = {
        "requirement": [srs_json("Add", ["Add two numbers"], ["add(2, 3) == 5"])],
        "planner": [plan],
    }
    return Supervisor(config, LLMClient(config, ScriptedBackend(responses)))


def test_plan_only_stops_before_coding_and_saves_state(tmp_path):
    supervisor = make_supervisor(plan_json("adder.py", "Add numbers", ["def add(a, b)"]))
    state = supervisor.run("Add two numbers", str(tmp_path), plan_only=True)
    assert state.status == "planned"
    assert state.plan["modules"][0]["file"] == "adder.py"
    assert [entry["agent"] for entry in state.metrics] == ["requirement", "planner"]
    assert not state.files
    assert (tmp_path / "state.json").is_file()


def test_invalid_plan_records_the_failed_stage():
    state = make_supervisor('{"modules": [], "tasks": []}').run("Add", plan_only=True)
    assert state.status == "failed"
    assert state.history[-2]["agent"] == "planner"
    assert state.history[-2]["action"] == "stage_failed"
    assert "at least one module" in state.stop_reason


def test_canonical_mode_scores_the_reference_without_a_backend():
    problems = [{
        "task_id": "example",
        "prompt": "def add(a, b):\n",
        "canonical_solution": "    return a + b\n",
        "test": "def check(fn):\n    assert fn(2, 3) == 5",
        "entry_point": "add",
    }]
    report = evaluate(problems, "canonical", Config(backend="mock"))
    assert report["model"] is None
    assert report["pass@1"] == 1.0
    assert report["avg_tokens_per_problem"] == 0
