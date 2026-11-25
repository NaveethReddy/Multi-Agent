from __future__ import annotations

import json

from agents.base import Agent
from state import SharedState
from tools.sandbox import safe_relpath

SYSTEM = (
    "You are a software architect. Given an SRS, design a SMALL Python solution (prefer one "
    "or two modules). Reply with ONLY a JSON object with keys: "
    '"modules": list of {"file": "name.py", "purpose": str, "functions": list of '
    '{"signature": "def f(a: int) -> int", "description": str}}, and '
    '"tasks": list of {"id": int, "file": "name.py", "description": str}. '
    "Use only the Python standard library. File names must be simple relative paths. "
    "Keep descriptions brief and list only the public functions the request needs."
)


class PlannerAgent(Agent):
    name = "planner"

    def run(self, state: SharedState) -> None:
        data = self.llm.call_json(
            self.name, SYSTEM, "SRS:\n" + json.dumps(state.srs, indent=2),
            required_keys=["modules", "tasks"])
        if not isinstance(data["modules"], list) or not data["modules"]:
            raise ValueError("planner: expected at least one module")
        if not isinstance(data["tasks"], list) or not data["tasks"]:
            raise ValueError("planner: expected at least one implementation task")
        paths = set()
        for m in data["modules"]:
            if not isinstance(m, dict) or not isinstance(m.get("file"), str):
                raise ValueError("planner: each module needs a file path")
            path = safe_relpath(m["file"])
            if not path.endswith(".py") or path in paths:
                raise ValueError(f"planner: expected unique Python module paths, got {path!r}")
            paths.add(path)
        for task in data["tasks"]:
            if (not isinstance(task, dict) or not isinstance(task.get("file"), str)
                    or task["file"] not in paths or not isinstance(task.get("description"), str)
                    or not task["description"].strip()):
                raise ValueError("planner: each task needs a description and a planned module")
        state.plan = data
        state.log(self.name, "produced_plan",
                  {"files": [m["file"] for m in data["modules"]], "n_tasks": len(data["tasks"])})
