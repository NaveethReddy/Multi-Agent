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
    "Use only the Python standard library. File names must be simple relative paths."
)


class PlannerAgent(Agent):
    name = "planner"

    def run(self, state: SharedState) -> None:
        data = self.llm.call_json(
            self.name, SYSTEM, "SRS:\n" + json.dumps(state.srs, indent=2),
            required_keys=["modules", "tasks"])
        for m in data["modules"]:
            safe_relpath(m["file"])
        state.plan = data
        state.log(self.name, "produced_plan",
                  {"files": [m["file"] for m in data["modules"]], "n_tasks": len(data["tasks"])})
