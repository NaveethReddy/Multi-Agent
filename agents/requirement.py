from __future__ import annotations


from agents.base import Agent
from state import SharedState

SYSTEM = (
    "You are a requirements analyst. Turn the user's request into a precise software "
    "requirements specification (SRS). Reply with ONLY a JSON object with keys: "
    '"title" (string), "functional_requirements" (list of strings), "constraints" (list of '
    'strings), "acceptance_criteria" (list of strings, each checkable by a unit test). '
    "Be concise. Do not invent features that were not requested. Unless the request mentions "
    "files, a database or persistence, keep all data in memory."
)


class RequirementAgent(Agent):
    name = "requirement"

    def run(self, state: SharedState) -> None:
        data = self.llm.call_json(
            self.name, SYSTEM, f"User request:\n{state.request}",
            required_keys=["title", "functional_requirements", "acceptance_criteria"])
        data.setdefault("constraints", [])
        state.srs = data
        state.log(self.name, "produced_srs",
                  {"n_requirements": len(data["functional_requirements"]),
                   "n_acceptance": len(data["acceptance_criteria"])})
