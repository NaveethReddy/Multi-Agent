from __future__ import annotations

import json

from agents.base import Agent
from state import SharedState
from tools.static_analysis import analyze

SYSTEM = (
    "You are a senior code reviewer. You get final code, test results and static-analysis "
    "findings (ruff = style/quality, bandit = security). Reply with ONLY a JSON object: "
    '{"approved": bool, "findings": [{"severity": "high|medium|low", "file": str, '
    '"issue": str, "fix": str}], "summary": str}. Only list real problems; do not '
    "approve code with security findings of medium or higher severity."
)


class ReviewerAgent(Agent):
    name = "reviewer"

    def run(self, state: SharedState) -> None:
        static = analyze(state.files)
        tr = state.test_result
        prompt = (
            "Code:\n" + "\n\n".join(f"# {p}\n{c}" for p, c in state.files.items())
            + f"\n\nTests: {tr.passed} passed, {tr.failed} failed, {tr.errors} errors."
            + "\n\nruff findings:\n" + json.dumps(static["ruff"]["issues"][:20])
            + "\n\nbandit findings:\n" + json.dumps(static["bandit"]["issues"][:20]))
        data = self.llm.call_json(self.name, SYSTEM, prompt,
                                  required_keys=["approved", "findings"])
        # Hard rules from the design doc: tools, not the LLM's opinion, have the last word.
        approved = bool(data["approved"])
        reasons = []
        if not tr.all_passed:
            approved = False
            reasons.append("tests are not all passing")
        if static["blocking_security"] > 0:
            approved = False
            reasons.append(f"{static['blocking_security']} bandit finding(s) of medium+ severity")
        findings = list(data.get("findings", []))
        for i in static["bandit"]["issues"]:
            if i["severity"] in ("HIGH", "MEDIUM"):
                findings.append({"severity": i["severity"].lower(), "file": i["file"],
                                 "issue": f"[bandit {i['test_id']}] {i['message']}",
                                 "fix": "remove or replace the insecure construct"})
        state.review = {"approved": approved, "findings": findings,
                        "summary": data.get("summary", ""), "forced_reasons": reasons,
                        "standards_score": static["standards_score"],
                        "ruff_issues": len(static["ruff"]["issues"]),
                        "bandit_issues": len(static["bandit"]["issues"])}
        state.review_cycles += 0 if approved else 1
        state.log(self.name, "reviewed", {"approved": approved, "n_findings": len(findings),
                                          "standards_score": static["standards_score"]})
