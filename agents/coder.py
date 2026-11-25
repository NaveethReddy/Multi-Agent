from __future__ import annotations

import json

from agents.base import Agent, parse_files_or_single
from state import SharedState

SYSTEM = (
    "You are an expert Python developer. Write clean, correct, standard-library-only Python. "
    "Add type hints and short docstrings. Never use eval/exec, shell commands or network "
    "access. Include EVERY import the code uses (json, os, ...). Initialize mutable instance "
    "state in __init__; do not use mutable class attributes for per-instance data. "
    "Reply with ONLY the file in this exact format:\n"
    "### FILE: <path>\n```python\n<code>\n```"
)


class CoderAgent(Agent):
    name = "coder"

    def run(self, state: SharedState) -> None:
        """First pass: write each planned module. After a review rejection, revise
        the files using the reviewer's findings (design doc, workflow step 7)."""
        revising = bool(state.review and not state.review.get("approved", True) and state.files)
        for module in state.plan["modules"]:
            path = module["file"]
            prompt = [
                "SRS:\n" + json.dumps(state.srs, indent=2),
                f"Module to write: {path}\nPurpose: {module.get('purpose', '')}",
                "Required functions:\n" + "\n".join(
                    f"- {f['signature']}: {f.get('description', '')}"
                    for f in module.get("functions", [])),
            ]
            if revising:
                prompt.append(f"Current code of {path}:\n```python\n{state.files.get(path, '')}\n```")
                prompt.append("Fix these review findings while keeping all behaviour:\n"
                              + json.dumps(state.review.get("findings", []), indent=2))
            resp = self.llm.call(self.name, SYSTEM, "\n\n".join(prompt))
            files = parse_files_or_single(resp.text, path)
            # keep only the module we asked for if the model returned extras
            code = files.get(path) or next(iter(files.values()))
            state.files[path] = code
        state.log(self.name, "revised_code" if revising else "wrote_code",
                  {"files": list(state.files)})
