"""Shared helpers for agents: file-block parsing and the Agent base class."""
from __future__ import annotations

import re

from llm import LLMClient, extract_code
from state import SharedState

# Models are asked to answer with blocks of the form:
#   ### FILE: calc.py
#   ```python
#   ...code...
#   ```
_FILE_BLOCK = re.compile(
    r"###\s*FILE:\s*(?P<path>[^\n`]+?)\s*\n\s*```[a-zA-Z]*\s*\n(?P<code>.*?)```", re.S)


def parse_file_blocks(text: str) -> dict[str, str]:
    out = {}
    for m in _FILE_BLOCK.finditer(text):
        out[m.group("path").strip()] = m.group("code").rstrip("\n") + "\n"
    return out


def parse_files_or_single(text: str, default_path: str) -> dict[str, str]:
    """Prefer explicit FILE blocks; fall back to the first code block as `default_path`."""
    files = parse_file_blocks(text)
    if files:
        return files
    return {default_path: extract_code(text)}


class Agent:
    name = "agent"

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def run(self, state: SharedState) -> None:  # pragma: no cover - interface
        raise NotImplementedError
