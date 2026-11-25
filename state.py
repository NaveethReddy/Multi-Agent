"""Shared state object owned by the supervisor (design doc, section 4)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class TestResult:
    __test__ = False  # not a pytest test class
    passed: int = 0
    failed: int = 0
    errors: int = 0
    traceback: str = ""
    all_passed: bool = False
    raw: str = ""


@dataclass
class SharedState:
    request: str
    srs: Optional[dict] = None
    plan: Optional[dict] = None
    files: dict[str, str] = field(default_factory=dict)
    tests: dict[str, str] = field(default_factory=dict)
    test_result: Optional[TestResult] = None
    iteration: int = 0            # debug-loop count
    review_cycles: int = 0
    review: Optional[dict] = None
    history: list[dict] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)
    status: str = "running"       # running | success | partial | failed
    stop_reason: str = ""
    failure_signatures: list[str] = field(default_factory=list)
    debug_hint: str = ""          # supervisor can ask the debugger to change strategy
    needs_review: bool = True     # set whenever code changes, cleared by the reviewer
    last_agent: str = ""

    def log(self, agent: str, action: str, detail: Any = None):
        self.history.append({"t": round(time.time(), 3), "agent": agent,
                             "action": action, "detail": detail})

    def to_dict(self) -> dict:
        return asdict(self)

    def dump(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
