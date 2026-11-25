"""LLM client with token and latency accounting.

Backends:
  * OllamaBackend - real local model via the Ollama HTTP API (reports token counts).
  * MockBackend   - deterministic scripted backend so the pipeline can be tested
                    offline (no GPU, no network). See tests/ and eval/mock_llm.py.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional

from config import Config


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    seconds: float


class BudgetExceeded(Exception):
    pass


class OllamaBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> tuple[str, int, int]:
        import requests  # imported lazily so mock runs need no network libs

        payload = {
            "model": self.cfg.model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.cfg.temperature, "seed": self.cfg.seed},
        }
        if self.cfg.max_output_tokens:
            payload["options"]["num_predict"] = self.cfg.max_output_tokens
        if json_mode:
            payload["format"] = "json"
        r = requests.post(f"{self.cfg.ollama_url}/api/generate", json=payload,
                          timeout=self.cfg.llm_timeout_s)
        r.raise_for_status()
        d = r.json()
        return d.get("response", ""), d.get("prompt_eval_count", 0), d.get("eval_count", 0)


class MockBackend:
    """Calls `handler(agent_hint, system, prompt) -> str`. Token counts are
    approximated as words so budget logic is exercised."""

    def __init__(self, handler: Callable[[str, str, str], str]):
        self.handler = handler

    def generate(self, system: str, prompt: str, json_mode: bool = False) -> tuple[str, int, int]:
        out = self.handler(system, prompt, json_mode)
        return out, len((system + prompt).split()), len(out.split())


def extract_json(text: str):
    """Parse JSON from model output, tolerating code fences and surrounding prose."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", text, 0)


def extract_code(text: str, lang: str = "python") -> str:
    """Return the first fenced code block, or the raw text if there is none."""
    m = re.search(rf"```(?:{lang}|py)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip("\n") + "\n"


class LLMClient:
    """Wraps a backend; every call is metered and appended to the run metrics."""

    def __init__(self, cfg: Config, backend=None):
        self.cfg = cfg
        self.backend = backend or (OllamaBackend(cfg) if cfg.backend == "ollama" else None)
        if self.backend is None:
            raise ValueError("mock backend must be passed explicitly")
        self.calls: list[dict] = []
        self.total_in = 0
        self.total_out = 0

    @property
    def total_tokens(self) -> int:
        return self.total_in + self.total_out

    def call(self, agent: str, system: str, prompt: str, json_mode: bool = False) -> LLMResponse:
        if self.total_tokens >= self.cfg.token_budget:
            raise BudgetExceeded(f"token budget {self.cfg.token_budget} exhausted")
        t0 = time.time()
        text, tin, tout = self.backend.generate(system, prompt, json_mode)
        dt = time.time() - t0
        self.total_in += tin
        self.total_out += tout
        self.calls.append({"agent": agent, "tokens_in": tin, "tokens_out": tout,
                           "seconds": round(dt, 3)})
        return LLMResponse(text, tin, tout, dt)

    def call_json(self, agent: str, system: str, prompt: str,
                  required_keys: Optional[list[str]] = None,
                  validate: Optional[Callable[[dict], None]] = None) -> dict:
        """Call and validate JSON, with one automatic repair retry (design doc, section 6)."""
        last_err = None
        for attempt in range(self.cfg.json_repair_retries + 1):
            p = prompt if attempt == 0 else (
                prompt + f"\n\nYour previous reply was invalid ({last_err}). "
                         "Reply again with ONLY a valid JSON object.")
            resp = self.call(agent, system, p, json_mode=True)
            try:
                data = extract_json(resp.text)
                if not isinstance(data, dict):
                    raise TypeError("expected a JSON object")
                missing = [k for k in (required_keys or []) if k not in data]
                if missing:
                    raise KeyError(f"missing keys {missing}")
                if validate is not None:
                    validate(data)
                return data
            except (ValueError, KeyError, TypeError) as e:
                last_err = str(e)
        raise ValueError(f"{agent}: invalid JSON after repair retry: {last_err}")
