"""Central configuration. Override with environment variables."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # LLM backend: "ollama" (real) or "mock" (offline, deterministic, for tests/demo)
    backend: str = os.getenv("AGENTIC_BACKEND", "ollama")
    model: str = os.getenv("AGENTIC_MODEL", "qwen2.5-coder:7b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("AGENTIC_TEMP", "0.2"))
    seed: int = int(os.getenv("AGENTIC_SEED", "42"))
    llm_timeout_s: int = 300

    # Termination rules (architecture doc, section 5)
    max_debug_iterations: int = 5
    max_review_cycles: int = 2
    token_budget: int = 60_000        # total in+out tokens per run
    wall_clock_timeout_s: int = 900   # per run
    json_repair_retries: int = 1

    # Sandbox
    sandbox_timeout_s: int = 20
    use_docker: bool = os.getenv("AGENTIC_DOCKER", "0") == "1"

    runs_dir: str = os.getenv("AGENTIC_RUNS", "runs")


def default_config() -> Config:
    return Config()
