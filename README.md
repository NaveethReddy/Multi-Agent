# Agentic AI-Based Autonomous Software Engineering Assistant (Phase-I)

Multi-agent LLM pipeline that turns a plain-English requirement into tested, reviewed Python code.
M.Tech. CSE, Project Work Phase-I (UE25CS7A1A). Implements the design in *System Architecture and
Agent Workflow Design*.

```
request -> Requirement -> Planner -> Coder -> Tester --pass--> Reviewer -> final code
                                        ^        | fail            | reject (max 2 cycles)
                                        |        v                 |
                                        +----- Debugger (max 5) <--+
                    Supervisor (state machine) owns the shared state and every decision
```

## What is implemented (Phase-I scope)

| Component | File | Notes |
|---|---|---|
| LLM client + token/latency accounting | `llm.py` | Ollama backend, JSON validation with one repair retry, token budget |
| Shared state | `state.py` | request, srs, plan, files, tests, test_result, iteration, review, history, metrics |
| Supervisor / workflow | `supervisor.py` | Pure `next_action()` routing; debug limit, review-cycle limit, token budget, wall-clock timeout, repeated-failure strategy change and escalation |
| Agents | `agents/` | Requirement, Planner, Coder, Tester, Debugger, Reviewer |
| Sandbox | `tools/sandbox.py` | Temp dir, subprocess timeout, CPU/memory limits, scrubbed env, path-traversal guard, optional Docker (`--network none`) |
| Static analysis | `tools/static_analysis.py` | ruff (quality) + bandit (security); reviewer cannot approve failing tests or medium+ security findings |
| Evaluation | `eval/` | HumanEval Pass@k (unbiased estimator), 5 custom end-to-end tasks with hidden tests, single-agent baselines |

Phase-II (not in this repo yet): RAG memory, documentation agent, explainability, dashboard.

## Setup

From the project directory, create and activate a virtual environment:

```bash
# Debian/Ubuntu only, if venv support is missing (use the version matching python3)
sudo apt install python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, create it with `py -m venv .venv`, then activate it in PowerShell with
`.venv\Scripts\Activate.ps1` or in Command Prompt with `.venv\Scripts\activate.bat`. Once activated,
install the project dependencies:

```bash
pip install -r requirements.txt
# real runs need Ollama: https://ollama.com
ollama pull qwen2.5-coder:7b          # or any model that fits your GPU/RAM
```

Run `deactivate` to leave the virtual environment.

## Run

```bash
python main.py "Create a Python student management system with add, delete, update and search functions."
python main.py --demo                  # offline scripted demo (no Ollama)
AGENTIC_MODEL=deepseek-coder-v2:16b python main.py "..."     # pick the model
```

Each run writes `runs/<timestamp>/state.json` (full replayable log: history, per-call tokens and
seconds, review report) and `runs/<timestamp>/output/` (generated code and tests).

For the Phase I review, demonstrate requirements and planning with the real local model:

```powershell
.\.venv\Scripts\python.exe main.py --model qwen2.5-coder:7b --plan-only --max-output-tokens 640 --llm-timeout 420 "Create a Python student management system with add, delete, update and search functions"
```

`planned` means the run saved both an SRS and an implementation plan. Open the
reported `state.json` to show those artifacts and the stage trace. In a failed
run, `stage_failed` identifies the stage that stopped and records the error.
The optional output cap can truncate a long answer. The main CLI defaults and
benchmark sampling settings apply when no override is supplied.

Use `python main.py --demo` to demonstrate code generation, failing tests,
debugging, retesting and review offline. The demo uses scripted responses and
reports its backend in the CLI and run log.

See [Phase I review milestone](PHASE_I_REVIEW.md) for the review checklist.

## Evaluate (design doc, section 10)

Check the benchmark harness with the official reference solutions before
running model comparisons. The result measures the harness without Ollama.

```bash
python -m eval.humaneval_runner --data HumanEval.jsonl.gz --mode canonical
```

```bash
# 1. HumanEval: download data/HumanEval.jsonl.gz from github.com/openai/human-eval and gunzip it
for m in single single_retry multi; do
  python -m eval.humaneval_runner --dataset HumanEval.jsonl --mode $m --n 5 --k 1 5 --out runs/he_$m.json
done
# 2. Custom end-to-end tasks (hidden tests decide success)
for m in single single_retry multi; do python -m eval.run_custom --mode $m; done
```

The same model, temperature and budget are used in every mode. Samples within one problem use seeds
`seed, seed+1, ...` so Pass@k with `n > 1` is meaningful. The third baseline (human-assisted) is timed
manually on the custom tasks.

## Tests

```bash
python -m pytest -q        # 37 tests, ~30 s, fully offline
```

`eval/mock_llm.py` is a **scripted** backend used only to verify plumbing (routing, sandbox, parsers,
harness). Its outputs are not model results and must never be reported as such.

## Safety notes

Generated code runs only in the sandbox. Without Docker the sandbox does **not** block network access
(it only scrubs proxy and credential env vars); set `AGENTIC_DOCKER=1` for `--network none`.
