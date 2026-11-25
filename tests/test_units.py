import json
import subprocess
import pytest

from agents.base import parse_file_blocks, parse_files_or_single
from eval.humaneval_runner import pass_at_k
from llm import extract_code, extract_json
from supervisor import failure_signature
from tools.sandbox import parse_pytest, run_pytest, safe_relpath, run_script
from tools.static_analysis import analyze


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('blah {"a": 3} trailing') == {"a": 3}
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here")


def test_extract_code_and_file_blocks():
    assert extract_code("x\n```python\nprint(1)\n```\ny") == "print(1)\n"
    txt = "### FILE: a.py\n```python\nA=1\n```\n### FILE: pkg/b.py\n```python\nB=2\n```"
    assert parse_file_blocks(txt) == {"a.py": "A=1\n", "pkg/b.py": "B=2\n"}
    assert parse_files_or_single("```python\nZ=1\n```", "m.py") == {"m.py": "Z=1\n"}


def test_pass_at_k_matches_known_values():
    assert pass_at_k(1, 1, 1) == 1.0 and pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
    assert pass_at_k(10, 1, 10) == 1.0
    assert pass_at_k(4, 1, 2) == pytest.approx(1 - 3 / 6)   # 1 - C(3,2)/C(4,2)


def test_safe_relpath_blocks_traversal():
    for bad in ("../x.py", "/etc/passwd", "a/../../b.py", "..\\x.py", "\\etc\\x",
                "C:\\x.py", "C:/x.py", ""):
        with pytest.raises(ValueError):
            safe_relpath(bad)
    assert safe_relpath("pkg/mod.py") == "pkg/mod.py"


def test_parse_pytest_summary():
    r = parse_pytest("....F\n=== 1 failed, 4 passed in 0.1s ===")
    assert (r.passed, r.failed, r.all_passed) == (4, 1, False)
    assert parse_pytest("=== 3 passed in 0.1s ===").all_passed
    assert not parse_pytest("=== no tests ran in 0.01s ===").all_passed


def test_sandbox_runs_and_detects_failure():
    ok = run_pytest({"m.py": "def f():\n    return 2\n"},
                    {"test_m.py": "from m import f\ndef test_f():\n    assert f() == 2\n"})
    assert ok.all_passed and ok.passed == 1
    bad = run_pytest({"m.py": "def f():\n    return 3\n"},
                     {"test_m.py": "from m import f\ndef test_f():\n    assert f() == 2\n"})
    assert not bad.all_passed and "assert" in bad.traceback


def test_sandbox_preserves_unicode_source():
    result = run_pytest(
        {"m.py": "VALUE = '\U0001f393'\n"},
        {"test_m.py": "from m import VALUE\ndef test_value():\n    assert VALUE == '\\U0001f393'\n"},
    )
    assert result.all_passed


def test_sandbox_requires_zero_pytest_exit_code(monkeypatch):
    def interrupted(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, "1 passed in 0.1s\n", "")

    monkeypatch.setattr(subprocess, "run", interrupted)
    result = run_pytest({}, {})
    assert not result.all_passed
    assert "exit code 2" in result.traceback


def test_sandbox_timeout_on_infinite_loop():
    r = run_pytest({"m.py": "def f():\n    while True:\n        pass\n"},
                   {"test_m.py": "from m import f\ndef test_f():\n    f()\n"}, timeout_s=3)
    assert not r.all_passed and "TIMEOUT" in r.traceback


def test_sandbox_rejects_path_traversal():
    with pytest.raises(ValueError):
        run_pytest({"../evil.py": "x=1"}, {})


def test_sandbox_scrubs_secrets_from_env(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "hunter2")
    rc, out = run_script("import os\nprint(os.environ.get('SECRET_TOKEN'))")
    assert rc == 0 and out.strip() == "None"


def test_static_analysis_flags_insecure_code():
    res = analyze({"m.py": "import subprocess\ndef f(c):\n    return eval(c)\n"
                           "def g(c):\n    subprocess.call(c, shell=True)\n"})
    assert res["bandit"]["available"] and res["blocking_security"] >= 1
    clean = analyze({"m.py": "def f(a: int) -> int:\n    return a + 1\n"})
    assert clean["blocking_security"] == 0 and clean["standards_score"] == 100


def test_failure_signature_stable():
    tb = "def test():\n>   assert 1 == 2\nE   assert 1 == 2\nE    +  where"
    assert failure_signature(tb) == failure_signature(tb)
    assert failure_signature("") == "unknown"
