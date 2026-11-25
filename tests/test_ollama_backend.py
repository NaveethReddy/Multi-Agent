"""OllamaBackend against a local fake of the Ollama HTTP API (no real model needed)."""
import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from config import Config
from llm import LLMClient

SEEN = []


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, body))
        reply = {"response": '{"ok": true}' if body.get("format") == "json" else "hello",
                 "prompt_eval_count": 11, "eval_count": 7}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_payload_and_token_accounting(server):
    SEEN.clear()
    cfg = dataclasses.replace(Config(), ollama_url=server, model="m1", temperature=0.0, seed=7)
    llm = LLMClient(cfg)
    r = llm.call("coder", "sys", "prompt")
    path, body = SEEN[0]
    assert path == "/api/generate" and body["model"] == "m1" and body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 7} and body["system"] == "sys"
    assert (r.text, r.tokens_in, r.tokens_out) == ("hello", 11, 7)
    assert llm.total_tokens == 18 and llm.calls[0]["agent"] == "coder"


def test_json_mode_sets_format(server):
    SEEN.clear()
    llm = LLMClient(dataclasses.replace(Config(), ollama_url=server))
    assert llm.call_json("requirement", "s", "p") == {"ok": True}
    assert SEEN[0][1]["format"] == "json"
