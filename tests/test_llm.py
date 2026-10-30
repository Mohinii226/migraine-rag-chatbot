import contextlib
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from rag import llm

PASSAGES = [{"text": "XGBoost reached the highest accuracy.", "source": "paper.pdf", "page": 3, "score": 0.9}]
ENV_KEYS = ["LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
            "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"]


@contextlib.contextmanager
def env(**values):
    """Set environment variables for one test and restore them afterwards (ignores any real .env)."""
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    original_loader = llm._load_env
    llm._load_env = lambda: None
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(values)
    try:
        yield
    finally:
        llm._load_env = original_loader
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def fake_llm_server(reply="Fake answer [1]."):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(length))
            payload = json.dumps({"choices": [{"message": {"content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()


def test_nothing_configured_means_retrieval_only():
    with env():
        assert llm.get_settings() is None
        result = llm.generate_answer("question?", PASSAGES)
        assert result["mode"] == "retrieval-only"
        assert "retrieval-only" in llm.describe_llm()


def test_openai_compatible_provider_sends_key_and_prompt():
    with fake_llm_server() as (base_url, seen):
        with env(LLM_PROVIDER="openai_compatible", LLM_BASE_URL=base_url,
                 LLM_API_KEY="secret", LLM_MODEL="test-model"):
            result = llm.generate_answer("Which model won?", PASSAGES)
    assert result["mode"] == "llm" and result["answer"] == "Fake answer [1]."
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "test-model"
    roles = [m["role"] for m in seen["body"]["messages"]]
    assert roles == ["system", "user"]
    assert "[1] (source: paper.pdf, page 3)" in seen["body"]["messages"][1]["content"]


def test_ollama_needs_no_key_and_has_a_default_model():
    with fake_llm_server() as (base_url, seen):
        with env(LLM_PROVIDER="ollama", LLM_BASE_URL=base_url):
            result = llm.generate_answer("Which model won?", PASSAGES)
    assert result["mode"] == "llm"
    assert seen["auth"] is None
    assert seen["body"]["model"] == llm.DEFAULT_OLLAMA_MODEL


def test_unreachable_ollama_gives_a_helpful_message():
    with env(LLM_PROVIDER="ollama", LLM_BASE_URL="http://127.0.0.1:9/v1"):
        result = llm.generate_answer("question?", PASSAGES)
    assert result["mode"] == "error"
    assert "Ollama" in result["answer"] and "running" in result["answer"]


def test_incomplete_openai_compatible_settings_are_not_used():
    with env(LLM_PROVIDER="openai_compatible", LLM_BASE_URL="https://example.com/v1"):
        assert llm.get_settings() is None


def test_anthropic_is_chosen_when_only_its_key_exists():
    with env(ANTHROPIC_API_KEY="k"):
        settings = llm.get_settings()
        assert settings["provider"] == "anthropic" and settings["model"]
