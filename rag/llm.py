"""Step 6 of RAG: ask an LLM to write the answer from the retrieved passages.

Supported providers (chosen with environment variables or the .env file):

  * ollama             - a free model running on YOUR computer (no key, no account)
  * openai_compatible  - any provider with an OpenAI-style API: Groq, Google Gemini,
                         OpenRouter, ... (many have a free tier)
  * anthropic          - Claude through the Anthropic API (paid)

If nothing is configured the app still works in "retrieval-only" mode, so you can test
the search part for free.
"""
import json
import os
import urllib.error
import urllib.request

from . import config
from .prompt import SYSTEM_PROMPT, build_prompt

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
REQUEST_TIMEOUT_SECONDS = 300  # a small local model on a laptop CPU can be slow


def _load_env():
    from dotenv import load_dotenv

    load_dotenv()


def get_settings():
    """Read the LLM settings from the environment. Returns a dict, or None if not configured."""
    _load_env()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower().replace("-", "_")
    if not provider:
        provider = "anthropic" if os.getenv("ANTHROPIC_API_KEY") else ""

    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return None
        return {"provider": "anthropic", "api_key": key,
                "model": os.getenv("LLM_MODEL") or config.llm_model()}
    if provider == "ollama":
        return {"provider": "ollama", "api_key": "",
                "base_url": os.getenv("LLM_BASE_URL", OLLAMA_BASE_URL),
                "model": os.getenv("LLM_MODEL", DEFAULT_OLLAMA_MODEL)}
    if provider == "openai_compatible":
        base_url, model, key = (os.getenv("LLM_BASE_URL"), os.getenv("LLM_MODEL"),
                                os.getenv("LLM_API_KEY"))
        if not (base_url and model and key):
            return None
        return {"provider": "openai_compatible", "base_url": base_url,
                "model": model, "api_key": key}
    return None


def describe_llm():
    """Short text for the app's sidebar."""
    settings = get_settings()
    if settings is None:
        return "retrieval-only (no LLM configured)"
    return f"{settings['provider']} / {settings['model']}"


def llm_configured():
    return get_settings() is not None


def _chat_openai_compatible(settings, question, passages):
    url = settings["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": settings["model"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": build_prompt(question, passages)}],
        "temperature": 0.2,
        "max_tokens": 700,
    }
    headers = {"Content-Type": "application/json"}
    if settings.get("api_key"):
        headers["Authorization"] = "Bearer " + settings["api_key"]
    request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def _chat_anthropic(settings, question, passages):
    from anthropic import Anthropic

    client = Anthropic(api_key=settings["api_key"])
    message = client.messages.create(
        model=settings["model"], max_tokens=700, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt(question, passages)}])
    return message.content[0].text


def _friendly_error(settings, exc):
    provider = settings["provider"]
    if isinstance(exc, urllib.error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")[:300]
        except Exception:
            pass
        if provider == "ollama" and exc.code == 404:
            return (f"Ollama does not have the model '{settings['model']}'. "
                    f"Run: ollama pull {settings['model']}")
        return f"The LLM provider returned HTTP {exc.code}. {body}"
    if isinstance(exc, urllib.error.URLError):
        if provider == "ollama":
            return ("Could not reach Ollama at " + settings["base_url"] +
                    ". Is the Ollama app running? (Look for its icon near the clock, "
                    "or start it from the Start menu.)")
        return f"Could not reach the LLM provider: {exc.reason}"
    return f"The LLM call failed: {exc}"


def generate_answer(question, passages):
    """Return {"answer": str, "mode": "llm" | "retrieval-only" | "error", "model": str}."""
    settings = get_settings()
    if settings is None:
        return {"answer": ("Retrieval-only mode: no LLM is configured, so no answer was "
                           "written. The most relevant passages are listed below."),
                "mode": "retrieval-only", "model": ""}
    try:
        if settings["provider"] == "anthropic":
            text = _chat_anthropic(settings, question, passages)
        else:
            text = _chat_openai_compatible(settings, question, passages)
        return {"answer": text, "mode": "llm",
                "model": f"{settings['provider']} / {settings['model']}"}
    except Exception as exc:
        return {"answer": _friendly_error(settings, exc), "mode": "error",
                "model": f"{settings['provider']} / {settings['model']}"}
