"""Model "plugin" layer: one function that turns (provider, model, settings) into a LangChain chat model."""

import json
import os
import urllib.request
from dataclasses import dataclass, field

from config import DEFAULT_OLLAMA_URL, PROVIDERS


class MissingAPIKeyError(Exception):
    pass


@dataclass
class OllamaStatus:
    running: bool
    models: list[str] = field(default_factory=list)
    error: str = ""


def ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_URL


def check_ollama() -> OllamaStatus:
    try:
        with urllib.request.urlopen(f"{ollama_url()}/api/tags", timeout=2) as resp:
            data = json.load(resp)
    except Exception:
        return OllamaStatus(running=False, error="Ollama is not running.")

    models = [
        m["name"]
        for m in data.get("models", [])
        if "embedding" not in m.get("capabilities", []) and "embed" not in m["name"]
    ]
    return OllamaStatus(running=True, models=sorted(models))


def env_api_key(provider: str) -> str:
    env_var = PROVIDERS[provider]["env_var"]
    return (os.getenv(env_var) or "").strip() if env_var else ""


def resolve_api_key(provider: str, session_key: str = "") -> str:
    """A key typed in the sidebar wins over .env for that session."""
    return (session_key or "").strip() or env_api_key(provider)


def needs_api_key(provider: str) -> bool:
    return PROVIDERS[provider]["env_var"] is not None


def mask_key(key: str) -> str:
    if len(key) <= 8:
        return "••••"
    return f"{key[:3]}…{key[-4:]}"


def create_llm(provider: str, model: str, temperature: float, api_key: str = "", context_window: int = 8192):
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model, temperature=temperature, num_ctx=context_window, base_url=ollama_url())

    if not api_key:
        env_var = PROVIDERS[provider]["env_var"]
        raise MissingAPIKeyError(
            f"No API key for {PROVIDERS[provider]['label']}. "
            f"Add `{env_var}` to your .env file or paste a key in the sidebar."
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, api_key=api_key)
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature, api_key=api_key)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)

    raise ValueError(f"Unknown provider: {provider}")


def friendly_error(exc: Exception, provider: str, model: str) -> str:
    """Turn common provider failures into a message a non-developer can act on."""
    if isinstance(exc, MissingAPIKeyError):
        return str(exc)

    text = str(exc)
    lowered = text.lower()
    if provider == "ollama":
        if "not found" in lowered and "model" in lowered:
            return f"Model `{model}` isn't downloaded. Run this in a terminal:\n\n`ollama pull {model}`"
        if "connect" in lowered or "refused" in lowered:
            return "Can't reach Ollama. Start it with `ollama serve` (or open the Ollama app) and try again."
    if any(word in lowered for word in ("401", "unauthorized", "invalid api key", "authentication", "api_key_invalid")):
        return f"{PROVIDERS[provider]['label']} rejected the API key. Check that it's correct and active."
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered:
        return f"{PROVIDERS[provider]['label']} rate limit or quota reached. Wait a bit or check your plan."
    if "404" in lowered or "does not exist" in lowered or "not_found" in lowered:
        return f"Model `{model}` wasn't found on {PROVIDERS[provider]['label']}. Pick another model."
    return f"Something went wrong: {text}"
