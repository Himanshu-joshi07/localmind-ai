from pathlib import Path

APP_NAME = "LocalMind"
APP_ICON = "🧠"
APP_TAGLINE = "Your private AI chat — powered by LangGraph"

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chat_history.db"

DEFAULT_SYSTEM_PROMPT = (
    "You are LocalMind, a helpful, friendly AI assistant. "
    "Answer clearly and concisely. Use Markdown for formatting, "
    "and code blocks with a language tag for code."
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"

# Edit these lists to add/remove suggested models. Any model can also be typed in via "Custom model…".
PROVIDERS = {
    "ollama": {
        "label": "Ollama (local)",
        "env_var": None,
        "models": [],  # auto-detected from the local Ollama server
    },
    "openai": {
        "label": "OpenAI",
        "env_var": "OPENAI_API_KEY",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"],
    },
    "anthropic": {
        "label": "Anthropic",
        "env_var": "ANTHROPIC_API_KEY",
        "models": ["claude-sonnet-5", "claude-opus-5-5", "claude-haiku-4-5-20251001"],
    },
    "groq": {
        "label": "Groq",
        "env_var": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "gemini": {
        "label": "Google Gemini",
        "env_var": "GOOGLE_API_KEY",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro"],
    },
}

CUSTOM_MODEL_OPTION = "Custom model…"

SUPPORTED_FILE_TYPES = ["pdf", "docx", "txt", "md", "csv"]

# Must stay <= maxUploadSize in .streamlit/config.toml (Streamlit's hard cap).
HARD_UPLOAD_LIMIT_MB = 50

CONTEXT_WINDOW_OPTIONS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

PRESETS = {
    "🐢 Low-end": {
        "max_file_mb": 5,
        "max_pages": 20,
        "context_window": 4096,
        "chunk_tokens": 1500,
        "parallel": 1,
    },
    "⚖️ Balanced": {
        "max_file_mb": 10,
        "max_pages": 50,
        "context_window": 8192,
        "chunk_tokens": 3000,
        "parallel": 2,
    },
    "🚀 High-end / Cloud": {
        "max_file_mb": 50,
        "max_pages": 300,
        "context_window": 32768,
        "chunk_tokens": 8000,
        "parallel": 5,
    },
}
DEFAULT_PRESET = "⚖️ Balanced"
CUSTOM_PRESET = "🛠️ Custom"

DEFAULT_SETTINGS = {
    "provider": "ollama",
    "model": "qwen2.5:3b-instruct",
    "temperature": 0.7,
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "preset": DEFAULT_PRESET,
    **PRESETS[DEFAULT_PRESET],
}

# Share of the context window that chat history may use; the rest is left for the reply.
HISTORY_BUDGET_RATIO = 0.75
# A file is sent to the model whole if it uses at most this share of the context window;
# otherwise it is summarized with the map-reduce graph first.
DIRECT_FILE_RATIO = 0.5
CHARS_PER_TOKEN = 4

TITLE_MAX_CHARS = 40
