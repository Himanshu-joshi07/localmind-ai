from datetime import datetime

import streamlit as st

from config import (
    APP_ICON,
    APP_NAME,
    APP_TAGLINE,
    CONTEXT_WINDOW_OPTIONS,
    CUSTOM_MODEL_OPTION,
    CUSTOM_PRESET,
    DEFAULT_SYSTEM_PROMPT,
    HARD_UPLOAD_LIMIT_MB,
    PRESETS,
    PROVIDERS,
)
from core import database
from core.graph import delete_thread, get_messages
from core.providers import OllamaStatus, env_api_key, mask_key, needs_api_key
from ui.chat_view import chat_to_markdown
from ui.state import open_chat


def render_sidebar(graph, checkpointer, ollama_status: OllamaStatus) -> None:
    with st.sidebar:
        st.markdown(
            f"<div class='brand'><span class='brand-icon'>{APP_ICON}</span>"
            f"<span class='brand-name'>{APP_NAME}</span></div>"
            f"<div class='brand-tagline'>{APP_TAGLINE}</div>",
            unsafe_allow_html=True,
        )
        st.button("＋ New chat", type="primary", width="stretch", on_click=open_chat, args=(None,))

        _render_history(checkpointer)
        st.divider()
        _render_model_settings(ollama_status)
        _render_file_settings()
        _render_export(graph)
        st.markdown(
            "<div class='sidebar-footer'>🔒 Chats are stored only on this computer, in <code>chat_history.db</code>.</div>",
            unsafe_allow_html=True,
        )


# ---------- Chat history ----------

def _group_label(updated_at: str) -> str:
    days = (datetime.now().astimezone().date() - datetime.fromisoformat(updated_at).astimezone().date()).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days <= 7:
        return "Previous 7 days"
    return "Older"


def _ask_delete(thread_id: str) -> None:
    st.session_state.confirm_delete = thread_id


def _cancel_delete() -> None:
    st.session_state.confirm_delete = None


def _delete_chat(checkpointer, thread_id: str) -> None:
    delete_thread(checkpointer, thread_id)
    database.delete_chat(thread_id)
    if st.session_state.thread_id == thread_id:
        st.session_state.thread_id = None
    st.session_state.confirm_delete = None


def _render_history(checkpointer) -> None:
    chats = database.list_chats()
    if not chats:
        st.caption("No chats yet — say hi! 👋")
        return

    group = None
    for chat in chats:
        label = _group_label(chat["updated_at"])
        if label != group:
            st.markdown(f"<div class='history-group'>{label}</div>", unsafe_allow_html=True)
            group = label

        thread_id = chat["thread_id"]
        if st.session_state.confirm_delete == thread_id:
            st.caption(f"Delete “{chat['title']}”?")
            yes, no = st.columns(2)
            yes.button("Delete", key=f"yes_{thread_id}", type="primary", width="stretch",
                       on_click=_delete_chat, args=(checkpointer, thread_id))
            no.button("Cancel", key=f"no_{thread_id}", width="stretch", on_click=_cancel_delete)
            continue

        is_open = thread_id == st.session_state.thread_id
        title_col, delete_col = st.columns([0.84, 0.16], gap="small", vertical_alignment="center")
        title_col.button(
            chat["title"],
            key=f"open_{thread_id}",
            type="secondary" if is_open else "tertiary",
            width="stretch",
            wrap=False,
            on_click=open_chat,
            args=(thread_id,),
        )
        delete_col.button("🗑", key=f"del_{thread_id}", type="tertiary", help="Delete chat",
                          on_click=_ask_delete, args=(thread_id,))


# ---------- Model settings ----------

def _on_provider_change() -> None:
    models = PROVIDERS[st.session_state.provider]["models"]
    st.session_state.model_choice = models[0] if models else ""


def _store_session_key(provider: str, widget_key: str) -> None:
    st.session_state.session_keys[provider] = st.session_state[widget_key].strip()


def _model_options(provider: str, ollama_status: OllamaStatus) -> list[str]:
    if provider == "ollama":
        if not ollama_status.running:
            st.warning("Ollama isn't running. Start it with `ollama serve` (or open the Ollama app).", icon="⚠️")
        elif not ollama_status.models:
            st.warning("No Ollama models found. Download one, e.g.\n\n`ollama pull qwen2.5:3b-instruct`", icon="⚠️")
        return [*ollama_status.models, CUSTOM_MODEL_OPTION]
    return [*PROVIDERS[provider]["models"], CUSTOM_MODEL_OPTION]


def _render_model_settings(ollama_status: OllamaStatus) -> None:
    with st.expander("⚙️ Model settings", expanded=True):
        st.selectbox(
            "Provider",
            list(PROVIDERS),
            format_func=lambda p: PROVIDERS[p]["label"],
            key="provider",
            on_change=_on_provider_change,
        )
        provider = st.session_state.provider

        options = _model_options(provider, ollama_status)
        choice = st.session_state.model_choice
        if choice not in options:
            # A previously used model that isn't in the list (e.g. typed by hand) goes into the custom field.
            if choice and provider != "ollama":
                st.session_state.custom_model = choice
                st.session_state.model_choice = CUSTOM_MODEL_OPTION
            else:
                st.session_state.model_choice = options[0]

        st.selectbox("Model", options, key="model_choice")
        if st.session_state.model_choice == CUSTOM_MODEL_OPTION:
            st.text_input("Model name", key="custom_model", placeholder="e.g. llama3.2 or gpt-4o")

        if needs_api_key(provider):
            _render_api_key(provider)

        st.slider("Temperature", 0.0, 1.0, step=0.05, key="temperature",
                  help="Lower = focused and predictable. Higher = more creative.")
        st.text_area("System prompt", key="system_prompt", height=110,
                     help="Instructions that shape how LocalMind answers, in every chat.")
        if st.session_state.system_prompt != DEFAULT_SYSTEM_PROMPT:
            st.button("Reset system prompt", type="tertiary",
                      on_click=lambda: st.session_state.update(system_prompt=DEFAULT_SYSTEM_PROMPT))


def _render_api_key(provider: str) -> None:
    env_key = env_api_key(provider)
    env_var = PROVIDERS[provider]["env_var"]
    widget_key = f"_api_key_{provider}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.session_keys.get(provider, "")

    if env_key:
        st.success(f"API key loaded from .env ({mask_key(env_key)})", icon="✅")
        label = "Use a different key (this session only)"
    else:
        st.info(f"Add `{env_var}` to your .env file, or paste a key below.", icon="🔑")
        label = "API key (this session only)"

    st.text_input(label, type="password", key=widget_key, placeholder="Paste your API key",
                  on_change=_store_session_key, args=(provider, widget_key),
                  help="Kept in memory for this browser session only. Never saved to disk.")


# ---------- File settings ----------

def _apply_preset() -> None:
    values = PRESETS.get(st.session_state.preset)
    if values:
        for key, value in values.items():
            st.session_state[key] = value


def _mark_custom() -> None:
    values = PRESETS.get(st.session_state.preset)
    if values and any(st.session_state[key] != value for key, value in values.items()):
        st.session_state.preset = CUSTOM_PRESET


def _render_file_settings() -> None:
    with st.expander("📎 File settings"):
        st.selectbox("Preset", [*PRESETS, CUSTOM_PRESET], key="preset", on_change=_apply_preset,
                     help="Pick what fits your computer. Any slider change switches to Custom.")
        st.slider("Max file size (MB)", 1, HARD_UPLOAD_LIMIT_MB, key="max_file_mb", on_change=_mark_custom,
                  help=f"Hard cap is {HARD_UPLOAD_LIMIT_MB} MB (set in .streamlit/config.toml).")
        st.slider("Max pages", 1, 500, key="max_pages", on_change=_mark_custom,
                  help="Longer PDFs / Word files are rejected before processing.")
        st.select_slider("Context window (tokens)", CONTEXT_WINDOW_OPTIONS, key="context_window",
                         on_change=_mark_custom,
                         help="How much text the model sees at once. For Ollama this sets num_ctx — "
                              "higher needs more RAM/VRAM. Also decides when files get summarized "
                              "and how much chat history is sent.")
        st.slider("Chunk size (tokens)", 500, 16000, step=500, key="chunk_tokens", on_change=_mark_custom,
                  help="Size of each piece when a big file is split for summarizing.")
        st.slider("Parallel summaries", 1, 8, key="parallel", on_change=_mark_custom,
                  help="How many chunks are summarized at the same time. Keep low on slower machines.")


# ---------- Export ----------

def _render_export(graph) -> None:
    thread_id = st.session_state.thread_id
    if not thread_id:
        return
    messages = get_messages(graph, thread_id)
    chat = database.get_chat(thread_id)
    if not messages or not chat:
        return
    safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in chat["title"]).strip() or "chat"
    st.download_button(
        "⬇️ Export chat (.md)",
        chat_to_markdown(chat["title"], messages),
        file_name=f"{safe_name}.md",
        mime="text/markdown",
        width="stretch",
        on_click="ignore",
    )
