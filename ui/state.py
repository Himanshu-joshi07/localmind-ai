"""Session state: which chat is open, and the user's settings (loaded from / saved to SQLite)."""

import streamlit as st

from config import CUSTOM_MODEL_OPTION, DEFAULT_SETTINGS, PROVIDERS
from core import database

SETTING_KEYS = list(DEFAULT_SETTINGS)


def init_session() -> None:
    if st.session_state.get("_initialized"):
        return

    saved = {**DEFAULT_SETTINGS, **database.load_settings()}
    if saved["provider"] not in PROVIDERS:
        saved["provider"] = DEFAULT_SETTINGS["provider"]

    for key in SETTING_KEYS:
        st.session_state[key] = saved[key]
    # The sidebar picks from a list; "model" is resolved from these two widgets.
    st.session_state.model_choice = saved["model"]
    st.session_state.custom_model = saved["model"]

    st.session_state.thread_id = None
    st.session_state.session_keys = {}
    st.session_state.pending_prompt = None
    st.session_state.flash = None
    st.session_state.confirm_delete = None
    st.session_state._saved = saved
    st.session_state._initialized = True


def current_model() -> str:
    if st.session_state.model_choice == CUSTOM_MODEL_OPTION:
        return st.session_state.get("custom_model", "").strip()
    return st.session_state.model_choice


def current_settings() -> dict:
    settings = {key: st.session_state[key] for key in SETTING_KEYS}
    settings["model"] = current_model()
    return settings


def persist_settings() -> None:
    settings = current_settings()
    if settings != st.session_state._saved:
        database.save_settings(settings)
        st.session_state._saved = settings


def flash(kind: str, message: str) -> None:
    """Show a message after the next rerun (st.rerun() would otherwise wipe it)."""
    st.session_state.flash = (kind, message)


def open_chat(thread_id: str | None) -> None:
    st.session_state.thread_id = thread_id
    st.session_state.confirm_delete = None
