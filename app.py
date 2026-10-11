"""LocalMind — your private AI chat, powered by LangGraph.

Run from this folder:  streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from config import APP_ICON, APP_NAME, BASE_DIR

load_dotenv(BASE_DIR / ".env")
st.set_page_config(page_title=APP_NAME, page_icon=APP_ICON, layout="centered", initial_sidebar_state="expanded")

from core import database  # noqa: E402
from core.graph import build_chat_graph, create_checkpointer  # noqa: E402
from core.providers import check_ollama  # noqa: E402
from core.summarizer import build_summarizer_graph  # noqa: E402
from ui.chat_view import render_chat  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.state import init_session, persist_settings  # noqa: E402


@st.cache_resource
def load_graphs():
    database.init_db()
    checkpointer = create_checkpointer()
    return build_chat_graph(checkpointer), build_summarizer_graph(), checkpointer


@st.cache_data(ttl=5, show_spinner=False)
def ollama_status():
    return check_ollama()


chat_graph, summarizer, checkpointer = load_graphs()
init_session()
st.markdown(f"<style>{(BASE_DIR / 'ui' / 'styles.css').read_text()}</style>", unsafe_allow_html=True)

status = ollama_status()
render_sidebar(chat_graph, checkpointer, status)
persist_settings()
render_chat(chat_graph, summarizer, status)
