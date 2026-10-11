import html
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from config import APP_ICON, APP_NAME, HARD_UPLOAD_LIMIT_MB, PROVIDERS, SUPPORTED_FILE_TYPES, TITLE_MAX_CHARS
from core import database
from core.file_loader import FileLoadError, LoadedFile, load_file
from core.graph import ChatContext, get_messages, message_text, stream_reply
from core.providers import OllamaStatus, friendly_error, needs_api_key, resolve_api_key
from core.summarizer import run_summarizer
from ui.state import current_settings, flash

USER_AVATAR = "🧑‍💻"
AI_AVATAR = APP_ICON
DEFAULT_FILE_QUESTION = "Summarize this file and list the key points."

SUGGESTIONS = [
    ("💡 Explain a concept", "Explain how neural networks learn, in simple terms with an everyday analogy."),
    ("📄 Summarize a file", None),
    ("✍️ Help me write", "Help me write a short, polite email asking my manager for a day off next Friday."),
    ("🐍 Debug my code", "What are the most common causes of `IndexError: list index out of range` in Python, "
                        "and how do I fix each one?"),
]


# ---------- Message helpers ----------

def display_text(message: BaseMessage) -> str:
    """User messages store the full prompt (incl. file text) as content, and what they typed as 'display'."""
    if isinstance(message, HumanMessage):
        return message.additional_kwargs.get("display") or message_text(message)
    return message_text(message)


def chat_to_markdown(title: str, messages: list[BaseMessage]) -> str:
    lines = [f"# {title}", "", f"*Exported from {APP_NAME}*", ""]
    for message in messages:
        if isinstance(message, HumanMessage):
            attachment = message.additional_kwargs.get("attachment")
            file_note = f"\n\n📄 *{attachment['name']}*" if attachment else ""
            lines += [f"### 🧑‍💻 You\n\n{display_text(message)}{file_note}", ""]
        elif isinstance(message, AIMessage) and message_text(message):
            lines += [f"### {AI_AVATAR} {APP_NAME}\n\n{message_text(message)}", ""]
    return "\n".join(lines)


def _make_title(question: str, file_name: str | None) -> str:
    text = " ".join(question.split()) if question != DEFAULT_FILE_QUESTION or not file_name else f"📄 {file_name}"
    return text if len(text) <= TITLE_MAX_CHARS else text[: TITLE_MAX_CHARS - 1].rstrip() + "…"


# ---------- Rendering ----------

def _file_chip(attachment: dict) -> None:
    details = [html.escape(attachment["name"])]
    if attachment.get("pages"):
        details.append(f"{attachment['pages']} page{'s' if attachment['pages'] != 1 else ''}")
    if attachment.get("summarized"):
        details.append(f"summarized from {attachment['parts']} parts")
    st.markdown(f"<div class='file-chip'>📄 {' · '.join(details)}</div>", unsafe_allow_html=True)


def _render_user(text: str, attachment: dict | None) -> None:
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown("<span class='user-marker'></span>", unsafe_allow_html=True)
        st.markdown(text)
        if attachment:
            _file_chip(attachment)


def _render_message(message: BaseMessage) -> None:
    if isinstance(message, HumanMessage):
        _render_user(display_text(message), message.additional_kwargs.get("attachment"))
    elif isinstance(message, AIMessage):
        text = message_text(message)
        if text:
            with st.chat_message("assistant", avatar=AI_AVATAR):
                st.markdown(text)


def _use_suggestion(prompt: str | None) -> None:
    if prompt:
        st.session_state.pending_prompt = prompt
    else:
        flash("info", "Click ＋ in the message box below to attach a PDF, Word, text, Markdown or CSV file, "
                      "then ask your question — or just send it to get a summary.")


def _render_welcome() -> None:
    st.markdown(
        f"<div class='welcome'><div class='welcome-logo'>{APP_ICON}</div>"
        f"<h2>How can I help you today?</h2>"
        f"<p>Ask anything, or attach a file with ＋. Your chats stay on this computer.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, prompt) in enumerate(SUGGESTIONS):
        cols[i % 2].button(label, key=f"suggest_{i}", width="stretch", on_click=_use_suggestion, args=(prompt,))


def _show_flash() -> None:
    if st.session_state.flash:
        kind, message = st.session_state.flash
        (st.error if kind == "error" else st.info)(message)
        st.session_state.flash = None


def render_chat(graph, summarizer, ollama_status: OllamaStatus) -> None:
    settings = current_settings()
    model_label = html.escape(settings["model"] or "no model selected")
    st.markdown(
        f"<div class='model-badge'>{html.escape(PROVIDERS[settings['provider']]['label'])} · {model_label}</div>",
        unsafe_allow_html=True,
    )

    thread_id = st.session_state.thread_id
    messages = get_messages(graph, thread_id) if thread_id else []

    submission = st.chat_input(
        f"Message {APP_NAME}…",
        accept_file=True,
        file_type=SUPPORTED_FILE_TYPES,
        max_upload_size=HARD_UPLOAD_LIMIT_MB,
        submit_mode="stop",
    )

    question, upload = "", None
    if submission is not None:
        question = (submission.text or "").strip()
        upload = submission.files[0] if submission.files else None
    elif st.session_state.pending_prompt:
        question = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if not messages and not question and upload is None:
        _render_welcome()
    for message in messages:
        _render_message(message)
    _show_flash()

    if question or upload is not None:
        _handle_turn(graph, summarizer, settings, ollama_status, question, upload)


# ---------- One chat turn ----------

def _readiness_problem(settings: dict, api_key: str, ollama_status: OllamaStatus) -> str | None:
    provider = settings["provider"]
    if not settings["model"]:
        return "Pick a model in ⚙️ Model settings."
    if provider == "ollama" and not ollama_status.running:
        return "Ollama isn't running. Start it with `ollama serve` (or open the Ollama app), then try again."
    if needs_api_key(provider) and not api_key:
        return (f"No API key for {PROVIDERS[provider]['label']}. Add `{PROVIDERS[provider]['env_var']}` "
                f"to your .env file (then restart the app) or paste a key in ⚙️ Model settings.")
    return None


def _summarize_file(summarizer, loaded: LoadedFile, settings: dict, context: ChatContext) -> tuple[str, bool, int]:
    status = st.status(f"📄 Reading {loaded.name}…")
    result, summarized, parts = "", False, 0
    for event in run_summarizer(summarizer, loaded.name, loaded.text, settings["chunk_tokens"],
                                settings["parallel"], context):
        if event.kind == "progress":
            label = (f"📄 Summarizing {loaded.name} — part {event.done + 1} of {event.total}…"
                     if event.done < event.total else f"📄 Combining {event.total} summaries…")
            status.update(label=label)
        else:
            result, summarized, parts = event.result, event.summarized, event.total
    status.update(
        label=f"📄 Summarized {loaded.name} ({parts} parts)" if summarized else f"📄 Read {loaded.name}",
        state="complete",
    )
    return result, summarized, parts


def _prompt_with_file(question: str, loaded: LoadedFile, file_text: str, summarized: bool) -> str:
    description = loaded.name + (f", {loaded.pages} pages" if loaded.pages else "")
    if summarized:
        header = (f"The user attached the file '{description}'. It was too long to include in full, "
                  f"so here is a detailed summary of it:")
    else:
        header = f"The user attached the file '{description}'. Its full content is below:"
    return f"{question}\n\n---\n{header}\n\n{file_text}"


def _ensure_chat(title: str) -> str:
    if not st.session_state.thread_id:
        st.session_state.thread_id = str(uuid.uuid4())
        database.create_chat(st.session_state.thread_id, title)
    return st.session_state.thread_id


def _handle_turn(graph, summarizer, settings: dict, ollama_status: OllamaStatus, question: str, upload) -> None:
    provider, model = settings["provider"], settings["model"]
    api_key = resolve_api_key(provider, st.session_state.session_keys.get(provider, ""))

    problem = _readiness_problem(settings, api_key, ollama_status)
    if problem:
        st.error(problem)
        return

    loaded = None
    if upload is not None:
        try:
            loaded = load_file(upload.name, upload.getvalue(), settings["max_file_mb"], settings["max_pages"])
        except FileLoadError as exc:
            st.error(str(exc), icon="📄")
            return
    question = question or DEFAULT_FILE_QUESTION

    attachment = {"name": loaded.name, "pages": loaded.pages, "summarized": False, "parts": 0} if loaded else None
    _render_user(question, attachment)

    context = ChatContext(
        provider=provider,
        model=model,
        temperature=settings["temperature"],
        api_key=api_key,
        context_window=settings["context_window"],
        system_prompt=settings["system_prompt"],
    )

    thread_id = None
    with st.chat_message("assistant", avatar=AI_AVATAR):
        try:
            content = question
            if loaded:
                file_text, summarized, parts = _summarize_file(summarizer, loaded, settings, context)
                attachment.update(summarized=summarized, parts=parts)
                content = _prompt_with_file(question, loaded, file_text, summarized)

            thread_id = _ensure_chat(_make_title(question, loaded.name if loaded else None))
            user_message = HumanMessage(
                content=content,
                additional_kwargs={"display": question, "attachment": attachment},
            )
            st.write_stream(stream_reply(graph, thread_id, user_message, context))
        except Exception as exc:
            flash("error", friendly_error(exc, provider, model))

    if thread_id:
        database.touch_chat(thread_id)
    st.rerun()
