"""The chat graph: START → chat → END, with every conversation saved to SQLite by thread_id."""

import sqlite3
from dataclasses import dataclass
from typing import Iterator

from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime

from config import DB_PATH, HISTORY_BUDGET_RATIO
from core.providers import create_llm


@dataclass
class ChatContext:
    """Per-request settings. Passed as runtime context (not state), so it is never saved to the database —
    which matters because it carries the API key."""

    provider: str
    model: str
    temperature: float
    api_key: str
    context_window: int
    system_prompt: str


def message_text(message: BaseMessage) -> str:
    """Some providers (Anthropic, Gemini) return content as a list of blocks instead of a plain string."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str) or block.get("type") == "text"
    )


def fit_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Keep only the most recent messages that fit the model's context. SQLite still keeps everything."""
    trimmed = trim_messages(
        messages,
        max_tokens=max_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",
    )
    # A single huge message can exceed the budget on its own; still send it rather than nothing.
    return trimmed or messages[-1:]


def _for_model(message: BaseMessage) -> BaseMessage:
    # User messages carry UI-only extras (display text, attachment info) that shouldn't reach the provider.
    if isinstance(message, HumanMessage):
        return HumanMessage(content=message.content)
    return message


def chat_node(state: MessagesState, runtime: Runtime[ChatContext]) -> dict:
    ctx = runtime.context
    llm = create_llm(ctx.provider, ctx.model, ctx.temperature, ctx.api_key, ctx.context_window)

    system = SystemMessage(content=ctx.system_prompt)
    budget = int(ctx.context_window * HISTORY_BUDGET_RATIO) - count_tokens_approximately([system])
    history = fit_history(state["messages"], max(budget, 256))

    response = llm.invoke([system, *[_for_model(m) for m in history]])
    return {"messages": [response]}


def create_checkpointer() -> SqliteSaver:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)


def build_chat_graph(checkpointer: SqliteSaver):
    builder = StateGraph(MessagesState, context_schema=ChatContext)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


def _thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def stream_reply(graph, thread_id: str, user_message: HumanMessage, context: ChatContext) -> Iterator[str]:
    """Yield the assistant's reply token by token. The checkpointer saves both messages automatically."""
    for chunk, metadata in graph.stream(
        {"messages": [user_message]},
        _thread_config(thread_id),
        context=context,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "chat" and isinstance(chunk, AIMessageChunk):
            text = message_text(chunk)
            if text:
                yield text


def get_messages(graph, thread_id: str) -> list[BaseMessage]:
    return graph.get_state(_thread_config(thread_id)).values.get("messages", [])


def delete_thread(checkpointer: SqliteSaver, thread_id: str) -> None:
    checkpointer.delete_thread(thread_id)
