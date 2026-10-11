"""File summarizer graph (map-reduce).

    START → split ──(small file)──► use_full_text ──► END
                  └─(big file)───► summarize_chunk ×N (parallel, via Send) ──► combine ──► END
"""

import operator
from dataclasses import dataclass, replace
from typing import Annotated, Iterator, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send

from config import CHARS_PER_TOKEN, DIRECT_FILE_RATIO
from core.graph import ChatContext, message_text
from core.providers import create_llm


class SummaryState(TypedDict):
    file_name: str
    text: str
    chunk_tokens: int
    chunks: list[str]
    # Parallel chunk summaries land here; the reducer appends instead of overwriting.
    chunk_summaries: Annotated[list[tuple[int, str]], operator.add]
    result: str


class ChunkTask(TypedDict):
    file_name: str
    index: int
    total: int
    chunk: str


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def split_into_chunks(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split on paragraph boundaries where possible; hard-split paragraphs that are too long.

    Deliberately not langchain-text-splitters: it imports `transformers` when installed,
    which makes Streamlit's file watcher try to load torch.
    """
    pieces = []
    for paragraph in text.split("\n\n"):
        while len(paragraph) > chunk_chars:
            pieces.append(paragraph[:chunk_chars])
            paragraph = paragraph[chunk_chars:]
        if paragraph.strip():
            pieces.append(paragraph)

    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(piece) + 2 > chunk_chars:
            chunks.append(current)
            current = current[-overlap_chars:] if overlap_chars else ""
        current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        chunks.append(current)
    return chunks


def split(state: SummaryState, runtime: Runtime[ChatContext]) -> dict:
    window = runtime.context.context_window
    if estimate_tokens(state["text"]) <= window * DIRECT_FILE_RATIO:
        return {"chunks": []}

    # A chunk plus its prompt must fit in one model call.
    chunk_chars = min(state["chunk_tokens"], int(window * 0.6)) * CHARS_PER_TOKEN
    return {"chunks": split_into_chunks(state["text"], chunk_chars, min(200, chunk_chars // 10))}


def route_by_size(state: SummaryState):
    if not state["chunks"]:
        return "use_full_text"
    total = len(state["chunks"])
    return [
        Send("summarize_chunk", {"file_name": state["file_name"], "index": i, "total": total, "chunk": chunk})
        for i, chunk in enumerate(state["chunks"])
    ]


def use_full_text(state: SummaryState) -> dict:
    return {"result": state["text"]}


def _summary_llm(ctx: ChatContext):
    ctx = replace(ctx, temperature=0.2)
    return create_llm(ctx.provider, ctx.model, ctx.temperature, ctx.api_key, ctx.context_window)


def summarize_chunk(task: ChunkTask, runtime: Runtime[ChatContext]) -> dict:
    llm = _summary_llm(runtime.context)
    response = llm.invoke(
        [
            SystemMessage(content="You summarize parts of documents accurately and concisely."),
            HumanMessage(
                content=(
                    f"This is part {task['index'] + 1} of {task['total']} of the document "
                    f"'{task['file_name']}'.\n\nSummarize it in at most 150 words. Keep key facts, "
                    f"numbers, names, and conclusions.\n\n---\n{task['chunk']}"
                )
            ),
        ]
    )
    return {"chunk_summaries": [(task["index"], message_text(response))]}


def combine(state: SummaryState, runtime: Runtime[ChatContext]) -> dict:
    # Parallel branches can finish in any order, so sort by chunk index.
    ordered = [text for _, text in sorted(state["chunk_summaries"])]
    parts = "\n\n".join(f"Part {i + 1}:\n{text}" for i, text in enumerate(ordered))
    llm = _summary_llm(runtime.context)
    response = llm.invoke(
        [
            SystemMessage(content="You combine partial summaries into one clear, well-structured summary."),
            HumanMessage(
                content=(
                    f"Below are summaries of consecutive parts of the document '{state['file_name']}'. "
                    f"Combine them into one coherent, detailed summary of the whole document, "
                    f"with the key points as bullet points.\n\n{parts}"
                )
            ),
        ]
    )
    return {"result": message_text(response)}


def build_summarizer_graph():
    builder = StateGraph(SummaryState, context_schema=ChatContext)
    builder.add_node("split", split)
    builder.add_node("use_full_text", use_full_text)
    builder.add_node("summarize_chunk", summarize_chunk)
    builder.add_node("combine", combine)

    builder.add_edge(START, "split")
    builder.add_conditional_edges("split", route_by_size, ["use_full_text", "summarize_chunk"])
    builder.add_edge("summarize_chunk", "combine")
    builder.add_edge("use_full_text", END)
    builder.add_edge("combine", END)
    return builder.compile()


@dataclass
class SummaryEvent:
    kind: str  # "progress" or "done"
    done: int = 0
    total: int = 0
    result: str = ""
    summarized: bool = False


def run_summarizer(graph, file_name: str, text: str, chunk_tokens: int, parallel: int, context: ChatContext) -> Iterator[SummaryEvent]:
    """Run the summarizer and report progress after each chunk finishes."""
    total = done = 0
    result = ""
    for update in graph.stream(
        {"file_name": file_name, "text": text, "chunk_tokens": chunk_tokens, "chunk_summaries": []},
        {"max_concurrency": parallel},
        context=context,
        stream_mode="updates",
    ):
        for node, values in update.items():
            if node == "split":
                total = len(values["chunks"])
                if total:
                    yield SummaryEvent("progress", done=0, total=total)
            elif node == "summarize_chunk":
                done += 1
                yield SummaryEvent("progress", done=done, total=total)
            elif node in ("use_full_text", "combine"):
                result = values["result"]
    yield SummaryEvent("done", result=result, summarized=total > 0, total=total)
