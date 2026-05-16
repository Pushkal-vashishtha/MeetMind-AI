"""
core/extractor.py  —  Upgraded extraction with structured Pydantic outputs,
parallel processing, and richer prompts.
"""

from __future__ import annotations
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pydantic import BaseModel, Field
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ActionItem(BaseModel):
    task: str = Field(description="Clear description of the task")
    owner: str = Field(description="Person responsible. Use 'Unassigned' if unclear.")
    deadline: str = Field(description="Deadline if mentioned, else 'Not specified'")
    priority: str = Field(description="High / Medium / Low — infer from urgency cues")

class Decision(BaseModel):
    decision: str = Field(description="What was decided")
    rationale: str = Field(description="Why this was decided. 'Not stated' if unclear.")
    impact: str = Field(description="High / Medium / Low — infer from context")

class Question(BaseModel):
    question: str = Field(description="The unresolved question or topic")
    context: str = Field(description="Brief context around why it came up")
    owner: str = Field(description="Who raised it, or 'Unknown'")

class ExtractionResult(BaseModel):
    action_items: list[ActionItem] = []
    key_decisions: list[Decision] = []
    open_questions: list[Question] = []


# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.1,   # lower = more deterministic extraction
    )


# ── Generic chain builder ─────────────────────────────────────────────────────

def _build_str_chain(system_prompt: str):
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )


def _build_json_chain(system_prompt: str):
    """Returns a chain that outputs parsed JSON (list or dict)."""
    llm = get_llm()
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
        | RunnableLambda(_safe_parse_json)
    )


def _safe_parse_json(text: str):
    """Strip markdown fences then parse JSON safely."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return []


# ── Individual extractors (structured) ───────────────────────────────────────

ACTION_PROMPT = """You are an expert meeting analyst. Extract every action item from the transcript.

Return ONLY a valid JSON array. Each element must have:
- "task"     : string — what needs to be done
- "owner"    : string — who is responsible (or "Unassigned")
- "deadline" : string — deadline if mentioned (or "Not specified")
- "priority" : string — "High" | "Medium" | "Low" based on urgency language

If there are no action items return an empty array [].
Do NOT include any markdown or explanation — only the JSON array."""

DECISION_PROMPT = """You are an expert meeting analyst. Extract every key decision from the transcript.

Return ONLY a valid JSON array. Each element must have:
- "decision"  : string — what was decided
- "rationale" : string — why (or "Not stated")
- "impact"    : string — "High" | "Medium" | "Low"

If no decisions, return []. Only JSON, no markdown."""

QUESTION_PROMPT = """You are an expert meeting analyst. Extract every unresolved question or open topic.

Return ONLY a valid JSON array. Each element must have:
- "question" : string — the question or open topic
- "context"  : string — brief context
- "owner"    : string — who raised it (or "Unknown")

If none, return []. Only JSON, no markdown."""


def extract_action_items_structured(transcript: str) -> list[dict]:
    chain = _build_json_chain(ACTION_PROMPT)
    result = chain.invoke(transcript)
    return result if isinstance(result, list) else []


def extract_key_decisions_structured(transcript: str) -> list[dict]:
    chain = _build_json_chain(DECISION_PROMPT)
    result = chain.invoke(transcript)
    return result if isinstance(result, list) else []


def extract_questions_structured(transcript: str) -> list[dict]:
    chain = _build_json_chain(QUESTION_PROMPT)
    result = chain.invoke(transcript)
    return result if isinstance(result, list) else []


# ── Parallel extraction (3x faster) ──────────────────────────────────────────

def extract_all_parallel(transcript: str) -> dict:
    """
    Run all three extractions concurrently with ThreadPoolExecutor.
    Returns:
        {
            "action_items": [...],
            "key_decisions": [...],
            "open_questions": [...]
        }
    """
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_actions   = pool.submit(extract_action_items_structured, transcript)
        f_decisions = pool.submit(extract_key_decisions_structured, transcript)
        f_questions = pool.submit(extract_questions_structured, transcript)

    return {
        "action_items":   f_actions.result(),
        "key_decisions":  f_decisions.result(),
        "open_questions": f_questions.result(),
    }


# ── Legacy string interface (backward-compatible with app.py) ─────────────────
# These keep working even if you haven't updated the UI yet.

def _format_action_items(items: list[dict]) -> str:
    if not items:
        return "No action items found."
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. [{it.get('priority','?')} priority] {it.get('task','')}\n"
            f"   Owner: {it.get('owner','Unassigned')} | "
            f"Deadline: {it.get('deadline','Not specified')}"
        )
    return "\n".join(lines)

def _format_decisions(items: list[dict]) -> str:
    if not items:
        return "No key decisions found."
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. {it.get('decision','')}\n"
            f"   Rationale: {it.get('rationale','Not stated')} | "
            f"Impact: {it.get('impact','?')}"
        )
    return "\n".join(lines)

def _format_questions(items: list[dict]) -> str:
    if not items:
        return "No open questions found."
    lines = []
    for i, it in enumerate(items, 1):
        lines.append(
            f"{i}. {it.get('question','')}\n"
            f"   Context: {it.get('context','')} | Raised by: {it.get('owner','Unknown')}"
        )
    return "\n".join(lines)


def extract_action_items(transcript: str) -> str:
    return _format_action_items(extract_action_items_structured(transcript))

def extract_key_decisions(transcript: str) -> str:
    return _format_decisions(extract_key_decisions_structured(transcript))

def extract_questions(transcript: str) -> str:
    return _format_questions(extract_questions_structured(transcript))