"""Router agent determining user message intent, standalone rewrite, and dispatch parameters."""
import re
from typing import List, Dict, Any, Optional
from app.config import (
    GROQ_MODEL_FAST,
    ROUTER_HISTORY_TURNS,
    ROUTER_MIN_CONFIDENCE,
)
from app.llm import generate_json
from app.agents.schemas import RouterDecision

ROUTER_PROMPT = """You are a conversational router for an academic research assistant.
Analyze the user's message in the context of recent chat history and the available document sections.

Recent Chat History (most recent last, up to {history_turns} turns):
{history_str}

Last Output Kind: {last_output_kind}
Last Output Text Preview (first 300 chars):
{last_output_preview}

Available Document Sections:
{doc_sections_str}

INTENT DEFINITIONS:
1. paper_question: Asking for factual, conceptual, or methodological information from the paper, or questions about machine learning, natural language processing, mathematics, or science related to research papers. Any question about ML, NLP or science—even if the specific paper may not cover it—is paper_question (retrieval and the answer step decide whether the paper covers it).
   Note: A follow-up asking for new or additional information from the paper is paper_question.
2. summarize: Asking to summarize the whole paper or a specific section.
3. notes: Asking to generate comprehensive study notes, key terms, or revision concepts.
4. outline: Asking to generate a presentation outline or slide deck.
5. rephrase_previous: Asking to change the style, length, format, or language of the IMMEDIATELY PRECEDING output without asking for new facts (e.g. simpler, shorter, more concise, analogy, translate). Only valid if Last Output Kind is not 'none'.
6. off_topic: Questions clearly unrelated to research papers or studying (such as weather, general geography trivia, celebrity trivia, unrelated coding tasks, casual chit-chat).
7. needs_clarification: Genuinely ambiguous request where the user's intent cannot be inferred even with conversation context.

RULES:
- A reference such as "the other one", "that", "it" must be resolved ONLY from the visible history. If the history does not contain a clear single referent, return needs_clarification.
- First rewrite the user message as a concise, self-contained `standalone_query` (under 20 words) resolving pronouns and context using the visible history. Do not generate lists of synonyms or repetitive query expansions.
- If intent is summarize and a specific section is requested, set `section` to the exact matching section name from Available Document Sections, or null for whole paper.
- If intent is outline and a slide count is requested, set `num_slides` to that integer.
- When unsure between paper_question and anything else, choose paper_question, except for ambiguous references (which must be needs_clarification).

Return a valid JSON object matching the schema:
{{
  "intent": "<one of: paper_question, summarize, notes, outline, rephrase_previous, off_topic, needs_clarification>",
  "standalone_query": "<self-contained query>",
  "section": "<exact section name or null>",
  "num_slides": <integer or null>,
  "confidence": <float between 0.0 and 1.0>,
  "reason": "<short explanation>"
}}
"""


def _normalize_section_for_matching(name: str) -> str:
    """Strips leading section numbering (e.g. '6 ', '6.1 ') and lowercase for matching."""
    stripped = re.sub(r"^\d+(\.\d+)*\s*", "", name.strip()).strip().lower()
    return stripped


def _find_matching_section(requested: str, available_sections: List[str]) -> Optional[str]:
    """Matches requested section name against available sections ignoring leading numbers and case."""
    if not requested:
        return None
    req_norm = _normalize_section_for_matching(requested)
    req_raw_norm = requested.strip().lower()

    for sec in available_sections:
        sec_raw_norm = sec.strip().lower()
        sec_norm = _normalize_section_for_matching(sec)
        # Check exact raw match or normalized match
        if req_raw_norm == sec_raw_norm or req_norm == sec_norm:
            return sec
        # Check if requested is substring of section name or vice-versa
        if req_norm and req_norm in sec_norm:
            return sec
    return None


def route(
    user_message: str,
    history: List[Dict[str, str]],
    last_output: Optional[Dict[str, Any]],
    doc_sections: List[str],
) -> RouterDecision:
    """Routes a user message to an appropriate agent intent with Python code guards."""
    # 1. Format history (last ROUTER_HISTORY_TURNS turns)
    recent_history = history[-ROUTER_HISTORY_TURNS:] if history else []
    history_lines = []
    for h in recent_history:
        role = h.get("role", "user").capitalize()
        content = h.get("content", "")
        history_lines.append(f"{role}: {content}")
    history_str = "\n".join(history_lines) if history_lines else "None"

    # 2. Format last_output preview
    last_output_kind = last_output.get("kind", "none") if last_output else "none"
    raw_text = last_output.get("text", "") if last_output else ""
    last_output_preview = (raw_text[:300] + "...") if len(raw_text) > 300 else (raw_text or "none")

    doc_sections_str = "\n".join(f"- {s}" for s in doc_sections) if doc_sections else "None"

    prompt = ROUTER_PROMPT.format(
        history_turns=ROUTER_HISTORY_TURNS,
        history_str=history_str,
        last_output_kind=last_output_kind,
        last_output_preview=last_output_preview,
        doc_sections_str=doc_sections_str,
    )

    try:
        data = generate_json(
            prompt=f"{prompt}\n\nUser Message: {user_message}",
            schema=RouterDecision,
            model=GROQ_MODEL_FAST,
        )
        decision = RouterDecision.model_validate(data)
    except Exception as e:
        # Code guard: invalid JSON after retry -> paper_question with original message
        decision = RouterDecision(
            intent="paper_question",
            standalone_query=user_message,
            section=None,
            num_slides=None,
            confidence=0.0,
            reason=f"Fallback due to router JSON parse error: {e}",
        )

    # ----------------------------------------------------
    # Plain Python Code-Level Guards
    # ----------------------------------------------------

    # Guard 1: rephrase_previous with no last_output -> paper_question
    if decision.intent == "rephrase_previous" and (not last_output or not last_output.get("text")):
        decision.intent = "paper_question"
        decision.reason = "Guard: rephrase_previous converted to paper_question because no last_output exists."

    # Guard 2: low confidence and not off_topic or needs_clarification -> paper_question
    if decision.confidence < ROUTER_MIN_CONFIDENCE and decision.intent not in ("off_topic", "needs_clarification"):
        decision.intent = "paper_question"
        decision.reason = f"Guard: low confidence ({decision.confidence:.2f} < {ROUTER_MIN_CONFIDENCE}) routed to paper_question."

    # Guard 3: summarize with a section that does not match any real section
    if decision.intent == "summarize" and decision.section:
        matched_sec = _find_matching_section(decision.section, doc_sections)
        if matched_sec:
            decision.section = matched_sec
        else:
            decision.intent = "needs_clarification"
            valid_secs_str = ", ".join(f"'{s}'" for s in doc_sections)
            decision.reason = f"Guard: section '{decision.section}' not found. Available sections: {valid_secs_str}."

    # Guard 4: num_slides clamped to 5-15 (default 10)
    if decision.intent == "outline":
        if decision.num_slides is None:
            decision.num_slides = 10
        else:
            decision.num_slides = max(5, min(15, int(decision.num_slides)))

    # Ensure standalone_query is populated
    if not decision.standalone_query:
        decision.standalone_query = user_message

    return decision
