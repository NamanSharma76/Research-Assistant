"""LangGraph orchestration graph and ChatSession for academic research assistant."""
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

from app.config import (
    GROQ_MODEL_MAIN,
    GROQ_MODEL_FAST,
    RELEVANCE_THRESHOLD,
    MAX_QA_RETRIES,
    LLM_CALL_DELAY_SECONDS,
)
from app.llm import generate, get_call_count, reset_call_count
from app.rag.retrieval import search, list_doc_sections
from app.rag.qa import answer_question
from app.rag.prompts import is_refusal
from app.agents.schemas import RouterDecision
from app.agents.router import route
from app.agents.summarizer import summarize_section, summarize_document
from app.agents.notes import generate_notes
from app.agents.outline import generate_outline
from app.agents.render import render_summary, render_notes, render_outline


class AgentState(TypedDict):
    doc_id: str
    user_message: str
    history: List[Dict[str, str]]
    last_output: Dict[str, Any]  # {kind, text}
    decision: Optional[RouterDecision]
    chunks: List[Dict[str, Any]]
    answer: str
    qa_result: Optional[Dict[str, Any]]
    retries: int
    trace: List[str]
    result_text: str


class TurnResult:
    def __init__(
        self,
        answer: str,
        intent: str,
        standalone_query: str,
        trace: List[str],
        llm_calls: int,
        seconds: float,
    ):
        self.answer = answer
        self.intent = intent
        self.standalone_query = standalone_query
        self.trace = trace
        self.llm_calls = llm_calls
        self.seconds = seconds


# ==========================================
# Graph Nodes
# ==========================================

def router_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["router_node"]
    doc_id = state["doc_id"]
    doc_sections = list_doc_sections(doc_id)
    decision = route(
        user_message=state["user_message"],
        history=state.get("history", []),
        last_output=state.get("last_output"),
        doc_sections=doc_sections,
    )
    return {"decision": decision, "trace": trace}


def qa_retrieve(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["qa_retrieve"]
    decision = state["decision"]
    query = decision.standalone_query if decision else state["user_message"]
    doc_id = state["doc_id"]

    all_chunks = search(doc_id=doc_id, query=query, k=5)
    kept_chunks = [c for c in all_chunks if c.get("score", 0.0) >= RELEVANCE_THRESHOLD]
    return {"chunks": kept_chunks, "trace": trace}


def rewrite_query(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["rewrite_query"]
    retries = state.get("retries", 0) + 1
    decision = state["decision"]
    current_q = decision.standalone_query if decision else state["user_message"]

    prompt = (
        f"You are a research paper query optimizer. Rewrite the following student query into a "
        f"keyword-rich search query containing domain terminology and synonyms suitable for finding the exact "
        f"answer in an academic research paper. Do NOT answer the question. Return ONLY the search query keywords.\n\n"
        f"Original Query: {current_q}\n"
        f"Search Keywords:"
    )
    new_query = generate(prompt, model=GROQ_MODEL_FAST)
    time.sleep(LLM_CALL_DELAY_SECONDS)

    # Update decision standalone_query with optimized search keywords
    if decision:
        decision.standalone_query = new_query.strip()

    return {"retries": retries, "decision": decision, "trace": trace}


def qa_answer(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["qa_answer"]
    doc_id = state["doc_id"]
    decision = state["decision"]
    question = decision.standalone_query if decision else state["user_message"]
    chunks = state.get("chunks", [])

    qa_res = answer_question(doc_id=doc_id, question=question, chunks=chunks)
    answer_text = qa_res.get("answer", "")
    return {
        "qa_result": qa_res,
        "answer": answer_text,
        "result_text": answer_text,
        "trace": trace,
    }


def qa_refuse(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["qa_refuse"]
    refusal_text = "The paper does not appear to cover this."
    return {
        "answer": refusal_text,
        "result_text": refusal_text,
        "trace": trace,
    }


def summarize_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["summarize_node"]
    doc_id = state["doc_id"]
    decision = state["decision"]
    section = decision.section if decision else None

    if section:
        text = summarize_section(doc_id, section)
    else:
        doc_summary = summarize_document(doc_id)
        text = render_summary(doc_summary, doc_id)

    return {"result_text": text, "trace": trace}


def notes_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["notes_node"]
    doc_id = state["doc_id"]
    notes_data = generate_notes(doc_id)
    text = render_notes(notes_data, doc_id)
    return {"result_text": text, "trace": trace}


def outline_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["outline_node"]
    doc_id = state["doc_id"]
    decision = state["decision"]
    num_slides = decision.num_slides if decision and decision.num_slides else 10
    outline_data = generate_outline(doc_id, num_slides=num_slides)
    text = render_outline(outline_data, num_slides)
    return {"result_text": text, "trace": trace}


def rephrase_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["rephrase_node"]
    last_text = state.get("last_output", {}).get("text", "")
    instruction = state["user_message"]

    prompt = (
        f"Rewrite the text below according to the instruction. Do not add new facts. "
        f"Keep existing [Section, p. X] citations unchanged.\n\n"
        f"Instruction: {instruction}\n\n"
        f"Text:\n{last_text}\n\n"
        f"Rewritten Text:"
    )
    rephrased = generate(prompt, model=GROQ_MODEL_MAIN)
    time.sleep(LLM_CALL_DELAY_SECONDS)
    return {"result_text": rephrased.strip(), "trace": trace}


def offtopic_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["offtopic_node"]
    text = (
        "I am an academic research assistant dedicated to analyzing the uploaded paper. "
        "Please ask questions related to the paper, its methodology, or its findings."
    )
    return {"result_text": text, "trace": trace}


def clarify_node(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["clarify_node"]
    decision = state.get("decision")
    if decision and "Guard: section" in decision.reason:
        text = decision.reason
    else:
        text = (
            "Could you please clarify your request? You can ask questions about the paper, "
            "request a section summary, generate study notes, or create a presentation outline."
        )
    return {"result_text": text, "trace": trace}


def update_memory(state: AgentState) -> Dict[str, Any]:
    trace = list(state.get("trace", [])) + ["update_memory"]
    decision = state.get("decision")
    intent = decision.intent if decision else "paper_question"

    kind_map = {
        "paper_question": "answer",
        "summarize": "summary",
        "notes": "notes",
        "outline": "outline",
        "rephrase_previous": "rephrase",
        "off_topic": "off_topic",
        "needs_clarification": "clarify",
    }
    output_kind = kind_map.get(intent, "answer")
    result_text = state.get("result_text", "")

    new_last_output = {
        "kind": output_kind,
        "text": result_text,
    }

    history = list(state.get("history", []))
    history.append({"role": "user", "content": state["user_message"]})
    history.append({"role": "assistant", "content": result_text})

    return {
        "last_output": new_last_output,
        "history": history,
        "trace": trace,
    }


# ==========================================
# Conditional Edge Functions
# ==========================================

def route_decision_edge(state: AgentState) -> str:
    decision = state.get("decision")
    intent = decision.intent if decision else "paper_question"
    intent_to_node = {
        "paper_question": "qa_retrieve",
        "summarize": "summarize_node",
        "notes": "notes_node",
        "outline": "outline_node",
        "rephrase_previous": "rephrase_node",
        "off_topic": "offtopic_node",
        "needs_clarification": "clarify_node",
    }
    return intent_to_node.get(intent, "qa_retrieve")


def qa_retrieve_edge(state: AgentState) -> str:
    chunks = state.get("chunks", [])
    if chunks:
        return "qa_answer"
    retries = state.get("retries", 0)
    if retries < MAX_QA_RETRIES:
        return "rewrite_query"
    return "qa_refuse"


def qa_answer_edge(state: AgentState) -> str:
    qa_res = state.get("qa_result")
    retries = state.get("retries", 0)
    is_ref = qa_res.get("refused", False) if qa_res else False

    if is_ref and retries < MAX_QA_RETRIES:
        return "rewrite_query"
    return "update_memory"


# ==========================================
# Build & Compile Graph
# ==========================================

def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("qa_retrieve", qa_retrieve)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("qa_answer", qa_answer)
    workflow.add_node("qa_refuse", qa_refuse)
    workflow.add_node("summarize_node", summarize_node)
    workflow.add_node("notes_node", notes_node)
    workflow.add_node("outline_node", outline_node)
    workflow.add_node("rephrase_node", rephrase_node)
    workflow.add_node("offtopic_node", offtopic_node)
    workflow.add_node("clarify_node", clarify_node)
    workflow.add_node("update_memory", update_memory)

    workflow.set_entry_point("router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_decision_edge,
        {
            "qa_retrieve": "qa_retrieve",
            "summarize_node": "summarize_node",
            "notes_node": "notes_node",
            "outline_node": "outline_node",
            "rephrase_node": "rephrase_node",
            "offtopic_node": "offtopic_node",
            "clarify_node": "clarify_node",
        },
    )

    workflow.add_conditional_edges(
        "qa_retrieve",
        qa_retrieve_edge,
        {
            "qa_answer": "qa_answer",
            "rewrite_query": "rewrite_query",
            "qa_refuse": "qa_refuse",
        },
    )

    workflow.add_edge("rewrite_query", "qa_retrieve")

    workflow.add_conditional_edges(
        "qa_answer",
        qa_answer_edge,
        {
            "rewrite_query": "rewrite_query",
            "update_memory": "update_memory",
        },
    )

    workflow.add_edge("qa_refuse", "update_memory")
    workflow.add_edge("summarize_node", "update_memory")
    workflow.add_edge("notes_node", "update_memory")
    workflow.add_edge("outline_node", "update_memory")
    workflow.add_edge("rephrase_node", "update_memory")
    workflow.add_edge("offtopic_node", "update_memory")
    workflow.add_edge("clarify_node", "update_memory")

    workflow.add_edge("update_memory", END)

    return workflow


compiled_app = build_graph().compile()


# ==========================================
# ChatSession
# ==========================================

class ChatSession:
    """Manages conversation state and graph execution across turns for a document."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.history: List[Dict[str, str]] = []
        self.last_output: Dict[str, Any] = {"kind": "none", "text": ""}

    def ask(self, user_message: str) -> TurnResult:
        reset_call_count()
        t0 = time.time()

        initial_state: AgentState = {
            "doc_id": self.doc_id,
            "user_message": user_message,
            "history": list(self.history),
            "last_output": dict(self.last_output),
            "decision": None,
            "chunks": [],
            "answer": "",
            "qa_result": None,
            "retries": 0,
            "trace": [],
            "result_text": "",
        }

        final_state = compiled_app.invoke(initial_state)

        # Update persistent session state
        self.history = final_state.get("history", self.history)
        self.last_output = final_state.get("last_output", self.last_output)

        elapsed = time.time() - t0
        calls = get_call_count()
        decision = final_state.get("decision")
        intent = decision.intent if decision else "paper_question"
        standalone_q = decision.standalone_query if decision else user_message
        answer = final_state.get("result_text", "")
        trace = final_state.get("trace", [])

        return TurnResult(
            answer=answer,
            intent=intent,
            standalone_query=standalone_q,
            trace=trace,
            llm_calls=calls,
            seconds=elapsed,
        )
