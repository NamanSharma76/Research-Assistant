"""Notes view: structured study notes, key concepts, revision Q&A, and citation chips."""
import streamlit as st
from typing import Dict, Any, List
import api_client
from components import render_citations_to_html, render_error_message

KNOWN_SECTION_FIELDS = {
    "problem": "Problem Statement",
    "method": "Methodology & Architecture",
    "dataset_and_setup": "Dataset & Experimental Setup",
    "key_results": "Key Findings & Results",
    "limitations": "Limitations & Constraints",
}


def render_notes_view(token: str, doc_id: str):
    """Renders comprehensive study notes for the active document."""
    cache_key = f"notes_{doc_id}"

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    load_clicked = col_btn1.button("📖 Load Notes", use_container_width=True)
    regen_clicked = col_btn2.button("🔄 Regenerate Notes", use_container_width=True)

    if load_clicked or regen_clicked:
        force = bool(regen_clicked)
        try:
            with st.spinner("Generating study notes... this can take up to a minute"):
                data = api_client.get_notes(token, doc_id, force=force)
                st.session_state[cache_key] = data
        except api_client.ApiError as e:
            st.error(render_error_message(e))
            return
        except Exception as e:
            st.error(f"Failed to retrieve notes: {e}")
            return

    notes_data = st.session_state.get(cache_key)
    if not notes_data:
        st.info("Click 'Load Notes' to view or generate structured revision notes.")
        return

    # Render main note sections
    for field_key, field_label in KNOWN_SECTION_FIELDS.items():
        if field_key in notes_data:
            field_data = notes_data[field_key]
            with st.expander(field_label, expanded=True):
                if isinstance(field_data, dict):
                    text = field_data.get("text", "")
                    sources = field_data.get("sources", [])
                    st.markdown(render_citations_to_html(text), unsafe_allow_html=True)
                    if sources:
                        st.markdown("**Grounding Sources:**")
                        for src in sources:
                            st.markdown(render_citations_to_html(f"- {src}"), unsafe_allow_html=True)
                elif isinstance(field_data, str):
                    st.markdown(render_citations_to_html(field_data), unsafe_allow_html=True)

    # Key Terms
    key_terms: List[Dict[str, Any]] = notes_data.get("key_terms", [])
    if key_terms:
        with st.expander(f"🔑 Key Terms ({len(key_terms)})", expanded=True):
            for t in key_terms:
                term = t.get("term", "")
                definition = t.get("definition", "")
                sources = t.get("sources", [])
                st.markdown(f"**{term}**")
                st.markdown(render_citations_to_html(definition), unsafe_allow_html=True)
                if sources:
                    st.markdown(render_citations_to_html("Sources: " + ", ".join(sources)), unsafe_allow_html=True)
                st.markdown("---")

    # Revision Questions
    questions: List[Dict[str, Any]] = notes_data.get("revision_questions", [])
    if questions:
        with st.expander(f"❓ Revision Questions ({len(questions)})", expanded=True):
            for idx, q in enumerate(questions, start=1):
                question_text = q.get("question", "")
                expected = q.get("expected_answer", "")
                sources = q.get("sources", [])
                st.markdown(f"**Q{idx}: {question_text}**")
                st.markdown(render_citations_to_html(f"*Answer:* {expected}"), unsafe_allow_html=True)
                if sources:
                    st.markdown(render_citations_to_html("Sources: " + ", ".join(sources)), unsafe_allow_html=True)
                st.markdown("---")

    # Render any unrecognized extra keys safely without dropping
    extra_keys = [k for k in notes_data.keys() if k not in KNOWN_SECTION_FIELDS and k not in ("key_terms", "revision_questions")]
    for ek in extra_keys:
        with st.expander(f"📦 Additional Data ({ek})"):
            st.json(notes_data[ek])
