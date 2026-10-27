"""Summary view: executive overall summary and per-section breakdowns with citations."""
import streamlit as st
from typing import Dict, Any
import api_client
from components import render_citations_to_html, render_error_message


def render_summary_view(token: str, doc_id: str):
    """Renders executive and per-section summaries for the active document."""
    cache_key = f"summary_{doc_id}"

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    load_clicked = col_btn1.button("📄 Load Summary", use_container_width=True)
    regen_clicked = col_btn2.button("🔄 Regenerate Summary", use_container_width=True)

    if load_clicked or regen_clicked:
        force = bool(regen_clicked)
        try:
            with st.spinner("Generating document summary... this can take up to a minute"):
                data = api_client.get_summary(token, doc_id, force=force)
                st.session_state[cache_key] = data
        except api_client.ApiError as e:
            st.error(render_error_message(e))
            return
        except Exception as e:
            st.error(f"Failed to retrieve summary: {e}")
            return

    summary_data = st.session_state.get(cache_key)
    if not summary_data:
        st.info("Click 'Load Summary' to view or generate an executive summary.")
        return

    overall_text = summary_data.get("overall", "")
    if overall_text:
        st.subheader("Executive Overview")
        st.markdown(render_citations_to_html(overall_text), unsafe_allow_html=True)
        st.markdown("---")

    sections = summary_data.get("sections", {})
    if sections:
        st.subheader("Section Breakdowns")
        if isinstance(sections, dict):
            for sec_name, sec_text in sections.items():
                with st.expander(f"📑 {sec_name}", expanded=False):
                    if isinstance(sec_text, str):
                        st.markdown(render_citations_to_html(sec_text), unsafe_allow_html=True)
                    elif isinstance(sec_text, dict):
                        st.markdown(render_citations_to_html(sec_text.get("summary", str(sec_text))), unsafe_allow_html=True)
                    else:
                        st.write(sec_text)
        elif isinstance(sections, list):
            for item in sections:
                if isinstance(item, dict):
                    sec_name = item.get("section", item.get("title", "Section"))
                    sec_text = item.get("summary", item.get("content", str(item)))
                    with st.expander(f"📑 {sec_name}", expanded=False):
                        st.markdown(render_citations_to_html(sec_text), unsafe_allow_html=True)
                else:
                    st.write(item)
