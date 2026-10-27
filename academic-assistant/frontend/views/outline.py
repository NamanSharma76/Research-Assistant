"""Outline view: customizable slide count, presentation preview, and Markdown/text downloads."""
import streamlit as st
from typing import Dict, Any, List
import api_client
from components import (
    render_citations_to_html,
    outline_to_markdown,
    outline_to_text,
    render_error_message,
)


def render_outline_view(token: str, doc_id: str):
    """Renders presentation deck generator and slide preview."""
    col_ctrl1, col_ctrl2, _ = st.columns([2, 1, 3])
    slide_count = col_ctrl1.slider("Slide Count", min_value=5, max_value=15, value=8, step=1)
    generate_clicked = col_ctrl2.button("📊 Generate Outline", use_container_width=True)

    cache_key = f"outline_{doc_id}_{slide_count}"

    if generate_clicked:
        try:
            with st.spinner("Generating presentation slides... this can take up to a minute"):
                outline = api_client.get_outline(token, doc_id, slides=slide_count, force=True)
                st.session_state[cache_key] = outline
        except api_client.ApiError as e:
            st.error(render_error_message(e))
            return
        except Exception as e:
            st.error(f"Failed to generate outline: {e}")
            return

    outline_data = st.session_state.get(cache_key)
    if not outline_data:
        st.info("Select desired slide count and click 'Generate Outline' to create a slide deck.")
        return

    # Deck title & action buttons
    title = outline_data.get("title", "Presentation Outline")
    st.subheader(f"📑 {title}")

    # Build download artifacts
    md_content = outline_to_markdown(outline_data)
    txt_content = outline_to_text(outline_data)

    dl_col1, dl_col2, _ = st.columns([1, 1, 3])
    dl_col1.download_button(
        label="📥 Download Markdown (.md)",
        data=md_content,
        file_name=f"presentation_outline_{slide_count}_slides.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dl_col2.download_button(
        label="📄 Download Plain Text (.txt)",
        data=txt_content,
        file_name=f"presentation_outline_{slide_count}_slides.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.markdown("---")

    # Render slides sequentially
    slides = outline_data.get("slides", [])
    if isinstance(slides, list):
        for idx, slide in enumerate(slides, start=1):
            if isinstance(slide, dict):
                slide_num = slide.get("slide_number", idx)
                slide_title = slide.get("title", f"Slide {slide_num}")
                source = slide.get("source", "")
                bullets = slide.get("bullets", []) or slide.get("points", [])
            else:
                slide_num = idx
                slide_title = f"Slide {slide_num}"
                source = ""
                bullets = [str(slide)]

            with st.container(border=True):
                st.markdown(f"### Slide {slide_num}: {slide_title}")
                if source:
                    st.markdown(f"**Sources:** {render_citations_to_html(source)}", unsafe_allow_html=True)
                for b in bullets:
                    st.markdown(f"- {b}")
