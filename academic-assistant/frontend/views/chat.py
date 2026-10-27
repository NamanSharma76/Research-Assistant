"""Chat view: message history, interactive chat input, starter buttons, and trace inspector."""
import streamlit as st
from typing import Dict, Any, List
import api_client
from components import render_citations_to_html, format_trace, render_error_message


def render_chat_view(token: str, conv_id: str, show_trace: bool):
    """Renders the conversational interface for the active conversation."""
    # Ensure conversation messages cache in session state
    cache_key = f"msgs_{conv_id}"

    # Load messages if not already in session state or when refreshed
    if cache_key not in st.session_state:
        try:
            with st.spinner("Loading conversation..."):
                messages = api_client.list_messages(token, conv_id)
                st.session_state[cache_key] = messages
        except api_client.ApiError as e:
            st.error(render_error_message(e))
            return
        except Exception as e:
            st.error(f"Failed to load messages: {e}")
            return

    messages: List[Dict[str, Any]] = st.session_state[cache_key]

    # Empty conversation starters
    if not messages:
        st.info("Start a discussion about the paper by typing a question below or selecting a quick starter:")
        col1, col2, col3 = st.columns(3)
        prompt_to_send = None
        if col1.button("📄 Summarize this paper", use_container_width=True):
            prompt_to_send = "Summarize this paper"
        if col2.button("❓ What problem does this paper solve?", use_container_width=True):
            prompt_to_send = "What problem does this paper solve?"
        if col3.button("📝 Make study notes", use_container_width=True):
            prompt_to_send = "Make study notes"

        if prompt_to_send:
            _send_user_message(token, conv_id, prompt_to_send, messages, cache_key)
            st.rerun()

    # Render chronological messages
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        with st.chat_message(role):
            if role == "assistant":
                styled_html = render_citations_to_html(content)
                st.markdown(styled_html, unsafe_allow_html=True)

                if show_trace:
                    intent = msg.get("intent") or "N/A"
                    trace = msg.get("trace")
                    llm_calls = msg.get("llm_calls")
                    sec = msg.get("seconds")
                    trace_path = format_trace(trace)

                    with st.expander("🔍 Agent trace", expanded=False):
                        st.markdown(f"**Intent:** `{intent}`")
                        st.markdown(f"**Execution Path:** `{trace_path}`")
                        sec_str = f"{sec:.2f} s" if isinstance(sec, (int, float)) else "N/A"
                        calls_str = str(llm_calls) if llm_calls is not None else "N/A"
                        st.markdown(f"**LLM Calls:** `{calls_str}` | **Latency:** `{sec_str}`")
            else:
                st.markdown(content)

    # Chat input control
    user_input = st.chat_input("Ask a question about the paper, request notes, or an outline...")
    if user_input:
        if st.session_state.get("chat_busy", False):
            st.warning("A message is already being processed. Please wait for it to complete.")
            return

        _send_user_message(token, conv_id, user_input, messages, cache_key)
        st.rerun()


def _send_user_message(token: str, conv_id: str, content: str, messages: List[Dict[str, Any]], cache_key: str):
    """Sends a user message, displays temporary state, and updates session cache."""
    # Append optimistic user message to display immediately
    optimistic_user_msg = {
        "role": "user",
        "content": content,
    }
    messages.append(optimistic_user_msg)
    st.session_state[cache_key] = messages

    st.session_state["chat_busy"] = True
    try:
        with st.spinner("Thinking... this can take up to a minute when the AI provider is busy"):
            assistant_res = api_client.send_message(token, conv_id, content)
            messages.append(assistant_res)
            st.session_state[cache_key] = messages
    except api_client.ApiError as e:
        st.error(render_error_message(e))
        # Keep optimistic user message or allow user to retry
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    finally:
        st.session_state["chat_busy"] = False
