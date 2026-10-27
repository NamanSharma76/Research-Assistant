"""Main Streamlit application entry point for the Academic Research & Learning Assistant."""
import os
import sys
from pathlib import Path
import streamlit as st
from typing import List, Dict, Any, Optional

# Ensure frontend directory is at head of sys.path across cloud deployment environments
frontend_dir = Path(__file__).resolve().parent
if str(frontend_dir) not in sys.path:
    sys.path.insert(0, str(frontend_dir))

import api_client
from components import render_status_badge, render_error_message
from views.chat import render_chat_view
from views.notes import render_notes_view
from views.outline import render_outline_view
from views.summary import render_summary_view

# Configure page metadata
st.set_page_config(
    page_title="Academic Research & Learning Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def handle_logout():
    """Clear all session state and redirect to login."""
    st.session_state.clear()
    st.rerun()


def handle_api_error(err: api_client.ApiError):
    """Handles global API errors such as expired JWTs."""
    if err.status_code == 401:
        st.session_state.clear()
        st.session_state["auth_error"] = "Session expired or invalid. Please log in again."
        st.rerun()
    st.error(render_error_message(err))


def render_auth_gate():
    """Renders login and registration tabs when unauthenticated."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📚 Academic Research Assistant")
        st.caption("Agentic AI-powered research, grounded Q&A, and study synthesis.")

        if "auth_error" in st.session_state:
            st.warning(st.session_state.pop("auth_error"))

        tab_login, tab_register = st.tabs(["🔑 Log In", "📝 Register New Account"])

        with tab_login:
            with st.form("login_form"):
                login_email = st.text_input("Email", placeholder="researcher@university.edu").strip()
                login_password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Log In", use_container_width=True)

                if submit_login:
                    if not login_email or not login_password:
                        st.error("Please enter both email and password.")
                    else:
                        try:
                            token_res = api_client.login_user(login_email, login_password)
                            st.session_state["token"] = token_res["access_token"]
                            st.session_state["user_email"] = login_email
                            st.rerun()
                        except api_client.ApiError as e:
                            st.error(render_error_message(e))
                        except Exception as e:
                            st.error(f"Login failed: {e}")

        with tab_register:
            with st.form("register_form"):
                reg_email = st.text_input("Email", placeholder="researcher@university.edu").strip()
                reg_password = st.text_input("Password", type="password", help="Must be between 8 and 72 bytes.")
                reg_confirm = st.text_input("Confirm Password", type="password")
                submit_register = st.form_submit_button("Register Account", use_container_width=True)

                if submit_register:
                    if not reg_email or not reg_password:
                        st.error("Email and password are required.")
                    elif len(reg_password.encode("utf-8")) < 8 or len(reg_password.encode("utf-8")) > 72:
                        st.error("Password must be between 8 and 72 bytes long.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            api_client.register_user(reg_email, reg_password)
                            # Auto-login on successful registration
                            token_res = api_client.login_user(reg_email, reg_password)
                            st.session_state["token"] = token_res["access_token"]
                            st.session_state["user_email"] = reg_email
                            st.success("Registration successful! Logging in...")
                            st.rerun()
                        except api_client.ApiError as e:
                            st.error(render_error_message(e))
                        except Exception as e:
                            st.error(f"Registration failed: {e}")


@st.fragment(run_every=3)
def render_processing_monitor(token: str):
    """Isolated fragment to poll document status without disrupting user interaction."""
    try:
        updated_docs = api_client.list_documents(token)
        processing = [d for d in updated_docs if d.get("status") == "processing"]
        if not processing:
            # All done processing; reload page state
            st.rerun()
    except Exception:
        pass


def main():
    token = st.session_state.get("token")
    if not token:
        render_auth_gate()
        return

    # User Header in Sidebar
    st.sidebar.markdown(f"👤 Logged in as: **{st.session_state.get('user_email', 'User')}**")
    if st.sidebar.button("🚪 Log Out", use_container_width=True):
        handle_logout()
    st.sidebar.markdown("---")

    # Fetch Documents
    try:
        documents: List[Dict[str, Any]] = api_client.list_documents(token)
    except api_client.ApiError as e:
        handle_api_error(e)
        return
    except Exception as e:
        st.sidebar.error(f"Failed to load documents: {e}")
        documents = []

    has_processing = any(d.get("status") == "processing" for d in documents)
    if has_processing:
        st.sidebar.info("⚡ Ingestion in progress... auto-refreshing")
        render_processing_monitor(token)

    # Document Upload Section
    st.sidebar.subheader("📄 Upload Paper")
    uploaded_file = st.sidebar.file_uploader("Select PDF file", type=["pdf"], key="pdf_uploader")
    if st.sidebar.button("⬆️ Upload Document", disabled=uploaded_file is None, use_container_width=True):
        if uploaded_file is not None:
            try:
                with st.spinner("Uploading and indexing paper..."):
                    new_doc = api_client.upload_document(
                        token,
                        filename=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                    )
                    st.session_state["selected_doc_id"] = new_doc["id"]
                    st.sidebar.success(f"Uploaded {uploaded_file.name} successfully!")
                    st.rerun()
            except api_client.ApiError as e:
                st.sidebar.error(render_error_message(e))
            except Exception as e:
                st.sidebar.error(f"Upload failed: {e}")

    st.sidebar.markdown("---")

    # Document Selector
    st.sidebar.subheader("📚 Documents")
    selected_doc: Optional[Dict[str, Any]] = None

    if documents:
        doc_map = {d["id"]: d for d in documents}
        doc_options = list(doc_map.keys())

        # Determine default index
        curr_doc_id = st.session_state.get("selected_doc_id")
        if curr_doc_id not in doc_options:
            curr_doc_id = doc_options[0]
            st.session_state["selected_doc_id"] = curr_doc_id

        default_idx = doc_options.index(curr_doc_id)

        def format_doc_option(d_id: str) -> str:
            d = doc_map[d_id]
            badge_icon = "⏳" if d.get("status") == "processing" else ("✅" if d.get("status") == "ready" else "❌")
            return f"{badge_icon} {d.get('filename', 'Unknown')} ({d.get('status', '')})"

        chosen_doc_id = st.sidebar.selectbox(
            "Select Active Paper",
            options=doc_options,
            index=default_idx,
            format_func=format_doc_option,
        )
        st.session_state["selected_doc_id"] = chosen_doc_id
        selected_doc = doc_map.get(chosen_doc_id)

        # Document actions
        with st.sidebar.expander("⚙️ Manage Paper"):
            if selected_doc:
                st.caption(f"Doc ID: `{selected_doc.get('doc_id', '')[:12]}...`")
                st.caption(f"Chunks: {selected_doc.get('num_chunks', 0)}")
                delete_confirmed = st.checkbox("Confirm deletion of this paper", key="confirm_delete_doc")
                if st.button("🗑️ Delete Paper", type="primary", disabled=not delete_confirmed, use_container_width=True):
                    try:
                        api_client.delete_document(token, selected_doc["id"])
                        st.session_state.pop("selected_doc_id", None)
                        st.session_state.pop("selected_conv_id", None)
                        st.sidebar.success("Document deleted.")
                        st.rerun()
                    except api_client.ApiError as e:
                        st.sidebar.error(render_error_message(e))
                    except Exception as e:
                        st.sidebar.error(f"Failed to delete: {e}")
    else:
        st.sidebar.caption("No documents uploaded yet.")

    # Conversations Section (active only if a document is selected and ready)
    selected_conv_id: Optional[str] = None
    if selected_doc and selected_doc.get("status") == "ready":
        st.sidebar.markdown("---")
        st.sidebar.subheader("💬 Conversations")

        try:
            conversations = api_client.list_conversations(token, selected_doc["id"])
        except api_client.ApiError as e:
            handle_api_error(e)
            return
        except Exception as e:
            st.sidebar.error(f"Failed to load conversations: {e}")
            conversations = []

        if st.sidebar.button("➕ New Chat Session", use_container_width=True):
            try:
                new_conv = api_client.create_conversation(token, selected_doc["id"])
                st.session_state["selected_conv_id"] = new_conv["id"]
                st.rerun()
            except api_client.ApiError as e:
                st.sidebar.error(render_error_message(e))
            except Exception as e:
                st.sidebar.error(f"Could not create chat: {e}")

        if conversations:
            conv_map = {c["id"]: c for c in conversations}
            conv_options = list(conv_map.keys())

            curr_conv_id = st.session_state.get("selected_conv_id")
            if curr_conv_id not in conv_options:
                curr_conv_id = conv_options[0]
                st.session_state["selected_conv_id"] = curr_conv_id

            conv_idx = conv_options.index(curr_conv_id)
            selected_conv_id = st.sidebar.selectbox(
                "Select Chat",
                options=conv_options,
                index=conv_idx,
                format_func=lambda cid: conv_map[cid].get("title") or "Chat Session",
            )
            st.session_state["selected_conv_id"] = selected_conv_id

            with st.sidebar.expander("🗑️ Delete Chat"):
                confirm_del_chat = st.checkbox("Confirm delete chat", key="confirm_delete_chat")
                if st.button("Delete Chat", type="primary", disabled=not confirm_del_chat, use_container_width=True):
                    try:
                        api_client.delete_conversation(token, selected_conv_id)
                        st.session_state.pop("selected_conv_id", None)
                        st.session_state.pop(f"msgs_{selected_conv_id}", None)
                        st.rerun()
                    except api_client.ApiError as e:
                        st.sidebar.error(render_error_message(e))
                    except Exception as e:
                        st.sidebar.error(f"Failed to delete chat: {e}")
        else:
            # Automatically create first conversation if none exist
            try:
                auto_conv = api_client.create_conversation(token, selected_doc["id"])
                st.session_state["selected_conv_id"] = auto_conv["id"]
                selected_conv_id = auto_conv["id"]
                st.rerun()
            except Exception:
                pass

    # Settings Section in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Settings")
    st.sidebar.toggle(
        "Show Agent Trace",
        value=st.session_state.get("show_trace", False),
        key="show_trace",
        help="Inspect LangGraph node execution flow, LLM call counts, and response latency.",
    )

    # ==========================================
    # Main Content Area
    # ==========================================
    if not selected_doc:
        st.title("📚 Welcome to the Academic Research Assistant")
        st.markdown(
            """
            This platform uses **Agentic AI** to synthesize, explore, and analyze complex scientific literature.
            
            ### Quick Start
            1. **Upload a Paper:** Use the sidebar to upload a scientific PDF (e.g. *Attention Is All You Need*).
            2. **Processing:** The system will parse sections, calculate token budgets, and build semantic embeddings.
            3. **Synthesize & Explore:**
               - **💬 Chat:** Ask natural language questions grounded in verified excerpts.
               - **📖 Notes:** Review auto-generated structured summaries, formulas, and revision questions.
               - **📊 Outline:** Generate and download presentation decks in Markdown or plain text.
               - **📄 Summary:** Read executive summaries with per-section deep dives.
            """
        )
        return

    # Check status of selected document
    doc_status = selected_doc.get("status", "unknown")
    doc_filename = selected_doc.get("filename", "Paper")

    st.header(f"📄 {doc_filename}")

    if doc_status == "processing":
        st.info("⏳ This paper is currently being processed (extracting sections and computing embeddings). Please wait a moment...")
        return
    elif doc_status == "failed":
        st.error(f"❌ Ingestion failed: {selected_doc.get('error', 'Unknown error during PDF ingestion')}")
        return
    elif doc_status != "ready":
        st.warning(f"Document status is '{doc_status}'. Content is unavailable.")
        return

    # Document is ready: Render Tabs
    tab_chat, tab_notes, tab_outline, tab_summary = st.tabs(
        ["💬 Chat", "📖 Notes", "📊 Outline", "📄 Summary"]
    )

    with tab_chat:
        if selected_conv_id:
            render_chat_view(
                token=token,
                conv_id=selected_conv_id,
                show_trace=st.session_state.get("show_trace", False),
            )
        else:
            st.info("Creating a chat session...")

    with tab_notes:
        render_notes_view(token=token, doc_id=selected_doc["id"])

    with tab_outline:
        render_outline_view(token=token, doc_id=selected_doc["id"])

    with tab_summary:
        render_summary_view(token=token, doc_id=selected_doc["id"])


if __name__ == "__main__":
    main()
