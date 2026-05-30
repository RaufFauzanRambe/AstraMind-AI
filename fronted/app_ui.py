"""
AstraMind AI - Application UI
================================
Streamlit-based chat interface for interacting with AstraMind AI.
Provides a clean, conversational UI with memory context display.
"""

import os
import sys

import requests
import streamlit as st

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = os.getenv("ASTRAMIND_API_URL", "http://localhost:8000/api/v1")


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = None


def send_message(message: str, conversation_id: str = None) -> dict:
    """
    Send a message to the AstraMind API and return the response.

    Args:
        message: The user's message text.
        conversation_id: Optional conversation ID for continuity.

    Returns:
        Dictionary with the API response.
    """
    try:
        payload = {
            "message": message,
            "conversation_id": conversation_id,
        }
        response = requests.post(
            f"{API_BASE_URL}/chat/",
            json=payload,
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {
            "response": "⚠️ Tidak dapat terhubung ke server AstraMind. Pastikan server berjalan.",
            "agent": "system",
            "confidence": 0.0,
        }
    except requests.exceptions.Timeout:
        return {
            "response": "⚠️ Request timeout. Server membutuhkan waktu terlalu lama untuk merespons.",
            "agent": "system",
            "confidence": 0.0,
        }
    except Exception as e:
        return {
            "response": f"⚠️ Error: {str(e)}",
            "agent": "system",
            "confidence": 0.0,
        }


def check_server_health() -> bool:
    """Check if the AstraMind server is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/system/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.title("⚙️ Pengaturan")

        # Server status
        is_healthy = check_server_health()
        status_color = "🟢" if is_healthy else "🔴"
        st.markdown(f"**Server Status:** {status_color} {'Online' if is_healthy else 'Offline'}")

        st.divider()

        # Chat settings
        st.subheader("Chat Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Higher values produce more creative responses.",
        )
        max_tokens = st.slider(
            "Max Tokens",
            min_value=128,
            max_value=4096,
            value=2048,
            step=128,
            help="Maximum number of tokens in the response.",
        )

        st.divider()

        # Conversation controls
        st.subheader("Sesi")
        if st.button("🔄 Sesi Baru", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        if st.button("🗑️ Hapus Memori", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        st.divider()

        # About section
        st.subheader("Tentang")
        st.markdown(
            """
            **AstraMind AI** v0.1.0

            Asisten AI cerdas dengan kemampuan:
            - 🧠 Reasoning & Analysis
            - 💾 Persistent Memory
            - 🔧 Tool Integration
            - 🤖 Multi-Agent System
            """
        )


def render_chat():
    """Render the main chat interface."""
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                meta = message["metadata"]
                cols = st.columns(3)
                with cols[0]:
                    st.caption(f"🤖 Agent: {meta.get('agent', 'unknown')}")
                with cols[1]:
                    st.caption(f"📊 Confidence: {meta.get('confidence', 0):.1%}")
                with cols[2]:
                    if meta.get("tool_results"):
                        st.caption(f"🔧 Tools: {len(meta['tool_results'])}")

    # Chat input
    if prompt := st.chat_input("Ketik pesan Anda..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Berpikir..."):
                response = send_message(
                    message=prompt,
                    conversation_id=st.session_state.conversation_id,
                )

            st.markdown(response.get("response", "Maaf, terjadi kesalahan."))

            # Update conversation ID
            if response.get("conversation_id"):
                st.session_state.conversation_id = response["conversation_id"]

            # Show metadata
            cols = st.columns(3)
            with cols[0]:
                st.caption(f"🤖 Agent: {response.get('agent', 'unknown')}")
            with cols[1]:
                st.caption(f"📊 Confidence: {response.get('confidence', 0):.1%}")
            with cols[2]:
                if response.get("tool_results"):
                    st.caption(f"🔧 Tools: {len(response['tool_results'])}")

            # Store assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": response.get("response", ""),
                "metadata": response,
            })


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="AstraMind AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply custom CSS
    st.markdown(
        """
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()
    render_sidebar()

    # Main title
    st.title("🧠 AstraMind AI")
    st.caption("Asisten AI cerdas dengan reasoning, memory, dan multi-agent capabilities")

    render_chat()


if __name__ == "__main__":
    main()
