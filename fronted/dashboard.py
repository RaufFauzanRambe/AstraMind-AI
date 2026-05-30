"""
AstraMind AI - Dashboard
==========================
Streamlit-based monitoring dashboard for the AstraMind AI system.
Displays engine status, memory stats, and system health.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE_URL = os.getenv("ASTRAMIND_API_URL", "http://localhost:8000/api/v1")


def fetch_status():
    """Fetch engine status from the API."""
    import requests

    try:
        response = requests.get(f"{API_BASE_URL}/system/status", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "initialized": False}


def fetch_health():
    """Fetch health check from the API."""
    import requests

    try:
        response = requests.get(f"{API_BASE_URL}/system/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def render_header():
    """Render the dashboard header."""
    st.title("📊 AstraMind AI Dashboard")
    st.caption("Real-time monitoring dan system overview")


def render_status_overview():
    """Render the system status overview section."""
    st.subheader("System Status")

    status = fetch_status()

    if status.get("error"):
        st.error(f"⚠️ Error menghubungi server: {status['error']}")
        return

    # Status metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        init_status = "✅ Active" if status.get("initialized") else "❌ Inactive"
        st.metric(label="Engine", value=init_status)

    with col2:
        agent_count = len(status.get("registered_agents", []))
        st.metric(label="Agents", value=agent_count)

    with col3:
        tool_count = len(status.get("registered_tools", []))
        st.metric(label="Tools", value=tool_count)

    with col4:
        uptime = status.get("uptime")
        if uptime:
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            st.metric(label="Uptime", value=f"{hours}h {minutes}m")
        else:
            st.metric(label="Uptime", value="N/A")


def render_memory_stats():
    """Render the memory statistics section."""
    st.subheader("💾 Memory Statistics")

    status = fetch_status()

    if status.get("error"):
        st.warning("Tidak dapat mengambil statistik memory.")
        return

    memory_stats = status.get("memory_stats", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Short-Term Memory",
            value=memory_stats.get("short_term_size", 0),
        )

    with col2:
        st.metric(
            label="Long-Term Memory",
            value=memory_stats.get("long_term_size", 0),
        )

    with col3:
        st.metric(
            label="Active Conversations",
            value=memory_stats.get("active_conversations", 0),
        )


def render_agents_section():
    """Render the registered agents section."""
    st.subheader("🤖 Registered Agents")

    status = fetch_status()

    if status.get("error"):
        st.warning("Tidak dapat mengambil daftar agent.")
        return

    agents = status.get("registered_agents", [])

    if not agents:
        st.info("Tidak ada agent yang terdaftar.")
        return

    agent_descriptions = {
        "astra_core": "Agent utama untuk interaksi umum dan pertanyaan",
        "decision": "Agent khusus untuk pengambilan keputusan dan evaluasi",
        "planner": "Agent khusus untuk perencanaan dan eksekusi multi-langkah",
    }

    for agent in agents:
        desc = agent_descriptions.get(agent, "Agent terdaftar")
        with st.expander(f"🔧 {agent}"):
            st.markdown(f"**Deskripsi:** {desc}")


def render_tools_section():
    """Render the registered tools section."""
    st.subheader("🔧 Available Tools")

    status = fetch_status()

    if status.get("error"):
        st.warning("Tidak dapat mengambil daftar tools.")
        return

    tools = status.get("registered_tools", [])

    if not tools:
        st.info("Tidak ada tools yang terdaftar.")
        return

    tool_descriptions = {
        "calculator": "🔢 Kalkulator - Perhitungan matematika dan aritmatika",
        "web_search": "🔍 Web Search - Pencarian informasi di internet",
        "file_reader": "📂 File Reader - Membaca dan memparse berbagai format file",
        "data_analyzer": "📊 Data Analyzer - Analisis data dan komputasi statistik",
    }

    cols = st.columns(2)
    for i, tool in enumerate(tools):
        with cols[i % 2]:
            desc = tool_descriptions.get(tool, f"Tool: {tool}")
            st.info(desc)


def render_health_section():
    """Render the health check section."""
    st.subheader("❤️ Health Check")

    is_healthy = fetch_health()

    if is_healthy:
        st.success("🟢 Server AstraMind AI berjalan dengan baik!")
    else:
        st.error("🔴 Server AstraMind AI tidak dapat dijangkau. Pastikan server berjalan.")


def main():
    """Main entry point for the dashboard."""
    st.set_page_config(
        page_title="AstraMind AI Dashboard",
        page_icon="📊",
        layout="wide",
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10) if auto_refresh else 0

    render_header()
    st.divider()

    render_status_overview()
    st.divider()

    render_memory_stats()
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        render_agents_section()
    with col2:
        render_tools_section()

    st.divider()
    render_health_section()

    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
