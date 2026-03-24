import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
from streamlit_option_menu import option_menu
import os
import sys

# Page configuration
st.set_page_config(
    page_title="AInsights  - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 3.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin-bottom: 1rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 10px;
        font-weight: 600;
    }
    .source-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        color: #1a1a2e;
    }
    .source-box strong {
        color: #1a1a2e;
    }
    .source-box small {
        color: #4a5568;
    }
    .source-box a {
        color: #667eea;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatMessage {
        background-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ========== INITIALIZE SESSION STATE ==========
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []
if 'database_stats' not in st.session_state:
    st.session_state.database_stats = {}
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'last_api_check' not in st.session_state:
    st.session_state.last_api_check = 0

# API configuration
API_BASE_URL = "http://localhost:8001"


def check_api_connection():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if response.status_code == 200:
            st.session_state.api_connected = True
            st.session_state.database_stats = response.json()
            st.session_state.last_api_check = time.time()
            return True
        else:
            st.session_state.api_connected = False
            return False
    except:
        st.session_state.api_connected = False
        return False


def get_papers_list():
    """Get list of papers from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/papers", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"count": 0, "papers": []}
    except:
        return {"count": 0, "papers": []}


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="color: #667eea; margin-bottom: 0;">🔬</h1>
        <h2 style="color: #667eea; margin-top: 0;">AInsights</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # API Status
    current_time = time.time()
    if current_time - st.session_state.last_api_check > 5:
        check_api_connection()

    st.subheader("📡 API Status")

    if st.session_state.api_connected:
        st.success("✅ Connected")

        col1, col2 = st.columns(2)
        with col1:
            docs = st.session_state.database_stats.get('database', {}).get('document_count', 0)
            st.metric("Documents", docs)
        with col2:
            papers_data = get_papers_list()
            st.metric("Papers", papers_data.get("count", 0))
    else:
        st.error("❌ Not Connected")
        st.info("Run: `python app.py`")

    st.markdown("---")

    # Document Management
    st.subheader("📚 Document Management")

    uploaded_file = st.file_uploader(
        "Upload PDF Paper",
        type=['pdf'],
        help="Upload academic papers in PDF format"
    )

    if uploaded_file and st.button("Process PDF", use_container_width=True):
        with st.spinner("Uploading..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=30)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ {result.get('message', 'Uploaded')}")
                    st.session_state.uploaded_files.append({
                        "name": uploaded_file.name,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "chunks": result.get('details', {}).get('chunks_added', 0)
                    })
                    check_api_connection()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Upload failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Clear database
    st.markdown("---")
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        st.warning("Delete ALL papers?")
        if st.button("⚠️ Confirm", type="primary", use_container_width=True):
            try:
                response = requests.delete(f"{API_BASE_URL}/api/clear", timeout=10)
                if response.status_code == 200:
                    st.success("Database cleared!")
                    st.session_state.uploaded_files = []
                    st.session_state.chat_history = []
                    time.sleep(1)
                    st.rerun()
            except:
                st.error("Clear failed")

# ========== MAIN CONTENT ==========
st.markdown('<h1 class="main-title">AInsights AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced Research Assistant with RAG Technology</p>', unsafe_allow_html=True)

# Navigation
selected = option_menu(
    menu_title=None,
    options=["💬 Research Chat", "🔗 Find Connections", "📊 Dashboard"],
    icons=["chat", "lightbulb", "bar-chart"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px"},
        "nav-link-selected": {"background-color": "#667eea"},
    }
)

# ========== TAB 1: RESEARCH CHAT - SIMPLE WORKING VERSION ==========
if selected == "💬 Research Chat":

    # Display all previous messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input - SIMPLE AND WORKING
    if prompt := st.chat_input("Ask a research question..."):
        # Check API connection
        if not check_api_connection():
            st.error("❌ API server not connected.")
            st.stop()

        # Add user message to history and display it
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/api/ask",
                        json={"question": prompt, "top_k": 5},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            answer = result["answer"]
                            st.markdown(answer)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })

                            # Show sources if available
                            if result.get("sources"):
                                with st.expander(f"📚 Sources ({result.get('papers_used', 0)} papers)"):
                                    for source in result.get("sources", []):
                                        st.markdown(f"""
                                        <div class="source-box">
                                            <strong>📄 {source.get('title', 'Untitled')}</strong><br>
                                            <small>👤 Author: {source.get('author', 'Unknown')}</small><br>
                                            <small>📁 Source: {source.get('source', 'Unknown')}</small>
                                        </div>
                                        """, unsafe_allow_html=True)
                        else:
                            error_msg = result.get("error", "Unknown error")
                            st.error(f"Error: {error_msg}")
                    else:
                        st.error(f"API error {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ========== TAB 2: FIND CONNECTIONS ==========
elif selected == "🔗 Find Connections":
    st.header("🔗 Discover Research Connections")

    col1, col2 = st.columns([2, 1])

    with col1:
        new_idea = st.text_area(
            "Enter a new idea or concept:",
            height=150,
            placeholder="e.g., 'Applying machine learning to climate change prediction'"
        )

        depth = st.slider("Analysis depth", 3, 10, 5)

        if st.button("✨ Find Connections", type="primary"):
            if new_idea.strip():
                with st.spinner("Analyzing..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/api/connections",
                            json={"idea": new_idea, "depth": depth},
                            timeout=30
                        )

                        if response.status_code == 200:
                            result = response.json()
                            if result.get("success"):
                                st.success(f"✅ Analyzed {result.get('papers_analyzed', 0)} papers")
                                st.markdown("---")
                                st.markdown("### 📊 Connection Analysis")
                                st.markdown(result.get("analysis", "No analysis available"))
                            else:
                                st.error(result.get("error", "Analysis failed"))
                        else:
                            st.error("Failed to analyze connections")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter an idea")

    with col2:
        st.subheader("💡 How It Works")
        st.markdown("""
        1. Enter a research idea
        2. AI searches your papers
        3. Finds connections
        4. Provides insights
        """)

# ========== TAB 3: DASHBOARD ==========
elif selected == "📊 Dashboard":
    st.header("📊 AInsights Dashboard")

    if not check_api_connection():
        st.warning("Connect to API to see dashboard")
    else:
        papers_data = get_papers_list()

        # Metrics - FIXED: Removed 'key' parameter
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Papers", papers_data.get("count", 0))
        with col2:
            docs = st.session_state.database_stats.get('database', {}).get('document_count', 0)
            st.metric("Documents", docs)
        with col3:
            st.metric("Chat History", len(st.session_state.chat_history))

        st.markdown("---")

        # Papers list
        if papers_data.get("papers"):
            st.subheader("📚 Papers in Database")
            df = pd.DataFrame(papers_data["papers"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No papers in database yet")

# ========== FOOTER ==========
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
with col2:
    status = "✅" if st.session_state.api_connected else "❌"
    st.caption(f"{status} API")
with col3:
    st.caption("AInsights v1.0")