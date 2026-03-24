# streamlit_arxiv.py - FULL VERSION with Dark Theme + Light Answer Boxes
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import json
import time

# Page config
st.set_page_config(
    page_title="AInsights arxiv",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8000"

# Custom CSS - Dark Theme with Light Answer Boxes
st.markdown("""
<style>
    /* Main header - keep gradient */
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(90deg, #1E88E5 0%, #0D47A1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    /* Paper card - keep dark theme */
    .paper-card {
        background: #2d2d2d;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 6px solid #1E88E5;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        color: #e0e0e0;
    }

    .paper-card h3 {
        color: #90caf9;
        margin-bottom: 1rem;
    }

    .paper-card p {
        color: #e0e0e0;
        margin-bottom: 0.5rem;
    }

    .paper-card strong {
        color: #90caf9;
    }

    .paper-card a {
        color: #64b5f6;
    }

    /* Answer box - LIGHT THEME for readability */
    .answer-box {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 2px solid #1E88E5;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        color: #000000;
        font-size: 1.1rem;
        line-height: 1.8;
    }

    /* Answer headings - keep blue but on white */
    .answer-box h1 {
        color: #0D47A1;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #1E88E5;
        padding-bottom: 0.5rem;
    }

    .answer-box h2 {
        color: #1E88E5;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .answer-box h3 {
        color: #1565C0;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .answer-box h4 {
        color: #0D47A1;
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Answer paragraphs - black on white */
    .answer-box p {
        color: #000000;
        margin-bottom: 1.2rem;
        font-size: 1.1rem;
    }

    /* Answer lists - black on white */
    .answer-box ul, .answer-box ol {
        color: #000000;
        margin-left: 2rem;
        margin-bottom: 1.5rem;
    }

    .answer-box li {
        color: #000000;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    /* Answer strong/bold text */
    .answer-box strong {
        color: #0D47A1;
        font-weight: 700;
    }

    /* Answer emphasis text */
    .answer-box em {
        color: #2c3e50;
        font-style: italic;
    }

    /* Answer code blocks */
    .answer-box code {
        background: #f0f0f0;
        color: #d32f2f;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-family: monospace;
        font-size: 0.9rem;
    }

    /* Answer blockquotes */
    .answer-box blockquote {
        background: #f5f5f5;
        border-left: 4px solid #1E88E5;
        padding: 1rem;
        margin: 1rem 0;
        color: #333333;
        font-style: italic;
    }

    /* Source box - keep dark theme */
    .source-box {
        background: #2d2d2d;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        border-left: 4px solid #FFA000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #e0e0e0;
    }

    .source-box strong {
        color: #90caf9;
        font-weight: 600;
    }

    .source-box small {
        color: #b0b0b0;
        font-size: 0.9rem;
    }

    /* Button styling - keep dark theme */
    .stButton > button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }

    .stButton > button:hover {
        background-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Success message - keep dark theme */
    .success-msg {
        padding: 1.2rem;
        background-color: #1e3a2a;
        border: 2px solid #2e7d32;
        color: #a5d6a5;
        border-radius: 8px;
        margin: 1rem 0;
        font-weight: 500;
    }

    .success-msg strong {
        color: #90caf9;
    }

    /* Info box - keep dark theme */
    .info-box {
        background: #1a3a4a;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #0288d1;
        color: #b0e0ff;
        font-weight: 500;
    }

    /* Chunk preview - keep dark theme */
    .chunk-preview {
        background: #2d2d2d;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #FF9800;
        font-family: monospace;
        max-height: 250px;
        overflow-y: auto;
        color: #e0e0e0;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* Text input - keep dark theme */
    .stTextInput > div > input {
        color: #e0e0e0;
        background-color: #2d2d2d;
        border: 2px solid #404040;
    }

    /* Text area - keep dark theme */
    .stTextArea > div > textarea {
        color: #e0e0e0;
        background-color: #2d2d2d;
        border: 2px solid #404040;
    }

    /* Select box - keep dark theme */
    .stSelectbox > div > div {
        color: #e0e0e0;
        background-color: #2d2d2d;
    }

    /* Metric cards - keep dark theme */
    .stMetric {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
        color: #e0e0e0;
    }

    .stMetric label {
        color: #b0b0b0 !important;
    }

    .stMetric .metric-value {
        color: #90caf9 !important;
        font-weight: 700 !important;
    }

    /* DataFrame - keep dark theme */
    .stDataFrame {
        color: #e0e0e0;
    }

    .stDataFrame td {
        color: #e0e0e0;
    }

    /* Expander - keep dark theme */
    .streamlit-expanderHeader {
        color: #90caf9;
        font-weight: 600;
        background-color: #2d2d2d;
    }

    .streamlit-expanderContent {
        color: #e0e0e0;
        background-color: #2d2d2d;
        border: 1px solid #404040;
        padding: 1rem;
    }

    /* Tabs - keep dark theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #b0b0b0;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        color: #90caf9 !important;
        font-weight: 700;
    }

    /* Sidebar - keep dark theme */
    .css-1d391kg {
        background-color: #1a1a1a;
    }

    .sidebar-content {
        color: #e0e0e0;
    }

    /* Markdown text - keep dark theme */
    .stMarkdown {
        color: #e0e0e0;
    }

    /* Warning/Error/Info messages - keep dark theme */
    .stAlert {
        color: #e0e0e0;
        font-weight: 500;
    }

    /* Headers in main content - keep light blue */
    h1, h2, h3, h4, h5, h6 {
        color: #90caf9;
    }

    /* Links - keep light blue */
    a {
        color: #64b5f6;
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
        color: #90caf9;
    }

    /* Code blocks - keep dark theme */
    code {
        background-color: #2d2d2d;
        color: #f5f5f5;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
    }

    /* Tables - keep dark theme */
    table {
        color: #e0e0e0;
    }

    th {
        color: #90caf9;
        font-weight: 600;
    }

    td {
        color: #e0e0e0;
    }

    /* Override any light text */
    .st-emotion-cache-10trblm {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'current_answer' not in st.session_state:
    st.session_state.current_answer = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'full_info' not in st.session_state:
    st.session_state.full_info = None
if 'show_confirm' not in st.session_state:
    st.session_state.show_confirm = False


# Check API connection
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        st.session_state.api_connected = response.status_code == 200
        if st.session_state.api_connected:
            try:
                stats = requests.get(f"{API_URL}/api/stats").json()
                st.session_state.papers = stats.get("papers", [])
            except:
                pass
        return st.session_state.api_connected
    except:
        st.session_state.api_connected = False
        return False


# Header
st.markdown('<h1 class="main-header">📚 AInsights arxiv</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://arxiv.org/static/browse/0.3.4/images/arxiv-logo.svg", width=200)

    st.markdown("---")

    # API Status
    st.subheader("📡 API Status")
    if check_api():
        st.success("✅ Connected")

        try:
            response = requests.get(f"{API_URL}/api/stats")
            if response.status_code == 200:
                stats = response.json()
                st.metric("Papers in DB", stats["database"]["total_papers"])
                st.metric("Total Chunks", stats["database"]["total_chunks"])
        except:
            pass
    else:
        st.error("❌ Not Connected")
        st.info("Start API server:")
        st.code("python app_arxiv.py")
        st.stop()

    st.markdown("---")

    # Quick Actions
    st.subheader("⚡ Quick Actions")

    if st.button("📋 Refresh Paper List", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/api/papers")
            if response.status_code == 200:
                st.session_state.papers = response.json().get("papers", [])
                st.success(f"Found {len(st.session_state.papers)} papers")
                st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.button("📊 Database Stats", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/api/stats")
            if response.status_code == 200:
                stats = response.json()
                st.info(f"""
                **Papers:** {stats['database']['total_papers']}
                **Chunks:** {stats['database']['total_chunks']}
                """)
        except:
            pass

    st.markdown("---")

    # Clear database
    if st.button("🗑️ Clear Database", type="secondary", use_container_width=True):
        st.session_state.show_confirm = True

    if st.session_state.get('show_confirm', False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Yes, Clear", type="primary"):
                try:
                    response = requests.post(f"{API_URL}/api/clear")
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("success"):
                            st.success("Database cleared!")
                            st.session_state.papers = []
                            st.session_state.current_paper = None
                            st.session_state.current_answer = None
                            st.session_state.show_confirm = False
                            st.rerun()
                except:
                    st.error("Failed to clear")
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.show_confirm = False
                st.rerun()

    st.markdown("---")
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Search & Add",
    "📚 My Papers",
    "💬 Ask Questions",
    "🔎 Paper Details",
    "ℹ️ About"
])

# ========== TAB 1: Search & Add ==========
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Add Paper by arXiv ID")

        arxiv_id = st.text_input(
            "Enter arXiv ID",
            placeholder="e.g., 2303.08774 or 2511.21476v1",
            help="Enter the arXiv ID (with or without version)",
            key="arxiv_input"
        )

        col_a, col_b, col_c = st.columns(3)
        with col_b:
            if st.button("🔍 Fetch Paper", type="primary", use_container_width=True):
                if arxiv_id:
                    with st.spinner("Fetching from arXiv..."):
                        try:
                            # Fetch metadata
                            response = requests.post(
                                f"{API_URL}/api/fetch",
                                json={"arxiv_id": arxiv_id.strip()}
                            )

                            if response.status_code == 200:
                                paper = response.json()
                                st.session_state.current_paper = paper

                                # Also fetch full content
                                with st.spinner("Downloading full paper..."):
                                    full_response = requests.post(
                                        f"{API_URL}/api/fetch_full",
                                        json={"arxiv_id": arxiv_id.strip()}
                                    )

                                    if full_response.status_code == 200:
                                        full_info = full_response.json()
                                        st.session_state.full_info = full_info
                            else:
                                st.error("Paper not found")

                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Enter an arXiv ID")

        # Display paper preview
        if st.session_state.get('current_paper'):
            paper = st.session_state.current_paper

            st.markdown(f"""
            <div class="paper-card">
                <h3>{paper['title']}</h3>
                <p><strong>Authors:</strong> {', '.join(paper['authors'])}</p>
                <p><strong>Published:</strong> {paper.get('published', 'Unknown')}</p>
                <p><strong>arXiv ID:</strong> {paper['arxiv_id']}</p>
                <p><strong>Summary:</strong> {paper['summary'][:500]}...</p>
                <p><strong>PDF:</strong> <a href="{paper.get('pdf_url', '#')}" target="_blank">Download</a></p>
                <p><strong>HTML:</strong> <a href="{paper.get('html_url', '#')}" target="_blank">View HTML5</a></p>
            </div>
            """, unsafe_allow_html=True)

            # Show full paper info if available
            if st.session_state.get('full_info'):
                info = st.session_state.full_info
                st.info(f"📄 Full paper: {info['chunks']} chunks, {info['source']} source")

            # Add to database button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("➕ Add Full Paper to Database", type="primary", use_container_width=True):
                    with st.spinner("Processing full paper..."):
                        try:
                            response = requests.post(
                                f"{API_URL}/api/add",
                                json={"arxiv_id": paper['arxiv_id']}
                            )

                            if response.status_code == 200:
                                result = response.json()
                                if result.get("success"):
                                    st.markdown(f"""
                                    <div class="success-msg">
                                        ✅ {result['message']}<br>
                                        📊 {result.get('chunks', 0)} chunks created
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.toast(f"✅ Paper added! 📄 {result.get('chunks', 0)} chunks", icon="📚")

                                    # Clear current and refresh
                                    st.session_state.current_paper = None
                                    st.session_state.full_info = None
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error(result.get('message', 'Failed to add'))
                            else:
                                st.error(f"Failed to add: {response.status_code}")

                        except Exception as e:
                            st.error(f"Error: {e}")

    with col2:
        st.subheader("📝 Example Papers")
        examples = [
            {"id": "2511.21476v1", "desc": "Latest AI Paper (HTML available)"},
            {"id": "2303.08774", "desc": "GPT-4 Technical Report"},
            {"id": "2106.09685", "desc": "LLaMA: Open Foundation Models"},
            {"id": "2005.14165", "desc": "GPT-3 Paper"},
            {"id": "1706.03762", "desc": "Attention Is All You Need"}
        ]

        for ex in examples:
            if st.button(f"📄 {ex['desc']}", use_container_width=True):
                st.session_state.arxiv_input = ex['id']
                st.rerun()

        st.markdown("---")
        st.info("""
        **💡 Note:** Papers with HTML versions (like 2511.21476v1) 
        will have better text extraction than PDF-only papers.
        """)

# ========== TAB 2: My Papers ==========
with tab2:
    st.subheader("📚 Papers in Database")

    if st.button("🔄 Refresh List", use_container_width=True):
        try:
            response = requests.get(f"{API_URL}/api/papers")
            if response.status_code == 200:
                st.session_state.papers = response.json().get("papers", [])
                st.rerun()
        except:
            pass

    st.markdown("---")

    if st.session_state.papers:
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Papers", len(st.session_state.papers))

        # Display papers
        for paper in st.session_state.papers:
            with st.expander(f"📄 {paper.get('title', 'Unknown')}"):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**arXiv ID:** {paper.get('arxiv_id', 'N/A')}")
                    st.write(f"**Authors:** {paper.get('authors', 'Unknown')}")
                    st.write(f"**Added:** {paper.get('added_date', 'Unknown')}")
                    st.write(f"**Chunks:** {paper.get('chunks', 0)}")

                with col2:
                    if st.button("🔍 View Details", key=f"view_{paper.get('arxiv_id')}"):
                        st.session_state.view_paper = paper.get('arxiv_id')
                        st.rerun()

                    if st.button("💬 Ask about this", key=f"ask_{paper.get('arxiv_id')}"):
                        st.session_state.ask_paper = paper.get('arxiv_id')
                        st.session_state.chat_mode = "specific"
                        st.rerun()

        # Download option
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("📥 Download as JSON", use_container_width=True):
                json_str = json.dumps(st.session_state.papers, indent=2)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_str,
                    file_name=f"arxiv_papers_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    else:
        st.info("📭 No papers in database. Use 'Search & Add' tab to add papers.")

# ========== TAB 3: Ask Questions ==========
with tab3:
    st.subheader("💬 Ask Questions About Your Papers")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Paper selector
        papers_list = [{"id": "all", "title": "🔍 Search All Papers"}] + [
            {"id": p.get("arxiv_id"), "title": p.get("title", "Unknown")}
            for p in st.session_state.papers
        ]

        selected_paper = st.selectbox(
            "Focus on specific paper (optional):",
            options=[p["id"] for p in papers_list],
            format_func=lambda x: next(
                (f"{p['title'][:50]}..." if len(p['title']) > 50 else p['title']
                 for p in papers_list if p["id"] == x), x
            ),
            key="paper_selector"
        )

        # Question input
        question = st.text_area(
            "Your question:",
            placeholder="e.g., What are the main contributions? What methodology was used?",
            height=100,
            key="question_input"
        )

        # Number of papers to search
        top_k = st.slider("Number of papers to search", 1, 5, 3)

        # Ask button
        if st.button("🔍 Ask Question", type="primary", use_container_width=True):
            if question:
                with st.spinner("Searching papers and formulating answer..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/api/ask",
                            json={
                                "question": question,
                                "arxiv_id": None if selected_paper == "all" else selected_paper,
                                "top_k": top_k
                            }
                        )

                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.current_answer = result

                            # Add to chat history
                            st.session_state.chat_history.append({
                                "timestamp": datetime.now().strftime("%H:%M:%S"),
                                "question": question,
                                "answer": result.get("answer", ""),
                                "sources": result.get("sources", [])
                            })
                        else:
                            st.error(f"Error: {response.status_code}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a question")

    with col2:
        st.markdown("### 💡 Example Questions")
        examples = [
            "What are the main contributions?",
            "What methodology was used?",
            "What are the limitations?",
            "What datasets were used?",
            "How does this compare to previous work?",
            "What future work is suggested?"
        ]

        for ex in examples:
            if st.button(ex, use_container_width=True):
                st.session_state.question_input = ex
                st.rerun()

        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.metric("Papers Available", len(st.session_state.papers))

        if st.session_state.papers:
            total_chunks = sum(p.get('chunks', 0) for p in st.session_state.papers)
            st.metric("Total Chunks", total_chunks)

    # Display current answer
    if st.session_state.get('current_answer'):
        result = st.session_state.current_answer

        st.markdown("---")
        st.markdown("### 📝 Answer:")

        st.markdown(f"""
        <div class="answer-box">
            {result.get('answer', 'No answer generated')}
        </div>
        """, unsafe_allow_html=True)

        if result.get('sources'):
            st.markdown("### 📚 Sources:")
            for source in result['sources']:
                st.markdown(f"""
                <div class="source-box">
                    <strong>📄 {source.get('title', 'Unknown')}</strong><br>
                    <small>arXiv: {source.get('arxiv_id', 'N/A')}</small><br>
                    <small>Authors: {', '.join(source.get('authors', ['Unknown']))}</small>
                </div>
                """, unsafe_allow_html=True)

        st.metric("Papers Used", result.get('papers_used', 0))

    # Chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Chat History")

        for chat in reversed(st.session_state.chat_history[-5:]):
            with st.expander(f"❓ {chat['question'][:50]}... ({chat['timestamp']})"):
                st.markdown(f"**Q:** {chat['question']}")
                st.markdown(f"**A:** {chat['answer'][:200]}...")

# ========== TAB 4: Paper Details ==========
with tab4:
    st.subheader("🔎 Paper Details")

    # Paper selector for details
    if st.session_state.papers:
        paper_options = [p.get('arxiv_id') for p in st.session_state.papers]
        paper_labels = [f"{p.get('title', 'Unknown')[:50]}..." for p in st.session_state.papers]

        selected_detail = st.selectbox(
            "Select paper to view details:",
            options=paper_options,
            format_func=lambda x: next(
                (label for opt, label in zip(paper_options, paper_labels) if opt == x), x
            ),
            key="detail_selector"
        )

        if selected_detail and st.button("🔍 Load Details", type="primary"):
            with st.spinner("Loading paper details..."):
                try:
                    response = requests.get(f"{API_URL}/api/paper/{selected_detail}")

                    if response.status_code == 200:
                        paper = response.json()
                        st.session_state.viewing_paper = paper
                except Exception as e:
                    st.error(f"Error: {e}")

        # Display paper details
        if st.session_state.get('viewing_paper'):
            paper = st.session_state.viewing_paper

            st.markdown(f"""
            <div class="paper-card">
                <h3>{paper.get('title', 'Unknown')}</h3>
                <p><strong>arXiv ID:</strong> {paper.get('arxiv_id', 'N/A')}</p>
                <p><strong>Authors:</strong> {', '.join(paper.get('authors', []))}</p>
                <p><strong>Added:</strong> {paper.get('added_date', 'Unknown')}</p>
                <p><strong>Source Type:</strong> {paper.get('source_type', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)

            # Show chunks
            if paper.get('chunks'):
                st.markdown(f"### 📊 Paper Chunks ({len(paper['chunks'])})")

                for i, chunk in enumerate(paper['chunks'][:5]):  # Show first 5 chunks
                    with st.expander(f"Chunk {i + 1}"):
                        # Handle both dict and string chunks
                        if isinstance(chunk, dict):
                            chunk_text = chunk.get('text', '') or chunk.get('content', '')
                        else:
                            chunk_text = str(chunk)

                        st.markdown(f"""
                        <div class="chunk-preview">
                            {chunk_text[:500]}...
                        </div>
                        """, unsafe_allow_html=True)

                if len(paper['chunks']) > 5:
                    st.info(f"... and {len(paper['chunks']) - 5} more chunks")
    else:
        st.info("No papers in database to view details.")

# ========== TAB 5: About ==========
with tab5:
    st.subheader("ℹ️ AInsights arxiv")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📚 Features

        **Paper Management:**
        - ✅ Fetch papers by arXiv ID
        - ✅ Download full PDF/HTML content
        - ✅ Smart text extraction
        - ✅ Automatic chunking
        - ✅ Persistent storage

        **Q&A Capabilities:**
        - ✅ Ask questions about papers
        - ✅ Search across multiple papers
        - ✅ Contextual answers
        - ✅ Source citations
        - ✅ Chat history
        - ✅ Structured answers with introduction, findings, and conclusion

        **New in v2.0:**
        - ✨ HTML5 support (arXiv beta)
        - ✨ Better text extraction
        - ✨ Full paper processing
        - ✨ Improved chunking
        - ✨ Enhanced answer formatting
        """)

    with col2:
        st.markdown("""
        ### 🔧 Technology Stack

        | Component | Technology |
        |-----------|------------|
        | Backend | FastAPI |
        | Database | ChromaDB |
        | Frontend | Streamlit |
        | PDF Processing | PyPDF2 |
        | HTML Parsing | BeautifulSoup4 |
        | arXiv API | arxiv.py |

        ### 📖 How to Use

        1. **Add papers** - Enter arXiv ID in Search tab
        2. **Ask questions** - Use the Questions tab
        3. **View papers** - Check My Papers tab
        4. **Explore details** - Paper Details tab

        ### 🎨 Visual Design

        • **Dark theme** for comfortable viewing
        • **Light answer boxes** for maximum readability
        • **Blue accents** for important elements
        • **High contrast** where it matters most
        """)

    # System status
    st.markdown("---")
    st.subheader("🔧 System Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.session_state.api_connected:
            st.success("✅ API Server")
        else:
            st.error("❌ API Server")

    with col2:
        if st.session_state.papers:
            st.info(f"📚 Database: {len(st.session_state.papers)} papers")
        else:
            st.warning("📚 Database: Empty")

    with col3:
        st.info(f"💬 Q&A: Ready")

    with col4:
        st.info(f"🕐 {datetime.now().strftime('%H:%M:%S')}")

# Footer
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.caption(f"📚 AInsights arxiv")

with col2:
    if st.session_state.api_connected:
        st.caption("✅ API Connected")
    else:
        st.caption("❌ API Disconnected")

with col3:
    st.caption(f"📊 Papers: {len(st.session_state.papers)}")

with col4:
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")