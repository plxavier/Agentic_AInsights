# streamlit_agent_mobile.py - mobile friendly
import streamlit as st
import requests
import re
import json
import os
from datetime import datetime
import uuid

# Page config
st.set_page_config(
    page_title="AInsights Research Assistant",
    page_icon="🤖",
    layout="wide"
)

AGENT_API_URL = "http://localhost:8001"
RESULTS_FILE = "research_results.json"

st.title("🤖 AInsights Research Assistant")
st.caption("Powered by arXiv + LangChain + LangGraph + Langfuse")


# ============================================================
# Helper Functions
# ============================================================

def save_result(query_id, result):
    """Save result to file"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
        else:
            data = {}

        data[query_id] = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'query': result.get('query', '')
        }

        with open(RESULTS_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        st.error(f"Error saving: {e}")
        return False


def load_result(query_id):
    """Load result from file"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
            return data.get(query_id, {}).get('result')
    except:
        pass
    return None


def get_all_results():
    """Get all saved results"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}


def delete_result(query_id):
    """Delete a result"""
    try:
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r') as f:
                data = json.load(f)
            if query_id in data:
                del data[query_id]
                with open(RESULTS_FILE, 'w') as f:
                    json.dump(data, f)
                return True
    except:
        pass
    return False


def fix_latex(text):
    if not text:
        return text
    replacements = {
        '∑': '\\sum', '∫': '\\int', '∇': '\\nabla', '∂': '\\partial',
        '∞': '\\infty', '→': '\\rightarrow', 'α': '\\alpha', 'β': '\\beta',
        'γ': '\\gamma', 'θ': '\\theta', 'λ': '\\lambda', 'μ': '\\mu',
        'σ': '\\sigma', 'φ': '\\phi', 'ψ': '\\psi', 'ω': '\\omega'
    }
    for u, l in replacements.items():
        text = text.replace(u, l)
    text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text, flags=re.DOTALL)
    return text


def run_research(query):
    try:
        response = requests.post(
            f"{AGENT_API_URL}/agent/research",
            json={"query": query},
            timeout=600
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# Check API health
try:
    health = requests.get(f"{AGENT_API_URL}/agent/health", timeout=2)
    api_ok = health.status_code == 200
except:
    api_ok = False

# Initialize session state
if 'current_query_id' not in st.session_state:
    st.session_state.current_query_id = None
if 'viewing_result_id' not in st.session_state:
    st.session_state.viewing_result_id = None
if 'last_displayed_result' not in st.session_state:
    st.session_state.last_displayed_result = None

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("📡 Status")
    if api_ok:
        st.success("✅ Agent API Connected")
    else:
        st.error("❌ Agent API Not Running")

    st.markdown("---")
    st.header("📚 Saved Results")

    saved_results = get_all_results()
    if saved_results:
        sorted_results = sorted(
            saved_results.items(),
            key=lambda x: x[1].get('timestamp', ''),
            reverse=True
        )

        for qid, data in sorted_results[:10]:
            query_preview = data.get('query', 'Unknown')[:40]
            timestamp = data.get('timestamp', '')[:16]

            if st.button(
                    f"📄 {query_preview}...\n{timestamp}",
                    key=f"view_{qid}",
                    use_container_width=True
            ):
                st.session_state.viewing_result_id = qid
                st.rerun()
    else:
        st.info("No saved results yet")

    st.markdown("---")
    st.header("🔗 View Token Data")
    st.markdown("[Open Langfuse Dashboard](https://cloud.langfuse.com)")

    st.markdown("---")
    if st.button("🗑️ Clear All Results", use_container_width=True):
        if os.path.exists(RESULTS_FILE):
            os.remove(RESULTS_FILE)
        st.session_state.viewing_result_id = None
        st.session_state.current_query_id = None
        st.session_state.last_displayed_result = None
        st.rerun()

    st.caption(f"Session: {datetime.now().strftime('%H:%M:%S')}")

# ============================================================
# MAIN INPUT
# ============================================================
st.markdown("### 🔍 Research Question")
query = st.text_area(
    "",
    placeholder="e.g., Explain diffusion model in protein design",
    height=80,
    label_visibility="collapsed"
)

# ============================================================
# RESEARCH BUTTON
# ============================================================
if st.button("🔬 Research", type="primary", use_container_width=True):
    if query:
        if not api_ok:
            st.error("❌ Agent API not running.")
        else:
            with st.spinner("🔍 Researching... (1-2 minutes)"):
                result = run_research(query)

                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    query_id = str(uuid.uuid4())[:8]
                    result['query'] = query
                    result['query_id'] = query_id

                    if save_result(query_id, result):
                        st.success("✅ Research completed and saved!")
                        st.info(f"📌 Result ID: {query_id}")

                        # Store for auto-display
                        st.session_state.current_query_id = query_id
                        st.session_state.viewing_result_id = query_id
                        st.session_state.last_displayed_result = result

                        # FULL DEBUG JSON - like before
                        with st.expander("🔧 Debug: Full Result JSON", expanded=False):
                            st.json(result)

                        # Also show the result immediately
                    else:
                        st.error("Failed to save result")
    else:
        st.warning("Enter a question.")

# ============================================================
# DISPLAY RESULT (Auto-display after research)
# ============================================================

# Check for result to display
result_id = st.session_state.viewing_result_id or st.session_state.current_query_id
display_result = st.session_state.get('last_displayed_result', None)

if result_id and not display_result:
    display_result = load_result(result_id)

if display_result:
    result = display_result

    # Trace link
    trace_id = result.get('trace_id')
    if trace_id:
        st.info(f"🔗 **[View token usage in Langfuse](https://cloud.langfuse.com/trace/{trace_id})**")

    st.caption(f"📌 Result ID: {result_id}")

    # Research Metrics
    st.markdown("---")
    st.markdown("## 📈 Research Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📚 Papers", result.get('papers_used', 0))
    with col2:
        st.metric("❓ Questions", len(result.get('answers', [])))
    with col3:
        st.metric("📖 Citations", len(result.get('citations', [])))
    with col4:
        st.metric("⏱️ Duration", f"{result.get('duration_seconds', 0):.1f}s")

    # Research Synthesis
    st.markdown("---")
    st.markdown("## 📝 Research Synthesis")
    synthesis = result.get('synthesis', '')
    if synthesis:
        st.markdown(fix_latex(synthesis))
    else:
        st.warning("No synthesis generated")

    # Detailed Q&A
    answers = result.get('answers', [])
    if answers:
        with st.expander(f"📋 Detailed Q&A ({len(answers)} questions)", expanded=False):
            for i, ans in enumerate(answers, 1):
                question = ans.get('question', f'Q{i}')
                answer = ans.get('answer', '')
                st.markdown(f"**Q{i}:** {question[:200]}{'...' if len(question) > 200 else ''}")
                st.markdown(fix_latex(answer))
                st.markdown("---")

    # Citations
    citations = result.get('citations', [])
    if citations:
        st.markdown("---")
        st.markdown(f"## 📚 Sources ({len(citations)})")
        for i, citation in enumerate(citations[:15], 1):
            if isinstance(citation, dict):
                title = citation.get('title', 'Unknown')
                arxiv_id = citation.get('arxiv_id', '')
                st.markdown(f"**{i}. {title}**")
                if arxiv_id:
                    st.markdown(f"   📄 arXiv: `{arxiv_id}`")
                st.markdown("")
            else:
                st.markdown(f"- {citation}")

    # FULL DEBUG JSON at bottom (like before)
    with st.expander("🔧 Debug: Full Result JSON", expanded=False):
        st.json(result)

    # Delete button
    if st.button("🗑️ Delete This Result", use_container_width=True):
        if delete_result(result_id):
            if st.session_state.viewing_result_id == result_id:
                st.session_state.viewing_result_id = None
            if st.session_state.current_query_id == result_id:
                st.session_state.current_query_id = None
            st.session_state.last_displayed_result = None
            st.rerun()

elif result_id:
    st.warning(f"Result {result_id} not found.")
    st.session_state.viewing_result_id = None
    st.session_state.current_query_id = None
else:
    st.info("💡 Enter a question above and click 'Research' to get started.")
    st.caption("Results are saved and can be viewed on any device using the same link.")

st.markdown("---")
st.caption("🔬 AInsights | Results saved locally on server | Shareable across devices")