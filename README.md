# 🤖 Agentic_AInsights: 3-level AI-Powered Agentic Research Assistant

An intelligent research assistant that progressively unlocks capabilities from basic PDF Q&A to advanced agentic literature synthesis.

## 🎯 Overview

AInsights provides three tiers of research assistance, each building on the previous:

| Tier        | Capabilities                             |
|:------------|:-----------------------------------------|
| **Level 1** | PDF upload, basic RAG, ChromaDB storage  |
| **Level 2** | arXiv search, paper retrieval, citations |
| **Level 3** | Agentic AI, LangGraph orchestration, Langfuse observability |

## ✨ Features

- **📄 PDF Upload & RAG** - Upload papers, get instant answers
- **🔍 arXiv Integration** - Search and retrieve papers from arXiv
- **🤖 Agentic Research** - Multi-question synthesis with equations and citations
- **📊 Langfuse Observability** - Token tracking, cost monitoring, latency analytics
- **📱 Cross-Platform** - Access via mobile with ngrok tunneling


## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- (Optional) Langfuse account for observability

### Installation

```bash
# Clone the repository
git clone https://github.com/plxavier/Agentic_AInsights.git
cd Agentic_AInsights

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```
# Choose a level:
python app.py   #Level1
python app_arxiv.py   #Level2
python app_arxiv_agent_with_langfuse.py #Level3

# In another terminal, start the UI:
streamlit run streamlit/app_level_1.py --server.port 8501
or
streamlit run streamlit/app_level2.py
or
streamlit run streamlit/agent_mobile_level3.py
or 
streamlit run langfuse_monitor.py 


# For mobile access:
ngrok http 8501
