# !/usr/bin/env python3
"""
AInsights Launcher
Choose which tier to run:
- Level 1: Basic RAG
- Level 2: arXiv Integration
- Level 3: Agentic AI
"""

import sys
import subprocess


def main():
    print("=" * 50)
    print("🤖 AInsights - 3-Tier Research Agentic Assistant")
    print("=" * 50)
    print("1. Level 1 - Basic RAG (PDF upload)")
    print("2. Level 2 - arXiv-API Integration")
    print("3. Level 3 - Agentic AI (LangChain + LangGraph + Langfuse)")
    print("4. Streamlit UI")
    print("5. Observability Monitor Dashboard")
    print("=" * 50)

    choice = input("Select an option (1-5): ").strip()

    if choice == "1":
        subprocess.run(["python", "app.py"])
    elif choice == "2":
        subprocess.run(["python", "app_arxiv.py"])
    elif choice == "3":
        subprocess.run(["python", "app_arxiv_agent_with_langfuse.py"])
    elif choice == "4":
        subprocess.run(["streamlit", "run", "streamlit/app.py", "--server.port", "8501"])
    elif choice == "5":
        subprocess.run(["streamlit", "run", "streamlit/monitor.py", "--server.port", "8502"])
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
