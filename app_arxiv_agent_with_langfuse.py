# ============================================================================
# app_arxiv_agent.py - WITH TOKEN TRACKING AND LANGFUSE INTEGRATION
# ============================================================================

import os
import json
import requests
import time
from typing import List, Dict, Any, TypedDict, Annotated, Literal, Optional
from datetime import datetime
import operator
import re
import traceback

from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver as InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, Request
from dotenv import load_dotenv
import uvicorn



#Langfuse Integration
try:
    from langfuse import Langfuse
    from langfuse.callback import CallbackHandler

    load_dotenv()
    LANGFUSE_AVAILABLE = True

    langfuse_client = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )

    langfuse_handler = CallbackHandler(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    print("Langfuse enabled")
except Exception as error:
    LANGFUSE_AVAILABLE = False
    langfuse_client = None
    langfuse_handler = None
    print(f"Langfuse not available: {error}")

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

ARXIV_API_URL = os.getenv("ARXIV_API_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

print(f"Using model: {MODEL_NAME}")

# LLM Configuration with Langfuse callback
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0.2,
    api_key=OPENAI_API_KEY,
    callbacks=[langfuse_handler] if langfuse_handler else []
)

print(f"OpenAI client initialized with model: {llm.model_name}")


# ============================================================================
# State Definition
# ============================================================================

class ResearchState(TypedDict):
    """State for the research agent"""
    messages: Annotated[List[Dict], operator.add]
    query: str
    search_results: List[Dict]
    selected_papers: List[Dict]
    paper_details: Dict[str, Dict]
    current_question: str
    answers: List[Dict]
    synthesis: str
    plan: List[str]
    current_step: int
    research_goal: str
    iteration: int
    max_iterations: int
    citations: List[str]
    asked_questions: List[str]


# ============================================================================
# Tools - All with docstrings
# ============================================================================

@tool
def search_papers(query: str, top_k: int = 5) -> List[Dict]:
    """
    Search for papers relevant to a research query using the arXiv API.

    Args:
        query: The research question or topic to search for
        top_k: Number of papers to return (default: 5)

    Returns:
        List of paper dictionaries with title, authors, summary, arxiv_id
    """
    try:
        response = requests.post(
            f"{ARXIV_API_URL}/api/search",
            json={"question": query, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("papers", [])
        return []
    except Exception as e:
        print(f"Search error: {e}")
        return []


@tool
def get_paper_details(arxiv_id: str) -> Dict:
    """
    Get full details of a specific paper including all chunks and content.

    Args:
        arxiv_id: The arXiv ID of the paper (e.g., '2405.06693')

    Returns:
        Dictionary with full paper details including title, authors, abstract, and content chunks
    """
    try:
        response = requests.get(
            f"{ARXIV_API_URL}/api/paper/{arxiv_id}",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as error:
        print(f"Paper details error: {error}")
        return {}


@tool
def ask_question(question: str, arxiv_id: str = None, top_k: int = 3) -> Dict:
    """
    Ask a specific question to the research system and get an answer from papers.

    Args:
        question: The question to ask about the research topic
        arxiv_id: Optional specific paper ID to focus on
        top_k: Number of relevant chunks to retrieve (default: 3)

    Returns:
        Dictionary with 'answer' field containing the response and 'sources' with citations
    """
    try:
        payload = {"question": question, "top_k": top_k}
        if arxiv_id:
            payload["arxiv_id"] = arxiv_id

        response = requests.post(
            f"{ARXIV_API_URL}/api/ask",
            json=payload,
            timeout=60
        )
        if response.status_code == 200:
            return response.json()
        return {"answer": "Failed to get answer", "sources": []}
    except Exception as error:
        print(f"Question error: {error}")
        return {"answer": f"Error: {error}", "sources": []}


# ============================================================================
# Agent Nodes
# ============================================================================

def research_planner(state: ResearchState) -> ResearchState:
    """Plan the research approach based on user query"""
    print(f"Planning research for: {state['query']}")

    system_msg = """You are a research planning expert. Break down the user's research query into a step-by-step plan.
    Output a numbered list of 5-7 research steps."""

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Create a research plan for: {state['query']}")
    ]

    response = llm.invoke(messages)

    plan = [line.strip() for line in response.content.split('\n') if line.strip() and line[0].isdigit()]

    state['plan'] = plan
    state['current_step'] = 0
    state['research_goal'] = state['query']
    state['iteration'] = 0
    state['max_iterations'] = 15
    state['asked_questions'] = []

    print(f"📋 Plan created with {len(plan)} steps")
    return state


def search_executor(state: ResearchState) -> ResearchState:
    """Execute searches based on current research goal"""
    print(f"🔍 Executing search: {state['research_goal']}")

    results = search_papers.invoke({
        "query": state['research_goal'],
        "top_k": 5
    })

    state['search_results'] = results
    print(f"Found {len(results)} papers")
    return state


def paper_selector(state: ResearchState) -> ResearchState:
    """Select the most relevant papers from search results"""
    if not state['search_results']:
        print("No search results to select from")
        return state

    print("📚 Selecting relevant papers...")

    papers_info = []
    for i, paper in enumerate(state['search_results'][:10]):
        authors = paper.get('authors', [])
        if authors is None:
            authors = []
        elif not isinstance(authors, list):
            authors = [str(authors)] if authors else []

        papers_info.append({
            "index": i,
            "title": paper.get('title', 'Unknown'),
            "authors": authors[:3],
            "summary": paper.get('summary', '')[:200]
        })

    system_msg = "You are a paper selection expert. Return a JSON array of selected paper indices (0-based)."

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=f"Research goal: {state['research_goal']}\n\nPapers:\n{json.dumps(papers_info, indent=2)}")
    ]

    response = llm.invoke(messages)

    try:
        indices = re.findall(r'\d+', response.content)
        selected = [int(i) for i in indices if int(i) < len(state['search_results'])]

        if not selected:
            selected = list(range(min(3, len(state['search_results']))))

        state['selected_papers'] = [state['search_results'][i] for i in selected]
        print(f"Selected {len(state['selected_papers'])} papers")
    except Exception as error:
        print(f"Error selecting papers: {error}")
        state['selected_papers'] = state['search_results'][:3]

    return state


def paper_detail_fetcher(state: ResearchState) -> ResearchState:
    """Fetch detailed information for selected papers"""
    print("Fetching paper details...")

    for paper in state['selected_papers']:
        arxiv_id = paper.get('arxiv_id')
        if arxiv_id and arxiv_id not in state['paper_details']:
            print(f"  Fetching: {arxiv_id}")
            details = get_paper_details.invoke({"arxiv_id": arxiv_id})
            if details:
                if 'authors' in details and details['authors'] is None:
                    details['authors'] = []
                state['paper_details'][arxiv_id] = details

    print(f"Fetched details for {len(state['paper_details'])} papers")
    return state


def question_generator(state: ResearchState) -> ResearchState:
    """Generate specific, non-repeating questions to ask about the papers"""
    print("Generating specific questions...")

    if 'asked_questions' not in state:
        state['asked_questions'] = []

    if len(state.get('answers', [])) >= 6:
        print("Already have 6 answers, moving to synthesis")
        state['current_question'] = ""
        return state

    topics = ["methodology", "results", "applications", "limitations", "future directions",
              "key findings", "experimental setup", "data analysis", "conclusions"]
    topic_idx = len(state['asked_questions']) % len(topics)
    question = f"What are the main {topics[topic_idx]} discussed in the papers regarding {state['research_goal']}?"

    state['asked_questions'].append(question)
    state['current_question'] = question
    print(f"Generated question: {question[:100]}...")
    return state


def qa_executor(state: ResearchState) -> ResearchState:
    """Execute Q&A on current question"""
    if not state.get('current_question'):
        print("No question to answer")
        return state

    print(f"Answering: {state['current_question'][:100]}...")

    result = ask_question.invoke({
        "question": state['current_question'],
        "top_k": 3
    })

    answer = result.get('answer', 'No answer generated')

    answer_entry = {
        "question": state['current_question'],
        "answer": answer,
        "sources": result.get('sources', [])
    }

    if 'answers' not in state:
        state['answers'] = []
    state['answers'].append(answer_entry)

    if 'citations' not in state:
        state['citations'] = []
    for source in result.get('sources', []):
        if source not in state['citations']:
            state['citations'].append(source)

    print(f"Answer received ({len(answer)} chars)")
    print(f"Total answers collected: {len(state['answers'])}")

    state['current_question'] = ""
    return state


def synthesis_creator(state: ResearchState) -> ResearchState:
    """Create final synthesis from all answers"""
    print("Creating final synthesis...")

    if not state.get('answers') or len(state['answers']) == 0:
        state['synthesis'] = "No answers were generated during research."
        return state

    system_msg = """You are a research synthesis expert. Create a comprehensive synthesis from all the answers gathered.
    Format with clear sections and citations. Use markdown formatting."""

    answers_summary = []
    for i, ans in enumerate(state['answers']):
        answers_summary.append(f"## Question {i + 1}: {ans['question']}\n\n{ans['answer']}\n")

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(
            content=f"Research goal: {state['research_goal']}\n\nAnswers gathered:\n\n" + "\n".join(answers_summary)
        )
    ]

    try:
        response = llm.invoke(messages)
        state['synthesis'] = response.content
        print(f"Synthesis created ({len(state['synthesis'])} chars)")
    except Exception as error:
        print(f"Error creating synthesis: {error}")
        state[
            'synthesis'] = f"# Research Synthesis: {state['research_goal']}\n\nBased on {len(state['answers'])} questions answered."

    return state


def router(state: ResearchState) -> Literal["continue", "synthesize", "end"]:
    """Route to next step based on current state"""
    if 'iteration' not in state:
        state['iteration'] = 0
    else:
        state['iteration'] += 1

    print(f"🔄 Router - Iteration: {state['iteration']}, Answers: {len(state.get('answers', []))}")

    if state['iteration'] >= state.get('max_iterations', 15):
        return "synthesize"

    if len(state.get('answers', [])) >= 6:
        return "synthesize"

    if len(state.get('paper_details', {})) == 0:
        return "continue"

    if state.get('current_question'):
        return "continue"

    if len(state.get('answers', [])) < 6:
        return "continue"

    return "synthesize"


# ============================================================================
# Build the Graph
# ============================================================================

def build_research_graph():
    """Construct the research agent graph"""
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", research_planner)
    workflow.add_node("searcher", search_executor)
    workflow.add_node("selector", paper_selector)
    workflow.add_node("detail_fetcher", paper_detail_fetcher)
    workflow.add_node("question_generator", question_generator)
    workflow.add_node("qa_executor", qa_executor)
    workflow.add_node("synthesizer", synthesis_creator)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "selector")
    workflow.add_edge("selector", "detail_fetcher")
    workflow.add_edge("detail_fetcher", "question_generator")

    workflow.add_conditional_edges(
        "question_generator",
        router,
        {"continue": "qa_executor", "synthesize": "synthesizer", "end": END}
    )

    workflow.add_conditional_edges(
        "qa_executor",
        router,
        {"continue": "question_generator", "synthesize": "synthesizer", "end": END}
    )

    workflow.add_edge("synthesizer", END)

    memory = InMemorySaver()
    graph = workflow.compile(checkpointer=memory)

    return graph



#Research Agent Class

class ArxivResearchAgent:
    """Main agent class for arXiv research"""

    def __init__(self):
        self.graph = build_research_graph()
        self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def research(self, query: str, thread_id: str = None, trace_id: str = None) -> Dict:
        """Execute research on a query"""
        if not thread_id:
            thread_id = self.conversation_id

        initial_state = {
            "messages": [],
            "query": query,
            "search_results": [],
            "selected_papers": [],
            "paper_details": {},
            "current_question": "",
            "answers": [],
            "synthesis": "",
            "plan": [],
            "current_step": 0,
            "research_goal": query,
            "iteration": 0,
            "max_iterations": 15,
            "citations": [],
            "asked_questions": []
        }

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50,
            "callbacks": [langfuse_handler] if langfuse_handler else []
        }

        print(f"\nResearching: {query}")
        print("-" * 50)

        try:
            start_time = time.time()
            result = self.graph.invoke(initial_state, config)
            duration = time.time() - start_time

            return {
                "success": True,
                "query": query,
                "synthesis": result.get("synthesis", "No synthesis generated"),
                "answers": result.get("answers", []),
                "citations": result.get("citations", []),
                "papers_used": len(result.get("paper_details", {})),
                "thread_id": thread_id,
                "trace_id": trace_id,
                "duration_seconds": round(duration, 2)
            }
        except Exception as error:
            print(f"Error during research: {error}")
            traceback.print_exc()
            return {
                "success": False,
                "query": query,
                "error": str(error),
                "thread_id": thread_id,
                "trace_id": trace_id
            }



#FastAPI Models

class ResearchRequest(BaseModel):
    """Request model for research endpoint"""
    query: str = Field(..., description="Research query to investigate")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for continuing research")

    model_config = ConfigDict(
        json_schema_extra={"example": {"query": "Latest advances in protein structure", "thread_id": "thread_12345"}}
    )


class FollowupRequest(BaseModel):
    """Request model for follow-up questions"""
    question: str = Field(..., description="Follow-up question")
    thread_id: str = Field(..., description="Thread ID of existing research session")

    model_config = ConfigDict(
        json_schema_extra={"example": {"question": "What are the methods?", "thread_id": "thread_12345"}}
    )



#FastAPI App


agent_app = FastAPI(
    title="arXiv Research Agent",
    description="Autonomous research agent for arXiv papers",
    version="1.0.0"
)

research_agent = ArxivResearchAgent()


@agent_app.get("/")
async def root():
    """Root endpoint"""
    return {"app": "arXiv Research Agent", "status": "running"}


@agent_app.get("/agent/health")
async def agent_health():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME, "timestamp": datetime.now().isoformat()}


@agent_app.post("/agent/research", status_code=200)
async def agent_research(request: ResearchRequest):
    """Execute research using the agent"""
    print(f"\n{'=' * 60}")
    print(f"📥 Research request: {request.query}")
    print(f"📋 Thread ID: {request.thread_id}")
    print(f"{'=' * 60}")

    start_time = time.time()

    # Start Langfuse trace
    trace = None
    trace_id = None
    if LANGFUSE_AVAILABLE and langfuse_client:
        try:
            trace = langfuse_client.trace(
                name="research_session",
                input={"query": request.query},
                metadata={"thread_id": request.thread_id or "new", "model": MODEL_NAME}
            )
            trace_id = trace.id
            print(f"🔗 Langfuse trace: {trace_id}")
        except Exception as error:
            print(f"Langfuse error: {error}")

    try:
        result = research_agent.research(
            query=request.query,
            thread_id=request.thread_id,
            trace_id=trace_id
        )

        duration = time.time() - start_time

        # Add trace_id to result
        result['trace_id'] = trace_id

        # Update trace with results
        if trace and result.get("success"):
            try:
                trace.update(
                    output={
                        "success": True,
                        "synthesis_preview": result.get("synthesis", "")[:500],
                        "answers_count": len(result.get("answers", [])),
                        "duration_seconds": round(duration, 2)
                    }
                )
                langfuse_client.flush()
            except Exception as error:
                print(f"Langfuse update error: {error}")

        print(f"Research completed in {duration:.2f}s")
        return result

    except Exception as error:
        duration = time.time() - start_time
        print(f"Langfuse trace error: {error}")

        if trace:
            try:
                trace.update(level="ERROR", output={"error": str(error)})
                langfuse_client.flush()
            except Exception as le:
                print(f"Langfuse error: {le}")

        print(f"Error after {duration:.2f}s: {str(error)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(error))


@agent_app.post("/agent/followup")
async def agent_followup(request: FollowupRequest):
    """Ask a follow-up question"""
    try:
        result = research_agent.research(query=request.question, thread_id=request.thread_id)
        return {"answer": result.get("synthesis", "No answer"), "thread_id": request.thread_id}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@agent_app.get("/agent/status/{thread_id}")
async def agent_status(thread_id: str):
    """Get status of a research session"""
    return {"thread_id": thread_id, "status": "active"}



#Main execution


if __name__ == "__main__":
    print("=" * 70)
    print("ARXIV RESEARCH AGENT API")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"API: http://localhost:8001")
    print(f"Docs: http://localhost:8001/docs")
    print(f"Health: http://localhost:8001/agent/health")
    print("=" * 70)

    uvicorn.run(
        agent_app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )