# app_arxiv.py - FULL VERSION with PDF processing and Q&A (Pure Retrieved Content)
import os
import sys
import traceback
import tempfile
import hashlib
from datetime import datetime
from typing import List, Optional, Dict
import requests
from bs4 import BeautifulSoup
import PyPDF2
from io import BytesIO

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Optional, Dict


# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import database
from src.database_arxiv import ArxivDatabase


# Initialize database
db = ArxivDatabase(persist_dir="./arxiv_db")

app = FastAPI(
    title="AInsights arxiv",
    description="Full paper processing and Q&A",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Add this after loading env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("MODEL_NAME", "gpt-4.1-mini")

if OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("✅ OpenAI client initialized")
    except Exception as e:
        print(f"⚠️  OpenAI init failed: {e}")
        openai_client = None
else:
    openai_client = None
    print("⚠️  OPENAI_API_KEY not found")


# ========== Request/Response Models ==========
class ArxivIDRequest(BaseModel):
    arxiv_id: str


class QuestionRequest(BaseModel):
    question: str
    arxiv_id: Optional[str] = None
    top_k: Optional[int] = 3


class PaperResponse(BaseModel):
    arxiv_id: str
    title: str
    summary: str
    authors: List[str]
    published: Optional[str] = None
    pdf_url: Optional[str] = None
    html_url: Optional[str] = None


class AddPaperResponse(BaseModel):
    success: bool
    message: str
    paper_id: Optional[str] = None
    title: Optional[str] = None
    chunks: Optional[int] = 0
    error: Optional[str] = None


class AnswerResponse(BaseModel):
    success: bool
    answer: str
    sources: List[dict]
    papers_used: int


# ========== Helper Functions ==========
def fetch_arxiv_metadata(arxiv_id: str) -> Optional[dict]:
    """Fetch paper metadata from arXiv API"""
    try:
        import arxiv

        search = arxiv.Search(id_list=[arxiv_id])
        client = arxiv.Client()
        papers = list(client.results(search))

        if not papers:
            return None

        paper = papers[0]

        return {
            "arxiv_id": arxiv_id,
            "title": paper.title,
            "summary": paper.summary,
            "authors": [str(a) for a in paper.authors],
            "published": paper.published.isoformat() if paper.published else None,
            "pdf_url": paper.pdf_url,
            "html_url": f"https://arxiv.org/html/{arxiv_id}",
            "categories": paper.categories
        }
    except Exception as e:
        print(f"Error fetching arXiv metadata: {e}")
        return None


def fetch_pdf_content(pdf_url: str) -> Optional[bytes]:
    """Download PDF content"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(pdf_url, headers=headers, timeout=120)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return None


def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 50) -> str:
    """Extract text from PDF bytes"""
    try:
        pdf_file = BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        total_pages = min(len(pdf_reader.pages), max_pages)

        for i in range(total_pages):
            try:
                page_text = pdf_reader.pages[i].extract_text()
                if page_text:
                    text += f"\n--- Page {i + 1} ---\n{page_text}\n"
            except:
                text += f"\n--- Page {i + 1} [Error extracting] ---\n"

        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def fetch_html_content(html_url: str) -> Optional[str]:
    """Fetch HTML content from arXiv HTML5 version"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(html_url, headers=headers, timeout=120)
        if response.status_code == 200:
            return response.text
        return None
    except Exception as e:
        print(f"Error fetching HTML: {e}")
        return None


def extract_text_from_html(html_content: str) -> str:
    """Extract clean text from HTML"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        print(f"Error extracting HTML text: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks"""
    if not text or len(text.strip()) == 0:
        return []

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        if i >= len(words):
            break
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk and len(chunk.strip()) > 100:  # Only keep substantial chunks
            chunks.append(chunk)

        if len(chunks) >= 100:  # Safety limit
            break

    return chunks


async def synthesize_with_openai(question: str, chunks: List[dict], papers: List[dict]) -> str:
    """Better OpenAI synthesis with proper year handling"""

    if not openai_client:
        return None

    # Create a mapping of paper titles to their metadata
    paper_metadata = {}
    for paper in papers:
        title = paper.get('title', 'Unknown')
        authors = paper.get('authors', [])

        # Try multiple sources for year
        year = ''
        if paper.get('published'):
            year = paper['published'][:4] if len(paper['published']) >= 4 else ''
        elif paper.get('arxiv_id'):
            # Extract from arXiv ID
            arxiv_id = paper['arxiv_id']
            parts = arxiv_id.split('.')
            if parts and len(parts[0]) >= 2:
                two_digit = parts[0][:2]
                if two_digit.isdigit():
                    year = f"20{two_digit}"

        paper_metadata[title] = {
            'authors': authors,
            'year': year,
            'arxiv_id': paper.get('arxiv_id', '')
        }

    # Prepare context with clean citations
    context = ""
    for i, chunk in enumerate(chunks[:5]):
        text = chunk.get('text', '')[:800].replace('\n', ' ').strip()
        paper_title = chunk.get('paper_title', 'Unknown')

        # Get metadata for this paper
        meta = paper_metadata.get(paper_title, {})
        authors = meta.get('authors', [])
        year = meta.get('year', '')

        # Format citation without 'n.d.'
        if authors and len(authors) > 0:
            first_author = authors[0].split()[-1] if ' ' in authors[0] else authors[0]
            if len(authors) > 1:
                author_text = f"{first_author} et al."
            else:
                author_text = first_author
        else:
            author_text = paper_title[:20]  # Fallback to short title

        # Only add year if we have it
        if year:
            citation = f"({author_text}, {year})"
        else:
            citation = f"({author_text})"

        context += f"\nExcerpt {i + 1} from {citation}:\n{text}...\n"

    # Rest of your function remains the same...
    prompt = f"""You are a research assistant. Answer this question using ONLY the provided excerpts.

QUESTION: {question}

RELEVANT EXCERPTS:
{context}

INSTRUCTIONS:
1. Synthesize a coherent, well-structured answer
2. Cite sources using the EXACT citations shown in parentheses
3. Start with a definition/overview, then key concepts, end with implications

Write in clear, academic language:"""

    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system",
                 "content": "You synthesize answers from academic paper excerpts with proper citations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None

def get_section_emoji(section: str) -> str:
    """Get emoji for section"""
    emojis = {
        "introduction": "🔬",
        "findings": "📊",
        "applications": "💡",
        "methods": "🛠️",
        "results": "📈",
        "conclusion": "📌",
        "discussion": "💭",
        "background": "📚",
        "abstract": "📄"
    }
    return emojis.get(section.lower(), "📄")


def extract_relevant_chunks(question: str, chunks: List) -> List[dict]:
    """Extract and score chunks relevant to question"""
    keywords = set(question.lower().split())
    relevant = []

    for chunk in chunks[:30]:  # Look at first 30 chunks
        if isinstance(chunk, dict):
            text = chunk.get('text', '') or chunk.get('content', '')
            # Preserve paper metadata
            paper_title = chunk.get('paper_title', 'Unknown')
            paper_authors = chunk.get('paper_authors', [])
            paper_year = chunk.get('paper_year', 'n.d.')
        else:
            text = str(chunk)
            paper_title = 'Unknown'
            paper_authors = []
            paper_year = 'n.d.'

        if text and len(text.strip()) > 100:
            text_lower = text.lower()
            matches = sum(1 for k in keywords if k in text_lower and len(k) > 3)
            if matches > 0:
                relevant.append({
                    "text": text[:1000],
                    "relevance": matches,
                    "score": matches / len(keywords) if keywords else 0,
                    "paper_title": paper_title,
                    "paper_authors": paper_authors,
                    "paper_year": paper_year
                })

    relevant.sort(key=lambda x: x["relevance"], reverse=True)
    return relevant[:8]  # Top 8 chunks


def detect_section(text: str) -> str:
    """Detect which section a chunk belongs to based on keywords"""
    text_lower = text.lower()

    section_keywords = {
        "introduction": ["introduction", "background", "overview", "motivation"],
        "abstract": ["abstract", "summary"],
        "methods": ["method", "approach", "technique", "algorithm", "procedure", "experimental"],
        "results": ["result", "finding", "experiment", "performance", "accuracy", "evaluation"],
        "discussion": ["discussion", "analysis", "interpretation"],
        "applications": ["application", "use", "biotech", "medicine", "therapy", "drug"],
        "conclusion": ["conclusion", "future", "implication", "summary", "outlook"]
    }

    for section, keywords in section_keywords.items():
        if any(k in text_lower for k in keywords):
            return section

    return "findings"  # Default section


def generate_structured_answer(question: str, chunks: List[dict], paper_chunks_map: dict) -> str:
    """Generate a structured answer using ONLY retrieved chunks"""

    if not chunks:
        return f"# 📚 **{question.capitalize()}**\n\nNo relevant content found in the papers."

    # Organize chunks by detected section
    sections = {
        "introduction": [],
        "abstract": [],
        "background": [],
        "methods": [],
        "results": [],
        "discussion": [],
        "applications": [],
        "conclusion": [],
        "findings": []  # Default
    }

    for chunk in chunks:
        section = detect_section(chunk["text"])
        if section in sections:
            sections[section].append(chunk)
        else:
            sections["findings"].append(chunk)

    # Sort chunks in each section by relevance
    for section in sections:
        sections[section].sort(key=lambda x: x["relevance"], reverse=True)

    # Build the answer using ONLY retrieved content
    answer_parts = []

    # Title
    answer_parts.append(f"# 📚 **{question.capitalize()}**\n")

    # Introduction (if available)
    if sections["introduction"]:
        answer_parts.append("## 🔬 **Introduction**\n")
        for chunk in sections["introduction"][:2]:
            text = chunk["text"][:600].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Abstract (if available, otherwise use introduction)
    elif sections["abstract"]:
        answer_parts.append("## 📄 **Abstract**\n")
        for chunk in sections["abstract"][:2]:
            text = chunk["text"][:600].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Key Findings - use top relevant chunks
    answer_parts.append("## 📊 **Key Findings**\n")
    findings = []
    if sections["findings"]:
        findings.extend(sections["findings"][:3])
    if sections["results"]:
        findings.extend(sections["results"][:2])

    if findings:
        for chunk in findings[:4]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"• {text}...\n")
    else:
        # Use top chunks overall
        for chunk in chunks[:4]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"• {text}...\n")

    # Methods (if available)
    if sections["methods"]:
        answer_parts.append("## 🛠️ **Methods**\n")
        for chunk in sections["methods"][:2]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Applications (if available)
    if sections["applications"]:
        answer_parts.append("## 💡 **Applications**\n")
        for chunk in sections["applications"][:2]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Results (if available and not already covered)
    if sections["results"] and not sections["findings"]:
        answer_parts.append("## 📈 **Results**\n")
        for chunk in sections["results"][:2]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Discussion (if available)
    if sections["discussion"]:
        answer_parts.append("## 💭 **Discussion**\n")
        for chunk in sections["discussion"][:2]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Conclusion (if available)
    if sections["conclusion"]:
        answer_parts.append("## 📌 **Conclusion**\n")
        for chunk in sections["conclusion"][:2]:
            text = chunk["text"][:400].strip()
            if text:
                answer_parts.append(f"{text}...\n")

    # Add note about sources
    answer_parts.append("\n*This answer is generated from the content of the papers listed in the sources section.*")

    return '\n'.join(answer_parts)


def generate_structured_answer_from_chunks(question: str, chunks: List, paper: dict) -> str:
    """Generate answer for a single paper using ONLY its chunks"""

    # Convert chunks to text if they're dicts
    chunk_texts = []
    for chunk in chunks[:20]:  # Use first 20 chunks
        if isinstance(chunk, dict):
            text = chunk.get('text', '') or chunk.get('content', '')
        else:
            text = str(chunk)
        if text and len(text.strip()) > 100:
            chunk_texts.append(text)

    if not chunk_texts:
        return f"# 📚 **{question.capitalize()}**\n\nNo detailed content found in the paper."

    # Find relevant sections
    keywords = set(question.lower().split())

    # Score chunks
    scored_chunks = []
    for text in chunk_texts:
        text_lower = text.lower()
        matches = sum(1 for k in keywords if k in text_lower and len(k) > 3)
        scored_chunks.append((matches, text))

    scored_chunks.sort(reverse=True)

    # Organize by detected sections
    sections = {}
    for matches, text in scored_chunks[:15]:
        section = detect_section(text)
        if section not in sections:
            sections[section] = []
        sections[section].append((matches, text))

    # Build answer using paper content
    answer_parts = []

    # Title
    answer_parts.append(f"# 📚 **{question.capitalize()}**\n")

    # Paper info
    answer_parts.append(
        f"*Based on the paper: **{paper.get('title', 'Unknown')}** by {', '.join(paper.get('authors', ['Unknown']))[:100]}*\n")

    # Introduction - use first relevant chunk
    answer_parts.append("## 🔬 **Introduction**\n")
    if scored_chunks:
        answer_parts.append(f"{scored_chunks[0][1][:600]}...\n")

    # Key Findings - use top chunks
    answer_parts.append("## 📊 **Key Findings**\n")
    for i, (_, text) in enumerate(scored_chunks[1:5]):  # Next 4 chunks
        if i < 4:
            answer_parts.append(f"• {text[:400]}...\n")

    # Methods (if found)
    if "methods" in sections:
        answer_parts.append("## 🛠️ **Methods**\n")
        for matches, text in sections["methods"][:2]:
            answer_parts.append(f"{text[:400]}...\n")

    # Results (if found)
    if "results" in sections:
        answer_parts.append("## 📈 **Results**\n")
        for matches, text in sections["results"][:2]:
            answer_parts.append(f"{text[:400]}...\n")

    # Applications (if found)
    if "applications" in sections:
        answer_parts.append("## 💡 **Applications**\n")
        for matches, text in sections["applications"][:2]:
            answer_parts.append(f"{text[:400]}...\n")

    # Conclusion (if found)
    if "conclusion" in sections:
        answer_parts.append("## 📌 **Conclusion**\n")
        for matches, text in sections["conclusion"][:2]:
            answer_parts.append(f"{text[:400]}...\n")

    return '\n'.join(answer_parts)


# ========== API Endpoints ==========
@app.get("/")
def root():
    return {
        "app": "ArXiv Research API",
        "version": "2.0",
        "status": "running",
        "features": [
            "Fetch paper metadata",
            "Download full PDF/HTML content",
            "Smart text chunking",
            "Pure content-based Q&A",
            "Section-aware answers"
        ]
    }


@app.get("/health")
def health():
    stats = db.get_stats() if hasattr(db, 'get_stats') else {"total_papers": db.count_papers(), "total_chunks": 0}
    return {
        "status": "healthy",
        "database": {
            "connected": True,
            "papers": stats.get("total_papers", db.count_papers()),
            "chunks": stats.get("total_chunks", 0),
            "path": "./arxiv_db"
        },
        "features": {
            "pdf_processing": True,
            "html_extraction": True,
            "text_chunking": True,
            "qa_capability": True,
            "pure_content_answers": True
        }
    }


@app.post("/api/fetch", response_model=PaperResponse)
async def fetch_paper(request: ArxivIDRequest):
    """Fetch paper metadata from arXiv"""
    paper = fetch_arxiv_metadata(request.arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return paper


@app.post("/api/fetch_full")
async def fetch_full_paper(request: ArxivIDRequest):
    """Fetch full paper content (PDF/HTML)"""
    paper = fetch_arxiv_metadata(request.arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    full_text = ""
    source = "unknown"

    # Try HTML first (arXiv beta)
    if paper.get('html_url'):
        html_content = fetch_html_content(paper['html_url'])
        if html_content:
            full_text = extract_text_from_html(html_content)
            source = "html"

    # Fallback to PDF
    if not full_text and paper.get('pdf_url'):
        pdf_bytes = fetch_pdf_content(paper['pdf_url'])
        if pdf_bytes:
            full_text = extract_text_from_pdf(pdf_bytes)
            source = "pdf"

    if not full_text:
        raise HTTPException(status_code=500, detail="Could not fetch paper content")

    # Chunk the text
    chunks = chunk_text(full_text)

    return {
        "arxiv_id": paper['arxiv_id'],
        "title": paper['title'],
        "source": source,
        "total_chars": len(full_text),
        "chunks": len(chunks),
        "preview": full_text[:1000] + "..."
    }


@app.post("/api/add", response_model=AddPaperResponse)
async def add_paper(request: ArxivIDRequest, background_tasks: BackgroundTasks):
    """Fetch and add full paper to database"""
    # Fetch metadata
    paper = fetch_arxiv_metadata(request.arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Try HTML first
    full_text = ""
    source = "metadata_only"

    if paper.get('html_url'):
        html_content = fetch_html_content(paper['html_url'])
        if html_content:
            full_text = extract_text_from_html(html_content)
            source = "html"

    # Fallback to PDF
    if not full_text and paper.get('pdf_url'):
        pdf_bytes = fetch_pdf_content(paper['pdf_url'])
        if pdf_bytes:
            full_text = extract_text_from_pdf(pdf_bytes)
            source = "pdf"

    # If no full text, use abstract
    if not full_text:
        full_text = paper['summary']
        source = "abstract_only"

    # Chunk the text
    chunks = chunk_text(full_text)

    # Prepare paper data for database
    paper_data = {
        "arxiv_id": paper['arxiv_id'],
        "title": paper['title'],
        "authors": paper['authors'],
        "summary": paper['summary'],
        "published": paper['published'],
        "full_text": full_text[:50000],  # Limit size
        "chunks": chunks,
        "source_type": source,
        "added_date": datetime.now().isoformat()
    }

    # Add to database
    if hasattr(db, 'add_full_paper'):
        result = db.add_full_paper(paper_data)
    else:
        # Fallback to basic add_paper
        result = db.add_paper(paper_data)
        result["chunks_added"] = len(chunks)

    if result.get("success", False):
        return {
            "success": True,
            "message": f"Paper '{paper['title']}' added successfully",
            "paper_id": result.get("paper_id", paper['arxiv_id']),
            "title": paper["title"],
            "chunks": len(chunks)
        }
    else:
        return {
            "success": False,
            "message": "Failed to add paper",
            "error": result.get("error", "Unknown error")
        }


@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask questions and get AI-synthesized answers"""
    try:
        print(f"📝 Question received: {request.question}")

        # Your existing code to get papers and chunks...
        papers = db.search_papers(request.question, n_results=request.top_k or 3)

        if not papers:
            return {
                "success": True,
                "answer": f"No relevant papers found for: {request.question}",
                "sources": [],
                "papers_used": 0
            }

        # Collect chunks with proper paper metadata
        all_chunks = []
        sources = []

        for p in papers[:3]:
            full_paper = db.get_full_paper(p.get('arxiv_id'))
            if full_paper and full_paper.get('chunks'):
                # Debug print to see what authors look like
                print(f"\n📚 PAPER METADATA:")
                print(f"   Title: {full_paper.get('title', 'Unknown')[:50]}...")
                print(f"   Authors: {full_paper.get('authors', [])}")
                print(f"   Type: {type(full_paper.get('authors'))}")

                # Ensure authors is a list
                authors = full_paper.get('authors', [])
                if isinstance(authors, str):
                    authors = [authors]

                chunks = extract_relevant_chunks(request.question, full_paper['chunks'])

                # Add paper title to each chunk for later reference
                for chunk in chunks:
                    chunk['paper_title'] = full_paper['title']
                    chunk['paper_authors'] = authors
                    chunk['paper_year'] = full_paper.get('published', '')[:4] if full_paper.get('published') else 'n.d.'

                all_chunks.extend(chunks)
                sources.append({
                    "title": full_paper['title'],
                    "arxiv_id": full_paper['arxiv_id'],
                    "authors": authors,
                    "published": full_paper.get('published', '')
                })

        # Try OpenAI synthesis if available
        if openai_client and all_chunks:
            print("🤖 Attempting OpenAI synthesis...")
            answer = await synthesize_with_openai(request.question, all_chunks, sources)
            if answer:
                print("✅ OpenAI synthesis successful")
                return {
                    "success": True,
                    "answer": answer,
                    "sources": sources,
                    "papers_used": len(sources)
                }
            else:
                print("⚠️  OpenAI failed, using fallback")

        # Your existing fallback answer generation
        answer = generate_structured_answer(request.question, all_chunks, {})

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "papers_used": len(sources)
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "answer": f"Error: {str(e)}",
            "sources": [],
            "papers_used": 0
        }


@app.post("/api/search")
async def search_papers(request: QuestionRequest):
    """Search for papers relevant to a question"""
    papers = db.search_papers(request.question, n_results=request.top_k or 5)

    return {
        "success": True,
        "papers": papers,
        "count": len(papers)
    }


@app.get("/api/papers")
async def list_papers():
    """List all papers in database"""
    papers = db.get_all_papers()
    return {
        "count": len(papers),
        "papers": papers
    }


@app.get("/api/paper/{arxiv_id}")
async def get_paper(arxiv_id: str):
    """Get full paper details including chunks"""
    if hasattr(db, 'get_full_paper'):
        paper = db.get_full_paper(arxiv_id)
    else:
        paper = db.get_paper(arxiv_id)

    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    return paper


@app.get("/api/stats")
async def get_stats():
    """Get database statistics"""
    if hasattr(db, 'get_stats'):
        stats = db.get_stats()
        return {
            "database": {
                "total_papers": stats.get("total_papers", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "total_documents": stats.get("total_documents", 0),
                "collection": "arxiv_papers"
            },
            "papers": db.get_all_papers()
        }
    else:
        papers = db.get_all_papers()
        total_chunks = sum(len(p.get('chunks', [])) if isinstance(p.get('chunks'), list) else 0 for p in papers)

        return {
            "database": {
                "total_papers": len(papers),
                "total_chunks": total_chunks,
                "collection": "arxiv_papers"
            },
            "papers": papers
        }


@app.post("/api/clear")
async def clear_database():
    """Clear all papers"""
    success = db.clear()
    return {
        "success": success,
        "message": "Database cleared" if success else "Failed to clear"
    }


if __name__ == "__main__":
    print("=" * 70)
    print("📚 ARXIV RESEARCH API v2.0 - PURE CONTENT ANSWERS")
    print("=" * 70)
    print(f"📡 API: http://localhost:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print(f"🩺 Health: http://localhost:8000/health")

    # Get paper count safely
    try:
        paper_count = db.count_papers()
        print(f"📊 Papers in DB: {paper_count}")
    except:
        print(f"📊 Papers in DB: Unknown")

    print("\n✨ Features:")
    print("   • Fetch full papers (PDF/HTML)")
    print("   • Smart text chunking")
    print("   • Section-aware Q&A")
    print("   • Pure content answers (no templates)")
    print("   • Multi-paper synthesis")
    print("=" * 70)

    uvicorn.run(
        "app_arxiv:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )