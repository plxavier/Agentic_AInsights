import os
import traceback

os.environ["ANONYMIZED_TELEMETRY"] = "false"  # Disable ChromaDB telemetry FIRST

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from datetime import datetime

# Configuration
TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
DEBUG_MODE = os.getenv("DEBUG_MODE", "true").lower() == "true"

print(f"🚀 Starting in {'TEST' if TEST_MODE else 'PRODUCTION'} mode")
print(f"🔧 Debug mode: {'ON' if DEBUG_MODE else 'OFF'}")

app = FastAPI(
    title="AInsights AI Research Assistant API" + (" [TEST MODE]" if TEST_MODE else ""),
    description=("Test mode: Simulated responses, no real processing" if TEST_MODE
                 else "Advanced RAG system for academic research analysis"),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize variables as None
db = None
ai = None
upload_service = None

# Test mode storage
test_uploads = []

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_mode():
    """Dependency to get current mode"""
    return TEST_MODE


def initialize_components():
    """Lazy initialization of components (production mode only)"""
    global db, ai, upload_service

    if db is None and not TEST_MODE:
        print("🔧 Initializing production components...")
        try:
            from src.database import VectorDatabase
            from src.ai import InsightsAI
            from upload_service import UploadService

            db = VectorDatabase()
            print("✅ Database initialized")

            ai = InsightsAI(db)
            print("✅ AI initialized")

            upload_service = UploadService(db)
            print("✅ Upload service initialized")

            print("✅ All production components initialized successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            traceback.print_exc()
        except Exception as e:
            print(f"❌ Initialization error: {e}")
            traceback.print_exc()
    elif TEST_MODE:
        print("✅ Running in test mode - no components initialized")

    return db, ai, upload_service


# Request models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    filters: Optional[Dict] = None


# ======================
# COMMON ENDPOINTS
# ======================
@app.get("/")
async def root():
    return {
        "app": "AInsights",
        "version": "2.1.0",
        "mode": "test" if TEST_MODE else "production",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "debug": DEBUG_MODE,
        "endpoints": {
            "health": "/health",
            "ask": "/api/ask (POST)",
            "upload": "/api/upload (POST)",
            "debug_upload": "/api/debug/upload (POST)",
            "papers": "/api/papers (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint with detailed info"""
    if TEST_MODE:
        return {
            "status": "healthy",
            "mode": "test",
            "debug": DEBUG_MODE,
            "test_uploads": len(test_uploads),
            "timestamp": datetime.now().isoformat()
        }
    else:
        try:
            db, _, _ = initialize_components()
            count = db.get_document_count() if hasattr(db, 'get_document_count') else 0
            return {
                "status": "healthy",
                "mode": "production",
                "debug": DEBUG_MODE,
                "database": {
                    "connected": True,
                    "document_count": count
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "degraded",
                "mode": "production",
                "error": str(e),
                "debug": DEBUG_MODE,
                "timestamp": datetime.now().isoformat()
            }


@app.post("/api/debug/upload")
async def debug_upload(file: UploadFile = File(...)):
    """Debug endpoint to test uploads without any processing"""
    print(f"🔍 DEBUG UPLOAD CALLED: {file.filename}")

    try:
        # Read file
        contents = await file.read()
        print(f"📄 File size: {len(contents)} bytes")
        print(f"📄 Content type: {file.content_type}")
        print(f"📄 Headers: {file.headers}")

        # Save a copy for inspection
        os.makedirs("debug_uploads", exist_ok=True)
        debug_path = f"debug_uploads/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(debug_path, "wb") as f:
            f.write(contents)
        print(f"💾 Saved debug copy to: {debug_path}")

        # Simple response
        return {
            "success": True,
            "mode": "debug",
            "filename": file.filename,
            "size": len(contents),
            "saved_to": debug_path,
            "message": "File received successfully in debug mode"
        }

    except Exception as e:
        print(f"❌ Debug upload error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Debug error: {str(e)}")


@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    """Upload and process a PDF research paper"""
    print(f"📤 UPLOAD ENDPOINT CALLED: {file.filename}")

    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            error_msg = f"Only PDF files are supported. Got: {file.filename}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Read file
        contents = await file.read()
        print(f"📄 File read successfully: {len(contents)} bytes")

        if len(contents) == 0:
            error_msg = "Uploaded file is empty"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=400, detail=error_msg)

        # Check if it's actually a PDF (simple check)
        if contents[:4] != b'%PDF':
            print(f"⚠️  Warning: File doesn't start with PDF magic number")

        if TEST_MODE:
            # Test mode - simulate upload
            print(f"🧪 TEST MODE: Simulating upload for {file.filename}")

            test_uploads.append({
                "name": file.filename,
                "size": len(contents),
                "time": datetime.now().isoformat(),
                "mode": "test_simulation"
            })

            return {
                "success": True,
                "message": f"Test upload: {file.filename} (simulated)",
                "details": {
                    "filename": file.filename,
                    "size_bytes": len(contents),
                    "chunks_added": 3,
                    "status": "simulated",
                    "mode": "test"
                },
                "test_uploads_count": len(test_uploads)
            }
        else:
            # Production mode
            print(f"⚙️  PRODUCTION MODE: Processing {file.filename}")

            # Initialize components
            db, _, upload_service = initialize_components()

            if upload_service is None:
                error_msg = "Upload service not initialized"
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

            # Process the file
            print(f"🔧 Calling upload_service.upload_pdf()...")
            result = await upload_service.upload_pdf(contents, file.filename)
            print(f"✅ Upload service returned: {result.get('success', 'unknown')}")

            return result

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except ValueError as e:
        error_msg = f"Validation error: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except ImportError as e:
        error_msg = f"Module import error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )


@app.get("/api/papers")
async def list_papers():
    """List all papers in the database"""
    if TEST_MODE:
        return {
            "count": len(test_uploads),
            "papers": test_uploads,
            "mode": "test",
            "timestamp": datetime.now().isoformat()
        }
    else:
        try:
            db, _, _ = initialize_components()
            if hasattr(db, 'get_all_papers'):
                papers = db.get_all_papers()
            else:
                papers = []
            return {
                "count": len(papers),
                "papers": papers,
                "mode": "production",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Ask research questions"""
    if TEST_MODE:
        return {
            "success": True,
            "answer": f"Test response to: '{request.question}'\n\nThis is a simulated response in test mode.",
            "sources": [],
            "mode": "test"
        }
    else:
        try:
            db, ai, _ = initialize_components()
            if ai is None:
                raise HTTPException(status_code=500, detail="AI not initialized")

            result = ai.answer_question(
                question=request.question,
                top_k=request.top_k
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# Error handlers with detailed logging
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    print(f"❌ HTTP Exception {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "endpoint": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
            "mode": "test" if TEST_MODE else "production"
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    print(f"❌ Unhandled Exception: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": str(exc),
            "type": type(exc).__name__,
            "endpoint": str(request.url.path),
            "timestamp": datetime.now().isoformat(),
            "mode": "test" if TEST_MODE else "production",
            "traceback": traceback.format_exc() if DEBUG_MODE else None
        }
    )


@app.get("/api/debug/db")
async def debug_database():
    """Debug endpoint to check database contents"""
    try:
        db, _, _ = initialize_components()

        # Get all documents
        all_docs = db.collection.get()

        # Count papers
        papers = {}
        for metadata in all_docs.get("metadatas", []):
            paper_id = metadata.get("paper_id", "unknown")
            if paper_id not in papers:
                papers[paper_id] = {
                    "title": metadata.get("title", "Unknown"),
                    "source": metadata.get("source", "Unknown"),
                    "chunks": 0
                }
            papers[paper_id]["chunks"] += 1

        return {
            "total_documents": len(all_docs.get("documents", [])),
            "total_papers": len(papers),
            "papers": papers,
            "sample_documents": all_docs.get("documents", [])[:2],
            "sample_metadata": all_docs.get("metadatas", [])[:2]
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/paper-content")
async def debug_paper_content():
    """See what content was actually extracted from papers"""
    try:
        db, _, _ = initialize_components()
        all_docs = db.collection.get()

        papers_content = []
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        for i in range(min(10, len(documents))):  # Show first 10
            text = documents[i] if i < len(documents) else ""
            metadata = metadatas[i] if i < len(metadatas) else {}

            papers_content.append({
                "chunk_id": i,
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "paper_id": metadata.get("paper_id", "unknown"),
                "first_300_chars": text[:300],
                "total_length": len(text),
                "has_protein": "protein" in text.lower() if text else False,
                "has_design": "design" in text.lower() if text else False,
                "word_count": len(text.split()) if text else 0
            })

        # Check search function
        test_query = "protein design"
        search_results = db.search_documents(test_query, k=3)

        return {
            "total_chunks": len(documents),
            "papers": papers_content,
            "search_test": {
                "query": test_query,
                "found": len(search_results),
                "results": [
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "has_query": test_query.lower() in doc.page_content.lower(),
                        "preview": doc.page_content[:100]
                    } for doc in search_results
                ]
            }
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/api/debug/search-test")
async def debug_search_test(query: str = "protein"):
    """Test search functionality"""
    try:
        db, _, _ = initialize_components()

        # Test both search methods
        simple_results = db.search(query, k=5)
        doc_results = db.search_documents(query, k=5)

        all_docs = db.collection.get()
        total_text = " ".join(all_docs.get("documents", []))

        return {
            "query": query,
            "total_documents": len(all_docs.get("documents", [])),
            "query_in_total_text": query.lower() in total_text.lower(),
            "simple_search_type": type(simple_results).__name__,
            "simple_search_length": len(simple_results) if isinstance(simple_results, list) else "N/A",
            "document_search_results": len(doc_results),
            "document_search_details": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "contains_query": query.lower() in doc.page_content.lower(),
                    "preview": doc.page_content[:200]
                } for doc in doc_results
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/debug/test-upload")
async def debug_test_upload(file: UploadFile = File(...)):
    """Test upload with detailed logging"""
    try:
        print(f"🧪 DEBUG UPLOAD: {file.filename}")

        contents = await file.read()
        print(f"📄 File size: {len(contents)} bytes")

        # Save for inspection
        os.makedirs("debug_uploads", exist_ok=True)
        debug_path = f"debug_uploads/debug_{file.filename}"
        with open(debug_path, "wb") as f:
            f.write(contents)

        # Try to extract text
        from src.database import DocumentLoader
        pdf_result = DocumentLoader.safe_load_pdf_bytes(contents, file.filename)

        return {
            "debug": True,
            "filename": file.filename,
            "file_size": len(contents),
            "saved_to": debug_path,
            "pdf_extraction": {
                "success": pdf_result["success"],
                "text_length": len(pdf_result.get("text", "")),
                "metadata": pdf_result.get("metadata", {}),
                "text_preview": pdf_result.get("text", "")[:500]
            }
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/api/test-connection")
async def test_connection():
    """Test endpoint for frontend connectivity"""
    return {
        "status": "connected",
        "service": "AInsights AI Backend",
        "port": PORT,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/debug/chunks")
async def debug_chunks():
    """See all document chunks"""
    try:
        db, _, _ = initialize_components()

        # Get all documents from Chroma
        all_docs = db.collection.get()
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        chunks = []
        for i in range(len(documents)):
            text = documents[i]
            metadata = metadatas[i] if i < len(metadatas) else {}

            # Check for protein design terms
            has_protein = "protein" in text.lower()
            has_design = "design" in text.lower()
            has_both = has_protein and has_design

            chunks.append({
                "chunk_id": i,
                "paper_id": metadata.get("paper_id", "unknown"),
                "title": metadata.get("title", "Unknown"),
                "chunk_size": len(text),
                "preview": text[:200],
                "contains_protein": has_protein,
                "contains_design": has_design,
                "contains_both": has_both,
                "word_count": len(text.split())
            })

        return {
            "total_chunks": len(documents),
            "chunks": chunks,
            "summary": {
                "chunks_with_protein": sum(1 for c in chunks if c["contains_protein"]),
                "chunks_with_design": sum(1 for c in chunks if c["contains_design"]),
                "chunks_with_both": sum(1 for c in chunks if c["contains_both"])
            }
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/search-test")
async def search_test(query: str = "protein design"):
    """Test search with detailed output"""
    try:
        db, _, _ = initialize_components()

        print(f"\n🧪 SEARCH TEST: '{query}'")

        # Run search
        docs = db.search_documents(query, k=3)

        results = []
        for i, doc in enumerate(docs):
            # Highlight search terms in preview
            preview = doc.page_content[:300]
            for word in query.lower().split():
                if word in preview.lower():
                    preview = preview.replace(word, f"**{word}**")

            results.append({
                "rank": i + 1,
                "title": doc.metadata.get("title", "Unknown"),
                "score": "N/A",
                "preview": preview + "..." if len(doc.page_content) > 300 else "",
                "contains_query": query.lower() in doc.page_content.lower(),
                "length": len(doc.page_content)
            })

        # Also check what happens with the AI
        if docs:
            ai_response = "Would generate AI answer with these documents"
        else:
            ai_response = "No documents found for AI to use"

        return {
            "search_query": query,
            "documents_found": len(docs),
            "results": results,
            "ai_status": ai_response,
            "total_in_database": db.get_document_count()
        }
    except Exception as e:
        return {"error": str(e)}


# ======================
# STARTUP
# ======================
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    print("=" * 50)
    print(f"🚀 AInsights AI Research Assistant")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Mode: {'TEST' if TEST_MODE else 'PRODUCTION'}")
    print(f"🐛 Debug: {'ON' if DEBUG_MODE else 'OFF'}")
    print("=" * 50)

    if not TEST_MODE:
        print("🔄 Initializing production components...")
        initialize_components()

    print(f"✅ Server ready")
    print(f"📡 API: http://localhost:8001")
    print(f"📚 Docs: http://localhost:8001/docs")
    print(f"🔍 Debug upload: http://localhost:8001/api/debug/upload")
    print("=" * 50)


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("chroma_db", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("debug_uploads", exist_ok=True)

    # Run the server
    print("Starting server...")
    uvicorn.run(
        app,  # Pass app directly to avoid import recursion
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info" if DEBUG_MODE else "warning"
    )