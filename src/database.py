import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter  # CHANGED: No recursion
from langchain.schema import Document
import os
from typing import List, Dict, Any, Optional
import PyPDF2
import tempfile
import hashlib
import sys
from datetime import datetime
from io import BytesIO
import fitz

import os
from dotenv import load_dotenv

#loading environment variables
load_dotenv(override=True)

#checking if key is loaded
api_key = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: API Key loaded: {'YES' if api_key else 'NO'}")
if api_key:
    print(f"Key starts with: {api_key[:10]}...")


class VectorDatabase:
    """Enhanced vector database with RAG optimizations"""

    def __init__(self, collection_name: str = "insights_research"):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

        #skipping OpenAI to avoid API key issues
        print("Initializing without OpenAI embeddings (temporary)")
        self.embeddings = None

        #initializing ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False,
                              allow_reset=True,
                              is_persistent=True
                              )
        )

        #get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "AInsights AI Research Database"}
        )

        #LangChain Chroma for easier operations
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=lambda texts: [[0.0] * 384 for _ in texts],  # Dummy embeddings
            persist_directory=self.persist_dir,
            client=self.client
        )

        #non-recursive text splitter
        self.text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n",
            length_function=len
        )

        #cache for text documents
        self.text_documents_cache = []

        #track uploaded files to prevent duplicates
        self.processed_hashes = set()


    def safe_add_paper(self, text: str, metadata: Dict = None, max_chunks: int = 100) -> Dict:
        """SAFE VERSION: Add paper with recursion protection"""
        if metadata is None:
            metadata = {}

        #generate hash to prevent duplicate processing
        text_hash = hashlib.md5(text.encode()).hexdigest()[:16]

        if text_hash in self.processed_hashes:
            print(f"Duplicate content detected, skipping")
            return {"chunks_added": 0, "status": "duplicate"}

        #limit text length to prevent issues
        if len(text) > 100000:
            text = text[:100000]
            print(f"Text truncated to 100k characters")

        #simple splitting - NO recursion
        words = text.split()
        chunks = []

        #create fixed-size chunks safely
        chunk_size = 800
        for i in range(0, len(words), chunk_size):
            if len(chunks) >= max_chunks:
                print(f"Hit max chunks limit ({max_chunks})")
                break
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

            if i > 0 and i % 5000 == 0:
                print(f"   Processed {i} words...")

        print(f"Created {len(chunks)} safe chunks")

        #prepare metadata
        import datetime
        metadata.update({
            "paper_id": f"paper_{text_hash}",
            "added_date": datetime.datetime.now().isoformat(),
            "title": metadata.get("title", "Unknown"),
            "chunks_created": len(chunks),
            "processing": "safe_mode"
        })

        #add to database with unique IDs
        ids = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()[:8]
            })

            #use Chroma directly - simpler
            self.collection.add(
                documents=[chunk],
                metadatas=[chunk_metadata],
                ids=[f"{metadata['paper_id']}_chunk_{i}_{text_hash}"]
            )

            #cache
            self.text_documents_cache.append({
                "text": chunk[:500],
                "metadata": chunk_metadata
            })

        self.processed_hashes.add(text_hash)

        print(f"Added paper safely with {len(chunks)} chunks")
        return {"chunks_added": len(chunks), "status": "success", "hash": text_hash}


    def search(self, query: str, k: int = 5, filters: Optional[Dict] = None):
        """Simple search - works without embeddings"""
        try:
            return self.collection.query(
                query_texts=[query],
                n_results=k,
                where=filters
            )
        except:
            # Fallback to simple text search
            results = []
            for doc in self.text_documents_cache[-100:]:  # Last 100 docs
                if query.lower() in doc["text"].lower():
                    results.append(Document(
                        page_content=doc["text"],
                        metadata=doc["metadata"]
                    ))
            return results[:k]


    def search_documents(self, query: str, k: int = 5, filters: Optional[Dict] = None) -> List[Document]:
        """Search and return LangChain Document objects - IMPROVED FOR SPECIFIC DETAILS"""
        try:
            print(f"\nSEARCHING for: '{query}'")

            #get ALL documents
            all_results = self.collection.get()
            all_docs = all_results.get("documents", [])
            all_metas = all_results.get("metadatas", [])

            if not all_docs:
                print("No documents in database!")
                return []

            print(f"Searching through {len(all_docs)} chunks")

            #convert query to lowercase
            query_lower = query.strip().lower()
            query_words = [w.strip() for w in query_lower.split() if len(w.strip()) > 2]

            scored_docs = []

            for i, (text, metadata) in enumerate(zip(all_docs, all_metas)):
                if not text or len(text.strip()) == 0:
                    continue

                text_lower = text.lower()
                score = 0

                #1.exact phrase match (highest priority)
                if query_lower in text_lower:
                    score += 100  # Very high score for exact phrase
                    print(f"Chunk {i}: Found exact phrase")

                #2.check for ALL query words (AND logic)
                all_words_present = all(word in text_lower for word in query_words if len(word) > 3)
                if all_words_present and len(query_words) > 1:
                    score += 80  # High score for all words

                #3.individual word matches with weighting
                for word in query_words:
                    if len(word) > 3:  # Only consider meaningful words
                        if word in text_lower:
                            #count occurrences
                            occurrences = text_lower.count(word)
                            score += occurrences * 15

                            #bonus for technical terms
                            technical_terms = ['rf', 'diffusion', 'protein', 'design', 'model', 'deep', 'learning',
                                               'structure']
                            if word in technical_terms:
                                score += 10

                #4.title/author/source bonus
                if metadata:
                    meta_text = " ".join([
                        str(metadata.get("title", "")),
                        str(metadata.get("author", "")),
                        str(metadata.get("source", ""))
                    ]).lower()

                    for word in query_words:
                        if len(word) > 3 and word in meta_text:
                            score += 30

                if score > 0:
                    scored_docs.append({
                        "score": score,
                        "text": text,
                        "metadata": metadata or {},
                        "index": i
                    })

            #sort by score (highest first)
            scored_docs.sort(key=lambda x: x["score"], reverse=True)

            #take top k
            top_docs = scored_docs[:k]

            print(f"\nSEARCH RESULTS:")
            print(f"   Found {len(scored_docs)} chunks with score > 0")
            print(f"   Returning top {len(top_docs)} chunks")

            if top_docs:
                print(f"   Best score: {top_docs[0]['score']}")
                print(f"   Best chunk preview: {top_docs[0]['text'][:200]}...")

            #convert to Document objects
            documents = []
            for item in top_docs:
                doc = Document(
                    page_content=item["text"],
                    metadata=item["metadata"]
                )
                documents.append(doc)

            return documents

        except Exception as error:
            print(f"Search error: {error}")
            import traceback
            traceback.print_exc()
            return []


    def get_document_count(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()


    def get_all_papers(self) -> List[Dict]:
        """Get metadata for all papers"""
        results = self.collection.get()
        papers = {}

        for metadata in results.get("metadatas", []):
            paper_id = metadata.get("paper_id")
            if paper_id and paper_id not in papers:
                papers[paper_id] = {
                    "title": metadata.get("title", "Unknown"),
                    "source": metadata.get("source", "Unknown"),
                    "author": metadata.get("author", "Unknown"),
                    "year": metadata.get("year", "Unknown"),
                    "chunks": metadata.get("total_chunks", 1),
                    "added_date": metadata.get("added_date", "")
                }

        return list(papers.values())


    def clear_database(self):
        """Clear all documents from the database"""
        self.collection.delete(where={})
        self.text_documents_cache = []
        self.processed_hashes.clear()
        print("Database cleared")


class DocumentLoader:
    """SAFE document loader with recursion protection"""

    @staticmethod
    def safe_load_pdf_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf", max_pages: int = 20) -> Dict:
        """SAFE VERSION: Load PDF from bytes WITHOUT recursion"""
        print(f"Processing {filename} safely...")

        try:
            #use BytesIO to avoid temp files
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            total_pages = len(pdf_reader.pages)
            pages_to_process = min(max_pages, total_pages)

            #simple extraction - NO recursion
            for i in range(pages_to_process):
                try:
                    page = pdf_reader.pages[i]
                    page_text = page.extract_text()

                    if page_text:
                        #clean and limit text
                        cleaned_text = ' '.join(page_text.split()[:500])  # Limit words
                        text += f"\n[Page {i + 1}] {cleaned_text}\n"

                    if i > 0 and i % 5 == 0:
                        print(f"   Extracted page {i + 1}/{pages_to_process}...")

                except Exception as page_error:
                    print(f"Error on page {i + 1}: {page_error}")
                    continue

            #extract basic metadata
            metadata = {}
            if pdf_reader.metadata:
                metadata = {
                    "title": str(pdf_reader.metadata.get("/Title", filename.replace(".pdf", ""))),
                    "author": str(pdf_reader.metadata.get("/Author", "Unknown")),
                    "source": filename
                }

            return {
                "success": True,
                "text": text[:50000],  # Limit total text
                "metadata": {
                    "source": filename,
                    "pages": total_pages,
                    "pages_processed": pages_to_process,
                    "text_length": len(text),
                    **metadata
                },
                "hash": hashlib.md5(text.encode()).hexdigest()[:16]
            }

        except Exception as error:
            print(f"PDF processing error: {error}")
            return {
                "success": False,
                "text": "",
                "metadata": {"source": filename, "error": str(e)},
                "hash": ""
            }


    @staticmethod
    def load_pdf(file_path: str, max_pages: int = 20) -> Dict:
        """Load local PDF file"""
        try:
            with open(file_path, 'rb') as f:
                pdf_bytes = f.read()
            return DocumentLoader.safe_load_pdf_bytes(pdf_bytes, os.path.basename(file_path), max_pages)
        except Exception as error:
            print(f"Error reading file {file_path}: {error}")
            return {"success": False, "text": "", "metadata": {"error": str(error)}}


    @staticmethod
    def safe_chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """SAFE chunking - NO recursion"""
        if not text or len(text.strip()) == 0:
            return []

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            if len(chunks) >= 50:  # Safety limit
                break
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks


# ====== FIXED UPLOAD ENDPOINT INTEGRATION ======

class FixedUploadAPI:
    """Integrated fixed upload handler"""

    def __init__(self, vector_db: VectorDatabase):
        self.db = vector_db
        self.upload_history = []

    async def handle_upload(self, file_bytes: bytes, filename: str) -> Dict:
        """Handle PDF upload safely"""
        print(f"Processing upload: {filename}")

        #step 1: Safe PDF extraction
        pdf_result = DocumentLoader.safe_load_pdf_bytes(file_bytes, filename)

        if not pdf_result["success"]:
            return {
                "success": False,
                "message": f"PDF extraction failed: {pdf_result['metadata'].get('error', 'Unknown')}",
                "details": {"error": "extraction_failed"}
            }

        #step 2: Check if already processed
        text_hash = pdf_result["hash"]
        for upload in self.upload_history:
            if upload.get("hash") == text_hash:
                return {
                    "success": True,
                    "message": f"'{filename}' already processed",
                    "details": {"status": "duplicate", "hash": text_hash}
                }

        #step 3: Add to database safely
        db_result = self.db.safe_add_paper(
            text=pdf_result["text"],
            metadata=pdf_result["metadata"]
        )

        #step 4: Record upload
        upload_record = {
            "filename": filename,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(pdf_result["text"]),
            "chunks_added": db_result.get("chunks_added", 0),
            "hash": text_hash,
            "status": db_result.get("status", "unknown")
        }

        self.upload_history.append(upload_record)

        return {
            "success": True,
            "message": f"'{filename}' processed successfully",
            "details": {
                "filename": filename,
                "chunks_added": db_result.get("chunks_added", 0),
                "text_length": len(pdf_result["text"]),
                "pages": pdf_result["metadata"].get("pages", 0),
                "hash": text_hash,
                "status": "processed_safely"
            }
        }


    def get_stats(self) -> Dict:
        """Get upload statistics"""
        return {
            "total_uploads": len(self.upload_history),
            "total_chunks": sum(u.get("chunks_added", 0) for u in self.upload_history),
            "recent_uploads": self.upload_history[-10:] if self.upload_history else []
        }



# ====== QUICK TEST ======
if __name__ == "__main__":
    print("Testing safe database...")

    #initialize
    db = VectorDatabase()
    upload_api = FixedUploadAPI(db)

    print(f"Database initialized")
    print(f"Upload API ready")
    print(f"Current documents: {db.get_document_count()}")

    #test with sample text
    test_text = "This is a test document. " * 100
    result = db.safe_add_paper(test_text, {"title": "Test Document", "source": "test"})

    print(f"Test added: {result}")
    print(f"Now documents: {db.get_document_count()}")