# database_arxiv.py - FULL VERSION with chunking and full paper support
import os
import hashlib
import PyPDF2
from io import BytesIO
from datetime import datetime
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings

# Disable telemetry
os.environ["CHROMA_TELEMETRY_ANONYMOUS"] = "false"


class ArxivDatabase:
    """Enhanced database for arXiv papers with full paper support"""

    def __init__(self, persist_dir: str = "./arxiv_db"):
        self.persist_dir = persist_dir

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"description": "ArXiv Papers Database"}
        )

        # In-memory cache
        self.papers_cache = {}
        print(f"✅ Database initialized at {persist_dir}")

    def add_paper(self, paper_data: Dict) -> Dict:
        """Add a paper abstract to the database"""
        try:
            # Generate unique ID
            paper_id = paper_data.get("arxiv_id", hashlib.md5(paper_data["title"].encode()).hexdigest()[:8])

            # Prepare document
            content = f"""
Title: {paper_data['title']}
Authors: {', '.join(paper_data['authors'])}
Published: {paper_data.get('published', 'Unknown')}
Summary: {paper_data['summary']}
            """.strip()

            # Add to ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[{
                    "title": paper_data["title"],
                    "authors": ", ".join(paper_data["authors"]),
                    "arxiv_id": paper_id,
                    "source": "arxiv",
                    "paper_type": "abstract_only",
                    "added_date": datetime.now().isoformat()
                }],
                ids=[paper_id]
            )

            # Cache
            self.papers_cache[paper_id] = paper_data

            return {
                "success": True,
                "paper_id": paper_id,
                "title": paper_data["title"]
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def add_full_paper(self, paper_data: Dict) -> Dict:
        """Add a full paper with chunks to the database"""
        try:
            # Generate unique ID
            paper_id = paper_data.get("arxiv_id", hashlib.md5(paper_data["title"].encode()).hexdigest()[:8])

            # Prepare the full text content summary
            full_content = f"""
TITLE: {paper_data['title']}
AUTHORS: {', '.join(paper_data['authors'])}
PUBLISHED: {paper_data.get('published', 'Unknown')}
ARXIV_ID: {paper_id}
SOURCE_TYPE: {paper_data.get('source_type', 'unknown')}
ABSTRACT: {paper_data['summary']}

--- FULL PAPER CONTENT ({paper_data.get('source_type', 'unknown').upper()}) ---
{paper_data.get('full_text', '')[:5000]}...
            """.strip()

            # Add the main paper document
            self.collection.add(
                documents=[full_content],
                metadatas=[{
                    "title": paper_data["title"],
                    "authors": ", ".join(paper_data["authors"]),
                    "arxiv_id": paper_id,
                    "source": "arxiv",
                    "paper_type": "full_paper",
                    "source_type": paper_data.get("source_type", "unknown"),
                    "added_date": paper_data.get("added_date", datetime.now().isoformat()),
                    "total_chunks": len(paper_data.get("chunks", [])),
                    "has_full_text": bool(paper_data.get("full_text"))
                }],
                ids=[f"{paper_id}_main"]
            )

            # Add each chunk as separate documents for granular search
            chunks_added = 0
            if paper_data.get("chunks"):
                chunk_ids = []
                chunk_docs = []
                chunk_metadatas = []

                # Limit to first 100 chunks to avoid overwhelming
                for i, chunk in enumerate(paper_data["chunks"][:100]):
                    if chunk and len(chunk.strip()) > 50:  # Only add non-empty chunks
                        chunk_id = f"{paper_id}_chunk_{i:03d}"
                        chunk_ids.append(chunk_id)
                        chunk_docs.append(chunk)
                        chunk_metadatas.append({
                            "title": paper_data["title"],
                            "arxiv_id": paper_id,
                            "chunk_index": i,
                            "chunk_type": "content",
                            "source": "arxiv",
                            "source_type": paper_data.get("source_type", "unknown"),
                            "added_date": paper_data.get("added_date", datetime.now().isoformat())
                        })
                        chunks_added += 1

                # Add chunks in batches
                if chunk_ids:
                    self.collection.add(
                        documents=chunk_docs,
                        metadatas=chunk_metadatas,
                        ids=chunk_ids
                    )
                    print(f"📚 Added {chunks_added} chunks for {paper_data['title']}")

            # Cache the paper data
            self.papers_cache[paper_id] = {
                **paper_data,
                "chunks_added": chunks_added
            }

            return {
                "success": True,
                "paper_id": paper_id,
                "title": paper_data["title"],
                "chunks_added": chunks_added
            }

        except Exception as e:
            print(f"❌ Error adding full paper: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_full_paper(self, arxiv_id: str) -> Optional[Dict]:
        """Get full paper details including chunks"""
        try:
            # Get all documents with this arxiv_id
            results = self.collection.get(
                where={"arxiv_id": arxiv_id}
            )

            if not results or not results["metadatas"]:
                return None

            # Separate main paper and chunks
            main_paper = None
            chunks = []

            for i, metadata in enumerate(results["metadatas"]):
                doc_id = results["ids"][i]

                if doc_id.endswith("_main"):
                    main_paper = {
                        "title": metadata.get("title"),
                        "arxiv_id": metadata.get("arxiv_id"),
                        "authors": metadata.get("authors", "").split(", "),
                        "source_type": metadata.get("source_type"),
                        "added_date": metadata.get("added_date"),
                        "total_chunks": metadata.get("total_chunks", 0),
                        "content": results["documents"][i] if results.get("documents") else None
                    }
                elif "_chunk_" in doc_id:
                    chunks.append({
                        "index": metadata.get("chunk_index"),
                        "text": results["documents"][i] if results.get("documents") else None,
                        "metadata": metadata
                    })

            # Sort chunks by index
            chunks.sort(key=lambda x: x.get("index", 0))

            if main_paper:
                main_paper["chunks"] = chunks
                return main_paper

            return None

        except Exception as e:
            print(f"❌ Error getting paper: {e}")
            return None

    def get_paper(self, paper_id: str) -> Optional[Dict]:
        """Get a paper by ID"""
        try:
            result = self.collection.get(ids=[paper_id])
            if result and result["metadatas"]:
                return result["metadatas"][0]
            return None
        except:
            return None

    def search_papers(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search papers by query - searches both abstracts and full text"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results * 3  # Get more to filter
            )

            papers = []
            seen_ids = set()

            if results and results["metadatas"] and results["metadatas"][0]:
                for i, metadata in enumerate(results["metadatas"][0]):
                    arxiv_id = metadata.get("arxiv_id")

                    # Only include main papers, not chunks
                    if arxiv_id and arxiv_id not in seen_ids:
                        # Check if this is a main paper or chunk
                        doc_id = results["ids"][0][i] if results.get("ids") else ""

                        if not doc_id.endswith("_chunk"):  # Main paper
                            seen_ids.add(arxiv_id)
                            papers.append({
                                "title": metadata.get("title"),
                                "arxiv_id": arxiv_id,
                                "authors": metadata.get("authors"),
                                "paper_type": metadata.get("paper_type", "abstract"),
                                "source_type": metadata.get("source_type", "unknown"),
                                "total_chunks": metadata.get("total_chunks", 0),
                                "relevance_score": 1 - results["distances"][0][i] if results.get("distances") else 0
                            })

                            if len(papers) >= n_results:
                                break

                # If not enough results, add papers from chunks
                if len(papers) < n_results:
                    for i, metadata in enumerate(results["metadatas"][0]):
                        arxiv_id = metadata.get("arxiv_id")
                        if arxiv_id and arxiv_id not in seen_ids:
                            seen_ids.add(arxiv_id)
                            papers.append({
                                "title": metadata.get("title"),
                                "arxiv_id": arxiv_id,
                                "authors": metadata.get("authors"),
                                "paper_type": "full_paper",
                                "source_type": metadata.get("source_type", "unknown"),
                                "relevance_score": 1 - results["distances"][0][i] if results.get("distances") else 0
                            })

                            if len(papers) >= n_results:
                                break

            return papers

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []

    def get_all_papers(self) -> List[Dict]:
        """Get all papers (main documents only, not chunks)"""
        try:
            results = self.collection.get()
            papers = []
            seen_ids = set()

            if results and results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"]):
                    doc_id = results["ids"][i] if results.get("ids") else ""
                    arxiv_id = metadata.get("arxiv_id")

                    # Only include main papers, not chunks
                    if arxiv_id and arxiv_id not in seen_ids and not doc_id.endswith("_chunk"):
                        seen_ids.add(arxiv_id)
                        papers.append({
                            "title": metadata.get("title", "Unknown"),
                            "arxiv_id": arxiv_id,
                            "authors": metadata.get("authors", "Unknown"),
                            "paper_type": metadata.get("paper_type", "abstract"),
                            "source_type": metadata.get("source_type", "unknown"),
                            "total_chunks": metadata.get("total_chunks", 0),
                            "added_date": metadata.get("added_date", "")
                        })
            return papers
        except Exception as e:
            print(f"❌ Error getting papers: {e}")
            return []

    def count_papers(self) -> int:
        """Get number of papers (main documents only)"""
        try:
            all_papers = self.get_all_papers()
            return len(all_papers)
        except:
            return 0

    def get_stats(self) -> Dict:
        """Get detailed database statistics"""
        try:
            results = self.collection.get()
            main_papers = 0
            total_chunks = 0
            source_types = {"html": 0, "pdf": 0, "abstract": 0, "unknown": 0}

            if results and results["metadatas"]:
                for i, metadata in enumerate(results["metadatas"]):
                    doc_id = results["ids"][i] if results.get("ids") else ""

                    if doc_id.endswith("_main"):
                        main_papers += 1
                        source = metadata.get("source_type", "unknown")
                        if source in source_types:
                            source_types[source] += 1
                        else:
                            source_types["unknown"] += 1
                    elif "_chunk_" in doc_id:
                        total_chunks += 1

            return {
                "total_papers": main_papers,
                "total_chunks": total_chunks,
                "total_documents": len(results.get("ids", [])),
                "source_breakdown": source_types
            }
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
            return {
                "total_papers": 0,
                "total_chunks": 0,
                "total_documents": 0,
                "source_breakdown": {}
            }

    def clear(self):
        """Clear database"""
        try:
            self.collection.delete(where={})
            self.papers_cache.clear()
            return True
        except:
            return False


class ArxivLoader:
    """Simple arXiv paper loader"""

    @staticmethod
    def fetch_by_id(arxiv_id: str) -> Optional[Dict]:
        """Fetch paper from arXiv by ID"""
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
            print(f"❌ Error fetching arXiv paper: {e}")
            return None

    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes, max_pages: int = 30) -> str:
        """Enhanced PDF text extraction"""
        try:
            pdf_file = BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            text = ""
            total_pages = min(len(pdf_reader.pages), max_pages)

            for i in range(total_pages):
                try:
                    page_text = pdf_reader.pages[i].extract_text()
                    if page_text:
                        # Clean up text
                        lines = page_text.split('\n')
                        cleaned_lines = [line.strip() for line in lines if line.strip()]
                        page_text = ' '.join(cleaned_lines)
                        text += f"\n[Page {i + 1}]\n{page_text}\n"
                    else:
                        text += f"\n[Page {i + 1} - No extractable text]\n"
                except Exception as page_error:
                    text += f"\n[Page {i + 1} - Error: {str(page_error)}]\n"

            return text[:100000]  # Increased limit to 100k chars

        except Exception as e:
            return f"Error extracting PDF: {e}"

    @staticmethod
    def extract_text_from_html(html_content: str) -> str:
        """Extract text from HTML (for arXiv beta)"""
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:150000]  # Limit size

        except Exception as e:
            return f"Error extracting HTML: {e}"

    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) == 0:
            return []

        # Simple word-based chunking
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            if i >= len(words):
                break
            chunk = ' '.join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Only keep substantial chunks
                chunks.append(chunk)

            # Safety limit
            if len(chunks) >= 200:
                break

        print(f"📄 Created {len(chunks)} chunks")
        return chunks