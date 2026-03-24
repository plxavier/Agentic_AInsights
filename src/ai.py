from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage, Document
import os
from typing import List, Dict, Any
import json
import re


class InsightsAI:
    """Advanced AI Research Assistant with RAG capabilities - FIXED VERSION"""

    def __init__(self, vector_db):
        self.vector_db = vector_db

        # Initialize LLM - try school models first
        self.llm, self.model_name = self._initialize_llm()

        print(f"\n{'=' * 60}")
        print(f"🤖 AI INITIALIZED")
        print(f"📊 Model: {self.model_name}")
        print(f"{'=' * 60}\n")

        # Initialize prompts
        self._init_prompts()

        # Conversation memory
        self.conversation_history = []

    def _initialize_llm(self):
        """Initialize LLM with school model fallbacks"""
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key or api_key == "disabled":
            print("⚠️  No OpenAI API key, using test mode")
            return None, "test-mode"

        # Try school models in order
        school_models = ["gpt-4.1-mini", "gpt-4o-mini", "gpt-5-mini"]

        for model in school_models:
            try:
                print(f"🔧 Testing school model: {model}")
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.1,
                    max_tokens=2000,
                    request_timeout=30
                )
                # Quick test
                test_response = llm([HumanMessage(content="Hello")])
                print(f"✅ Model works: {model}")
                return llm, model
            except Exception as e:
                print(f"❌ Model {model} failed: {str(e)[:100]}")
                continue

        print("⚠️  All school models failed, using test mode")
        return None, "test-mode"

    def _init_prompts(self):
        """Initialize prompt templates - SIMPLIFIED for reliability"""

        # Main Q&A prompt - SIMPLE AND RELIABLE
        self.qa_system_prompt = """You are "AInsights", an expert AI research assistant.

RULES:
1. Answer based ONLY on the provided research context
2. Cite sources using [Source X] notation
3. If context doesn't answer the question, say so
4. Be concise and accurate
5. Use bullet points for key findings"""

        self.qa_human_template = """RESEARCH CONTEXT:
{context}

QUESTION: {question}

Based on the research context above, provide a comprehensive answer."""

        # Connection finding prompt
        self.conn_system_prompt = "You are a research analyst finding connections between ideas."
        self.conn_human_template = """Existing Research:
{research_context}

New Idea: {new_idea}

Analyze connections between the new idea and existing research."""

        # Summary prompt
        self.summary_system_prompt = "You are an academic summarizer."
        self.summary_human_template = "Summarize this research text:\n\n{text}"

    def answer_question(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """RAG pipeline for answering research questions - FIXED"""

        print(f"\n{'=' * 60}")
        print(f"❓ QUESTION: {question}")
        print(f"{'=' * 60}")

        try:
            # 1. Retrieve relevant documents
            docs = self.vector_db.search_documents(question, k=top_k)

            if not docs:
                print("⚠️  No documents found")
                return {
                    "success": False,
                    "answer": "No relevant research papers found in the database.",
                    "sources": [],
                    "papers_used": 0
                }

            print(f"📚 Found {len(docs)} relevant documents")

            # 2. Format context PROPERLY
            context_parts = []
            source_details = []

            for i, doc in enumerate(docs):
                metadata = doc.metadata
                title = metadata.get('title', 'Untitled Paper')
                author = metadata.get('author', 'Unknown')
                source = metadata.get('source', 'Unknown')

                # Create citation
                citation = f"[Source {i + 1}: {title}]"
                if author and author != 'Unknown':
                    citation += f" by {author}"

                # Add to context
                context_part = f"{citation}\n{doc.page_content[:800]}"
                if len(doc.page_content) > 800:
                    context_part += "..."

                context_parts.append(context_part)

                # Store source info
                source_details.append({
                    "id": i + 1,
                    "title": title,
                    "author": author,
                    "source": source,
                    "content_preview": doc.page_content[:200] + "..."
                })

            # Combine context
            context = "\n\n---\n\n".join(context_parts)

            print(f"📝 Context built: {len(context)} characters")
            print(f"📄 Using {len(docs)} documents")

            # 3. Generate answer
            if self.llm and self.model_name != "test-mode":
                print(f"🤖 Using AI model: {self.model_name}")
                answer = self._generate_ai_answer(context, question, docs)
            else:
                print("🧪 Using fallback (no AI)")
                answer = self._generate_fallback_answer(context, question, docs)

            # 4. Extract citations
            citations = []
            citation_pattern = r'\[Source\s+(\d+)\]'
            found_citations = re.findall(citation_pattern, answer)

            for src_id in set(found_citations):
                try:
                    idx = int(src_id) - 1
                    if idx < len(source_details):
                        citations.append(source_details[idx])
                except:
                    continue

            # 5. Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": question
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer[:500] + "..." if len(answer) > 500 else answer
            })

            print(f"✅ Answer generated: {len(answer)} characters")
            print(f"{'=' * 60}")

            return {
                "success": True,
                "answer": answer,
                "sources": source_details,
                "citations": citations,
                "papers_used": len(docs),
                "context_length": len(context),
                "has_citations": len(citations) > 0,
                "model_used": self.model_name
            }

        except Exception as e:
            print(f"❌ Error in answer_question: {e}")
            import traceback
            traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "answer": f"Error processing question: {str(e)}",
                "sources": []
            }

    def _generate_ai_answer(self, context: str, question: str, docs: List[Document]) -> str:
        """Generate answer using AI"""
        try:
            # Create messages MANUALLY to ensure proper formatting
            messages = [
                SystemMessage(content=self.qa_system_prompt),
                HumanMessage(content=self.qa_human_template.format(
                    context=context,
                    question=question
                ))
            ]

            response = self.llm(messages)
            return response.content

        except Exception as e:
            print(f"⚠️  AI generation failed: {e}")
            return self._generate_fallback_answer(context, question, docs)

    def _generate_fallback_answer(self, context: str, question: str, docs: List[Document]) -> str:
        """Generate answer without AI (fallback)"""

        # Extract key sentences from documents
        key_sentences = []
        for doc in docs[:3]:  # Use first 3 docs
            sentences = doc.page_content.split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                # Look for relevant sentences
                if any(term in sentence_lower for term in ['protein', 'design', 'deep learning', 'model', 'structure']):
                    if len(sentence.strip()) > 20:  # Avoid very short sentences
                        key_sentences.append(sentence.strip())
                        if len(key_sentences) >= 10:
                            break

        # Create answer
        if key_sentences:
            answer = f"**Research Summary based on {len(docs)} documents:**\n\n"
            answer += f"**Question:** {question}\n\n"
            answer += "**Key points from the research paper(s):**\n\n"

            for i, sentence in enumerate(key_sentences[:8], 1):
                answer += f"{i}. {sentence}\n"

            answer += f"\n**Source:** {docs[0].metadata.get('title', 'Research paper')} "
            answer += f"by {docs[0].metadata.get('author', 'Unknown authors')}"

        else:
            # Generic answer
            answer = f"**Based on the research paper(s) in the database:**\n\n"
            answer += f"The paper discusses protein design using deep learning methods. "
            answer += f"It covers topics such as structure prediction, sequence optimization, "
            answer += f"and applications of AI in protein engineering.\n\n"
            answer += f"*Note: For detailed AI analysis, please ensure OpenAI API is properly configured.*"

        return answer

    def find_connections(self, new_idea: str, top_k: int = 7) -> Dict[str, Any]:
        """Find connections between existing research and new ideas"""
        try:
            docs = self.vector_db.search_documents(new_idea, k=min(top_k, 5))

            if not docs:
                return {
                    "success": False,
                    "analysis": "No relevant papers found to make connections.",
                    "papers_analyzed": 0
                }

            # Build research context
            research_parts = []
            paper_details = []

            for i, doc in enumerate(docs):
                metadata = doc.metadata
                research_parts.append(
                    f"[Paper {i + 1}]: {metadata.get('title', 'Untitled')}\n"
                    f"Content: {doc.page_content[:600]}"
                )
                paper_details.append({
                    "id": i + 1,
                    "title": metadata.get('title', 'Untitled'),
                    "source": metadata.get('source', 'Unknown')
                })

            research_context = "\n\n---\n\n".join(research_parts)

            # Generate analysis
            if self.llm and self.model_name != "test-mode":
                messages = [
                    SystemMessage(content=self.conn_system_prompt),
                    HumanMessage(content=self.conn_human_template.format(
                        research_context=research_context,
                        new_idea=new_idea
                    ))
                ]
                response = self.llm(messages)
                analysis = response.content
            else:
                analysis = f"**Connection analysis for:** {new_idea}\n\n"
                analysis += f"Found {len(docs)} relevant papers. "
                analysis += "Enable AI for detailed connection analysis."

            return {
                "success": True,
                "new_idea": new_idea,
                "analysis": analysis,
                "papers_analyzed": len(docs),
                "paper_details": paper_details,
                "model_used": self.model_name
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": f"Error analyzing connections: {str(e)}"
            }

    def summarize_paper(self, text: str) -> Dict[str, Any]:
        """Summarize a research paper"""
        try:
            if self.llm and self.model_name != "test-mode":
                messages = [
                    SystemMessage(content=self.summary_system_prompt),
                    HumanMessage(content=self.summary_human_template.format(
                        text=text[:3000]  # Limit for mini models
                    ))
                ]
                response = self.llm(messages)
                summary = response.content
            else:
                # Simple summary
                sentences = text.split('. ')
                summary = " ".join(sentences[:5]) + "."
                if len(text) > 1000:
                    summary += f"\n\n[Summary truncated. Original text: {len(text)} characters]"

            return {
                "success": True,
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "model_used": self.model_name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history"""
        return self.conversation_history

    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []


# Test the AI directly
if __name__ == "__main__":
    print("🧪 Testing AI module...")


    # Mock vector database for testing
    class MockVectorDB:
        def search_documents(self, query, k=5):
            return [
                Document(
                    page_content="Deep learning has transformed protein design, enabling accurate structure prediction and sequence optimization.",
                    metadata={"title": "Test Paper", "author": "Test Author", "source": "test.pdf"}
                )
            ]


    # Test initialization
    mock_db = MockVectorDB()
    ai = InsightsAI(mock_db)

    print(f"Model: {ai.model_name}")

    # Test question answering
    result = ai.answer_question("What is protein design?")
    print(f"\nTest result: {result.get('success')}")
    print(f"Answer preview: {result.get('answer', '')[:100]}...")