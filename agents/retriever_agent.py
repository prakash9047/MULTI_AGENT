import dotenv
dotenv.load_dotenv(dotenv_path="config/.env")
from typing import Dict, List, Optional
import numpy as np
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", message=".*pydantic.*")

# Handle optional dependencies with fallbacks
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # Removed warning print - will show in agent status instead

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Removed warning print - will show in agent status instead

# Simplified text splitter without LangChain dependencies
class SimpleTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end > len(text):
                end = len(text)
            
            # Try to break at word boundaries
            if end < len(text):
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        
        return chunks

# Create dummy FAISS class for fallback
class MockVectorStore:
    def __init__(self, texts):
        self.texts = texts
        
    def similarity_search(self, query, k=4):
        # Simple keyword-based search as fallback
        query_words = set(query.lower().split())
        scored_texts = []
        
        for text in self.texts:
            text_words = set(text.lower().split())
            score = len(query_words.intersection(text_words))
            scored_texts.append((score, text))
        
        # Sort by score and return top k
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        return [{"page_content": text, "metadata": {"score": score}} for score, text in scored_texts[:min(k, len(scored_texts))]]

    def add_texts(self, texts):
        """Add texts to the mock vector store."""
        self.texts.extend(texts)

# Simple FAISS-based vector store without LangChain
class SimpleVectorStore:
    def __init__(self, texts, embeddings):
        self.texts = texts
        self.embeddings = embeddings
        self.index = None
        self._build_index()
        
    def _build_index(self):
        """Build FAISS index."""
        try:
            if FAISS_AVAILABLE:
                # Get embeddings for texts
                text_embeddings = self.embeddings.embed_documents(self.texts)
                embeddings_array = np.array(text_embeddings).astype('float32')
                
                # Create FAISS index
                dimension = embeddings_array.shape[1]
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
                self.index.add(embeddings_array)
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None
            
    def similarity_search(self, query, k=4):
        """Search for similar documents."""
        try:
            if self.index is None or not FAISS_AVAILABLE:
                # Fallback to keyword search
                return self._keyword_search(query, k)
                
            # Get query embedding
            query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
            
            # Search
            scores, indices = self.index.search(query_embedding, min(k, len(self.texts)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.texts):
                    results.append({
                        "page_content": self.texts[idx],
                        "metadata": {"score": float(score), "index": int(idx)}
                    })
            return results
        except Exception as e:
            return self._keyword_search(query, k)
            
    def _keyword_search(self, query, k):
        """Fallback keyword search."""
        query_words = set(query.lower().split())
        scored_texts = []
        
        for i, text in enumerate(self.texts):
            text_words = set(text.lower().split())
            score = len(query_words.intersection(text_words))
            scored_texts.append((score, text, i))
        
        scored_texts.sort(key=lambda x: x[0], reverse=True)
        return [{"page_content": text, "metadata": {"score": score, "index": idx}} 
                for score, text, idx in scored_texts[:min(k, len(scored_texts))]]
    
    def add_texts(self, texts):
        """Add new texts to the vector store."""
        self.texts.extend(texts)
        self._build_index()  # Rebuild index with new texts

# Create a mock embeddings class for fallback
class MockEmbeddings:
    def __init__(self, **kwargs):
        pass
        
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]  # Return fake embeddings
        
    def embed_query(self, text):
        return [0.0] * 384  # Return fake embedding

# Simple tool class to replace LangChain Tool
class Tool:
    def __init__(self, name: str, func, description: str):
        self.name = name
        self.func = func
        self.description = description

from config.settings import settings
from agents.agent_helpers import SimpleAgent

class RetrieverAgent:
    def __init__(self):
        # Initialize embeddings based on availability
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
                self.embeddings = self
            except:
                self.embeddings = MockEmbeddings()
        else:
            self.embeddings = MockEmbeddings()
            
        self.text_splitter = SimpleTextSplitter(
            chunk_size=getattr(settings, 'CHUNK_SIZE', 1000),
            chunk_overlap=getattr(settings, 'CHUNK_OVERLAP', 200)
        )
        self.vector_store = None
        self.agent = self._create_agent()
        
        # Agent status for debugging
        self.status = {
            "faiss": FAISS_AVAILABLE,
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
            "embeddings": "real" if SENTENCE_TRANSFORMERS_AVAILABLE else "mock"
        }

    def embed_documents(self, texts):
        """Embed documents using sentence transformers or fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self, 'model'):
            try:
                return self.model.encode(texts).tolist()
            except:
                return [[0.0] * 384 for _ in texts]
        return [[0.0] * 384 for _ in texts]
        
    def embed_query(self, text):
        """Embed query using sentence transformers or fallback."""
        if SENTENCE_TRANSFORMERS_AVAILABLE and hasattr(self, 'model'):
            try:
                return self.model.encode([text])[0].tolist()
            except:
                return [0.0] * 384
        return [0.0] * 384
        
    def _create_agent(self):
        """Create the agent for retrieval operations."""
        return SimpleAgent(
            role="Information Retriever",
            goal="Retrieve relevant financial information from the vector store",
            backstory="Expert in semantic search and information retrieval",
            tools=[
                Tool(
                    name="search_documents",
                    func=self.search_documents,
                    description="Search for relevant documents in the vector store"
                ),
                Tool(
                    name="add_documents",
                    func=self.add_documents,
                    description="Add new documents to the vector store"
                )            ]
        )
        
    def initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Use custom FAISS implementation or mock version
            if FAISS_AVAILABLE:
                self.vector_store = SimpleVectorStore([
                    "Initial document", 
                    "Financial markets are complex systems", 
                    "Stocks represent ownership in companies"
                ], self.embeddings)
            else:
                # Use mock vector store
                self.vector_store = MockVectorStore([
                    "Initial document", 
                    "Financial markets are complex systems", 
                    "Stocks represent ownership in companies"
                ])
            return {"status": "success", "method": "real" if FAISS_AVAILABLE else "mock"}
        except Exception as e:
            # Fallback to mock even if FAISS is available but fails
            self.vector_store = MockVectorStore([
                "Initial document", 
                "Financial markets are complex systems", 
                "Stocks represent ownership in companies"
            ])
            return {"status": "fallback", "error": str(e)}

    def add_documents(self, documents: List[str]) -> Dict:
        """Add documents to the vector store."""
        try:
            if not self.vector_store:
                self.initialize_vector_store()
            
            # Handle both single document string and list of documents
            if isinstance(documents, str):
                texts = self.text_splitter.split_text(documents)
            else:
                texts = []
                for doc in documents:
                    texts.extend(self.text_splitter.split_text(doc))
            
            self.vector_store.add_texts(texts)
            return {
                "status": "success",
                "documents_added": len(texts)
            }
        except Exception as e:
            return {"error": str(e)}  
        
    def search_documents(self, query: str, k: int = None) -> Dict:
        """Search for relevant documents in the vector store."""
        try:
            if not self.vector_store:
                self.initialize_vector_store()
            
            k = k or getattr(settings, 'TOP_K_RESULTS', 4)
            results = self.vector_store.similarity_search(query, k=k)
            
            return {
                "results": [
                    {
                        "content": result["page_content"],
                        "score": result["metadata"].get("score", 0.0),
                        "metadata": result["metadata"]
                    }
                    for result in results
                ]
            }
        except Exception as e:
            return {"error": str(e)}
            
    def get_relevant_context(self, query: str) -> Dict:
        """Get relevant context for a given query."""
        try:
            # Initialize vector store if it doesn't exist
            if not self.vector_store:
                try:
                    # Try to load from disk first
                    load_result = self.load_vector_store()
                    if "error" in load_result:
                        # If loading fails, initialize a new one
                        self.initialize_vector_store()
                        # Add some default documents if needed
                        self.add_documents(["Financial markets refer to any marketplace where trading of securities occurs.",
                                          "Stocks represent ownership shares in a corporation.",
                                          "Bonds are debt securities that represent loans made by an investor to a borrower."])
                except Exception as init_error:
                    print(f"Vector store initialization error: {str(init_error)}")
                    # Return default placeholder context if all fails
                    return {
                        "results": [],
                        "query": query
                    }
            
            # Search for documents
            results = self.search_documents(query)
            if "error" in results:
                print(f"Document search error: {results['error']}")
                return {"results": [], "query": query}
            
            # Filter results based on confidence threshold or use all if none meet threshold
            filtered_results = [
                result for result in results.get("results", [])
                if result.get("score", 0) >= settings.CONFIDENCE_THRESHOLD
            ]
            
            # If no results meet threshold, use all results
            if not filtered_results and results.get("results"):
                filtered_results = results.get("results", [])
            
            confidence = 0
            if results.get("results") and len(results.get("results")) > 0:
                confidence = len(filtered_results) / len(results.get("results", []))
            
            return {
                "context": filtered_results,
                "confidence": confidence,
                "query": query
            }
        except Exception as e:
            print(f"Retriever context error: {str(e)}")
            return {"results": [], "query": query}

    def save_vector_store(self, path: str = None):
        """Save the vector store to disk."""
        try:
            if not self.vector_store:
                return {"error": "Vector store not initialized"}
            
            path = path or settings.FAISS_INDEX_PATH
            self.vector_store.save_local(path)
            return {"status": "success"}
        except Exception as e:
            return {"error": str(e)}

    def load_vector_store(self, path: str = None):
        """Load the vector store from disk."""
        try:
            if FAISS_AVAILABLE:
                # Use our custom FAISS implementation instead of LangChain
                path = path or getattr(settings, 'FAISS_INDEX_PATH', './vector_store')
                try:
                    # Try to load existing index (simplified version)
                    import pickle
                    import os
                    if os.path.exists(f"{path}/texts.pkl"):
                        with open(f"{path}/texts.pkl", 'rb') as f:
                            texts = pickle.load(f)
                        self.vector_store = SimpleVectorStore(texts, self.embeddings)
                    else:
                        # Create new store if no saved data
                        self.initialize_vector_store()
                except:
                    # Fallback to new initialization
                    self.initialize_vector_store()
                return {"status": "success", "method": "custom"}
            else:
                # Initialize mock vector store instead
                self.vector_store = MockVectorStore([
                    "Mock financial data",
                    "Market analysis information",
                    "Investment research content"
                ])
                return {"status": "mock", "note": "FAISS not available, using mock vector store"}
        except Exception as e:
            # Fallback to mock on any error
            self.vector_store = MockVectorStore([
                "Mock financial data",
                "Market analysis information", 
                "Investment research content"
            ])
            return {"status": "fallback", "error": str(e)} 