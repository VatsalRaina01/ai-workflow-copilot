"""
Vector Store Service - ChromaDB Integration with Hybrid Search

Handles document storage, chunking, semantic search, BM25 keyword search,
and hybrid score fusion with optional re-ranking.
"""
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings
from app.services.github_models import github_models
from typing import List, Dict, Any
import uuid
import math
import logging

logger = logging.getLogger(__name__)


class BM25Index:
    """Simple BM25 keyword search index for hybrid retrieval."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avg_dl = 0
        self.df: Dict[str, int] = {}  # document frequency
        self.N = 0
    
    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents to the BM25 index."""
        for doc in docs:
            tokens = self._tokenize(doc["content"])
            self.documents.append({**doc, "_tokens": tokens})
            self.doc_lengths.append(len(tokens))
            
            seen = set()
            for token in tokens:
                if token not in seen:
                    self.df[token] = self.df.get(token, 0) + 1
                    seen.add(token)
        
        self.N = len(self.documents)
        self.avg_dl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25 scoring."""
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, doc in enumerate(self.documents):
            score = self._score_document(query_tokens, doc["_tokens"], self.doc_lengths[i])
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            doc = self.documents[idx].copy()
            doc.pop("_tokens", None)
            doc["bm25_score"] = score
            results.append(doc)
        
        return results
    
    def _score_document(self, query_tokens: List[str], doc_tokens: List[str], doc_len: int) -> float:
        """Compute BM25 score for a single document."""
        score = 0.0
        tf_map = {}
        for token in doc_tokens:
            tf_map[token] = tf_map.get(token, 0) + 1
        
        for token in query_tokens:
            if token not in tf_map:
                continue
            
            tf = tf_map[token]
            df = self.df.get(token, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercasing tokenizer."""
        import re
        return re.findall(r'\w+', text.lower())
    
    def clear(self):
        """Clear the BM25 index."""
        self.documents = []
        self.doc_lengths = []
        self.avg_dl = 0
        self.df = {}
        self.N = 0


class VectorStoreService:
    """
    Vector store with hybrid search combining:
    1. Semantic similarity (ChromaDB cosine distance)
    2. BM25 keyword matching
    3. Reciprocal Rank Fusion for score combination
    """
    
    def __init__(self):
        # Use PersistentClient for data survival across restarts
        try:
            self.client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
            logger.info(f"ChromaDB using persistent storage at: {settings.CHROMA_PERSIST_DIR}")
        except Exception:
            self.client = chromadb.Client()
            logger.warning("Falling back to ephemeral ChromaDB (in-memory)")
        
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # BM25 index for hybrid search
        self.bm25_index = BM25Index()
        
        # Rebuild BM25 index from existing ChromaDB data
        self._rebuild_bm25_index()
    
    def _rebuild_bm25_index(self):
        """Rebuild BM25 index from existing ChromaDB documents."""
        try:
            count = self.collection.count()
            if count > 0:
                results = self.collection.get(include=["documents", "metadatas"])
                docs = []
                for i, doc_text in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}
                    docs.append({"content": doc_text, "metadata": metadata})
                self.bm25_index.add_documents(docs)
                logger.info(f"Rebuilt BM25 index with {count} documents")
        except Exception as e:
            logger.warning(f"Failed to rebuild BM25 index: {e}")
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to both vector store and BM25 index after chunking."""
        doc_id = str(uuid.uuid4())
        chunks = self.text_splitter.split_text(content)
        
        if not chunks:
            return doc_id
        
        # Generate embeddings for all chunks
        embeddings = github_models.get_embeddings(chunks)
        
        # Prepare data for ChromaDB
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        # Add to BM25 index
        bm25_docs = [
            {"content": chunk, "metadata": metadatas[i]}
            for i, chunk in enumerate(chunks)
        ]
        self.bm25_index.add_documents(bm25_docs)
        
        logger.info(f"Added document {doc_id}: {len(chunks)} chunks indexed")
        return doc_id
    
    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Semantic-only search (backward compatible)."""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        if self.collection.count() == 0:
            return []
        
        query_embedding = github_models.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": round(1 - results["distances"][0][i], 3) if results["distances"] else 0
                })
        
        return formatted
    
    def hybrid_search(self, query: str, top_k: int = None, semantic_weight: float = 0.7) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic similarity and BM25 keyword matching.
        Uses Reciprocal Rank Fusion (RRF) for score combination.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0.0-1.0), BM25 gets (1 - weight)
        """
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        if self.collection.count() == 0:
            return []
        
        # Get semantic search results
        semantic_results = self.search(query, top_k=top_k * 2)
        
        # Get BM25 results
        bm25_results = self.bm25_index.search(query, top_k=top_k * 2)
        
        # Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}
        k = 60  # RRF constant
        
        for rank, doc in enumerate(semantic_results):
            doc_key = doc["content"][:100]  # Use content prefix as key
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + semantic_weight / (k + rank + 1)
            doc_map[doc_key] = doc
        
        for rank, doc in enumerate(bm25_results):
            doc_key = doc["content"][:100]
            rrf_scores[doc_key] = rrf_scores.get(doc_key, 0) + (1 - semantic_weight) / (k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
        
        # Sort by RRF score
        sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        results = []
        for key in sorted_keys[:top_k]:
            doc = doc_map[key].copy()
            doc["score"] = round(rrf_scores[key] * 100, 3)  # Scale for readability
            doc.pop("bm25_score", None)
            results.append(doc)
        
        return results
    
    def get_document_count(self) -> int:
        """Get the total number of chunks in the store."""
        return self.collection.count()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks of a document."""
        try:
            existing = self.collection.get(
                where={"doc_id": doc_id}
            )
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                logger.info(f"Deleted document {doc_id}: {len(existing['ids'])} chunks removed")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def clear_all(self):
        """Clear all documents from both stores."""
        self.client.delete_collection("documents")
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index.clear()
        logger.info("All documents cleared")


# Singleton instance
vector_store = VectorStoreService()
