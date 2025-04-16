"""
Embeddings and storage utilities for the RAG chatbot.
Handles text embedding generation and storage in PostgreSQL with pgvector.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass
import random
import uuid

import psycopg2
from psycopg2.extras import execute_values
from google import genai
from google.genai import types

from .hybrid_search import HybridSearcher, SearchResult

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a chunk of text with its metadata."""
    text: str
    doc_id: str
    metadata: Dict[str, Any]
    chunk_index: int

@dataclass
class EmbeddedChunk(DocumentChunk):
    """A document chunk with its embedding vector."""
    embedding: List[float]

class EmbeddingService:
    """Service for generating embeddings using Google's text-embedding-004 model."""

    def __init__(self, client: genai.Client):
        """Initialize the embedding service.
        
        Args:
            client: Google GenAI client
        """
        self.client = client
        self.model_name = "text-embedding-004"
        self.embedding_dimension = 768  # Fixed dimension for text-embedding-004
        self.batch_size = 5  # Process 5 texts at a time to stay within rate limits
        self.min_delay = 0.1  # Minimum delay between API calls in seconds
        self.max_retries = 3  # Maximum number of retries for rate-limited requests

    def _retry_with_exponential_backoff(self, func, *args, **kwargs):
        """Execute a function with exponential backoff retry logic."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if "RESOURCE_EXHAUSTED" not in str(e) or attempt == self.max_retries - 1:
                    raise
                
                delay = min(1 + (2 ** attempt) + random.random(), 8)
                logger.warning(f"Rate limit hit, retrying in {delay:.2f} seconds...")
                time.sleep(delay)

    def _batch_embed_content(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts with rate limiting."""
        try:
            result = self._retry_with_exponential_backoff(
                self.client.models.embed_content,
                model=self.model_name,
                contents=texts
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as e:
            logger.error(f"Error in batch embedding: {str(e)}")
            return [[0.0] * self.embedding_dimension for _ in texts]

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and rate limiting."""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = self._batch_embed_content(batch)
            all_embeddings.extend(batch_embeddings)
            
            if i + self.batch_size < len(texts):
                time.sleep(self.min_delay)
        
        return all_embeddings

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        try:
            result = self._retry_with_exponential_backoff(
                self.client.models.embed_content,
                model=self.model_name,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * self.embedding_dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (LangChain interface)."""
        return self.generate_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Generate embedding for a single query text (LangChain interface)."""
        return self.generate_embedding(text)

    def embed_chunks(self, chunks: List[DocumentChunk]) -> List[EmbeddedChunk]:
        """Generate embeddings for multiple document chunks with batching."""
        texts = [chunk.text for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                text=chunk.text,
                doc_id=chunk.doc_id,
                metadata=chunk.metadata,
                chunk_index=chunk.chunk_index,
                embedding=embedding
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks

class VectorStore:
    """Handles storage and retrieval of document chunks and their embeddings in PostgreSQL."""
    
    def __init__(self, db_params: Dict[str, str]):
        """Initialize the vector store.
        
        Args:
            db_params: Database connection parameters
        """
        self.db_params = db_params

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Get a database connection."""
        return psycopg2.connect(**self.db_params)

    def list_documents(self, team_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List documents, optionally filtered by team_id.
        
        Args:
            team_id: Optional team ID to filter documents. If None, lists all documents (admin only).
            
        Returns:
            List of document dictionaries containing doc_id, filename, and metadata.
        """
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            if team_id is None:
                # Admin view - get all documents
                query = """
                    SELECT doc_id, filename, metadata, team_id, created_at, updated_at
                    FROM app_schema.documents
                    ORDER BY created_at DESC
                """
                cur.execute(query)
            else:
                # Team view - get only team's documents
                query = """
                    SELECT doc_id, filename, metadata, team_id, created_at, updated_at
                    FROM app_schema.documents
                    WHERE team_id = %s
                    ORDER BY created_at DESC
                """
                cur.execute(query, (team_id,))
            
            results = cur.fetchall()
            
            return [{
                'doc_id': r[0],
                'filename': r[1],
                'metadata': r[2],
                'team_id': r[3],
                'created_at': r[4],
                'updated_at': r[5]
            } for r in results]
            
        finally:
            cur.close()
            conn.close()

    def store_chunks(self, chunks: List[EmbeddedChunk], team_id: str):
        """Store document chunks with their embeddings."""
        if not chunks:
            return
        
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            # First ensure the document exists and belongs to the team
            doc_id = chunks[0].doc_id
            cur.execute("""
                SELECT team_id FROM app_schema.documents 
                WHERE doc_id = %s AND team_id = %s
            """, (doc_id, team_id))
            
            if not cur.fetchone():
                raise ValueError(f"Document {doc_id} not found or not owned by team {team_id}")
            
            # Store chunks
            chunk_data = [
                (str(uuid.uuid4()), chunk.doc_id, chunk.text, chunk.embedding)
                for chunk in chunks
            ]
            
            execute_values(cur, """
                INSERT INTO app_schema.chunks (chunk_id, doc_id, text, embedding)
                VALUES %s
            """, chunk_data)
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    def similarity_search(
        self,
        query_embedding: List[float],
        team_id: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
        admin_override: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks, filtered by team_id unless admin_override is True."""
        conn = self._get_connection()
        cur = conn.cursor()
        
        try:
            if admin_override:
                query = """
                    SELECT 
                        c.chunk_id,
                        c.doc_id,
                        d.filename,
                        c.text,
                        d.metadata,
                        d.team_id,
                        1 - (c.embedding <-> %s::vector) as similarity
                    FROM app_schema.chunks c
                    JOIN app_schema.documents d ON c.doc_id = d.doc_id
                    WHERE 1 - (c.embedding <-> %s::vector) > %s
                    ORDER BY c.embedding <-> %s::vector
                    LIMIT %s
                """
                params = (query_embedding, query_embedding, score_threshold or 0.0, query_embedding, k)
            else:
                query = """
                    SELECT 
                        c.chunk_id,
                        c.doc_id,
                        d.filename,
                        c.text,
                        d.metadata,
                        d.team_id,
                        1 - (c.embedding <-> %s::vector) as similarity
                    FROM app_schema.chunks c
                    JOIN app_schema.documents d ON c.doc_id = d.doc_id
                    WHERE d.team_id = %s AND 1 - (c.embedding <-> %s::vector) > %s
                    ORDER BY c.embedding <-> %s::vector
                    LIMIT %s
                """
                params = (query_embedding, team_id, query_embedding, score_threshold or 0.0, query_embedding, k)
            
            cur.execute(query, params)
            results = cur.fetchall()
            
            return [{
                'chunk_id': r[0],
                'doc_id': r[1],
                'filename': r[2],
                'text': r[3],
                'metadata': r[4],
                'team_id': r[5],
                'similarity': r[6]
            } for r in results]
            
        finally:
            cur.close()
            conn.close()

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        team_id: str,
        k: int = 4,
        weight: float = 0.5,
        score_threshold: Optional[float] = None,
        admin_override: bool = False
    ) -> List[SearchResult]:
        """Perform hybrid search combining vector similarity and keyword matching."""
        searcher = HybridSearcher(self.db_params)
        return searcher.search(
            query=query,
            query_embedding=query_embedding,
            team_id=team_id,
            k=k,
            weight=weight,
            score_threshold=score_threshold,
            admin_override=admin_override
        ) 