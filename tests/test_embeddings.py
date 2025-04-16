"""Tests for the embeddings and storage functionality."""

import os
import unittest
# Remove logging import if no longer needed, or adjust level
# import logging 
from typing import Dict, Any

from dotenv import load_dotenv
from google import genai

from app.utils.embeddings import EmbeddingService, VectorStore, DocumentChunk

# Remove logging setup
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TestEmbeddings(unittest.TestCase):
    """Test cases for embedding generation and storage."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Initialize Google GenAI client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        cls.client = genai.Client(api_key=api_key)
        
        # Initialize services
        cls.embedding_service = EmbeddingService(cls.client)
        
        # Database connection parameters
        cls.db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }
        
        cls.vector_store = VectorStore(cls.db_params)
    
    def test_generate_embedding(self):
        """Test generating a single embedding."""
        text = "This is a test document."
        embedding = self.embedding_service.generate_embedding(text)
        
        # Verify embedding format
        self.assertIsInstance(embedding, list)
        self.assertEqual(len(embedding), 768)  # text-embedding-004 dimension
        self.assertIsInstance(embedding[0], float)
    
    def test_embed_and_store_chunks(self):
        """Test embedding generation and storage for multiple chunks."""
        # Create test chunks
        chunks = [
            DocumentChunk(
                text="This is the first test chunk.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 1},
                chunk_index=0
            ),
            DocumentChunk(
                text="This is the second test chunk.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 1},
                chunk_index=1
            )
        ]
        
        # Generate embeddings
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        
        # Verify embedded chunks
        self.assertEqual(len(embedded_chunks), 2)
        self.assertEqual(len(embedded_chunks[0].embedding), 768)
        
        # Store chunks
        self.vector_store.store_chunks(embedded_chunks)
        
        # Clean up
        self.vector_store.delete_document("test_doc_1")

    def test_similarity_search(self):
        """Test vector similarity search functionality."""
        # Create and store test chunks
        chunks = [
            DocumentChunk(
                text="The quick brown fox jumps over the lazy dog.",
                doc_id="test_doc_2",
                metadata={"source": "test", "page": 1},
                chunk_index=0
            ),
            DocumentChunk(
                text="A lazy dog sleeps in the sun.",
                doc_id="test_doc_2",
                metadata={"source": "test", "page": 1},
                chunk_index=1
            ),
            DocumentChunk(
                text="Python is a great programming language.",
                doc_id="test_doc_2",
                metadata={"source": "test", "page": 2},
                chunk_index=2
            )
        ]
        
        # Generate embeddings and store
        embedded_chunks = self.embedding_service.embed_chunks(chunks)
        # logger.debug(f"Generated {len(embedded_chunks)} embedded chunks") # Removed
        
        self.vector_store.store_chunks(embedded_chunks)
        # logger.debug("Stored chunks in database") # Removed
        
        # Search using the embedding of a similar query
        query = "A dog that is lazy"
        query_embedding = self.embedding_service.generate_embedding(query)
        # logger.debug(f"Generated query embedding for: {query}") # Removed
        
        # Search with default parameters (including probes)
        results = self.vector_store.similarity_search(query_embedding, k=4, probes=5)
        # logger.debug(f"Got {len(results)} results from similarity search (k=4, probes=5)") # Removed
        
        # Verify results
        self.assertGreater(len(results), 0, "Expected at least one search result")
        self.assertIn("text", results[0])
        self.assertIn("similarity", results[0])
        self.assertIn("doc_id", results[0])
        self.assertIn("metadata", results[0])
        self.assertIn("chunk_index", results[0])
        
        # Verify ordering - chunks about dogs should be first
        self.assertTrue(
            any("dog" in result["text"].lower() for result in results[:2]),
            "Expected results about dogs to be ranked highly"
        )
        
        # Test with similarity threshold
        threshold_results = self.vector_store.similarity_search(
            query_embedding,
            k=4,
            probes=10, # Use more probes when filtering
            score_threshold=0.5
        )
        # logger.debug(f"Got {len(threshold_results)} results with threshold 0.5 (k=4, probes=10)") # Removed
        
        self.assertTrue(
            all(r["similarity"] >= 0.5 for r in threshold_results),
            "All results should meet the similarity threshold"
        )
        
        # Clean up
        self.vector_store.delete_document("test_doc_2")
        # logger.debug("Cleaned up test document") # Removed

if __name__ == "__main__":
    unittest.main() 