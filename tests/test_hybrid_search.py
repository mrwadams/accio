"""Test cases for hybrid search functionality."""

import unittest
from typing import List, Dict, Any
import os
import json
from dotenv import load_dotenv
from google import genai
import psycopg2

from app.utils.embeddings import DocumentChunk, EmbeddingService, VectorStore
from app.utils.hybrid_search import HybridSearcher, SearchResult

# Load environment variables
load_dotenv()

class TestHybridSearch(unittest.TestCase):
    """Test cases for hybrid search implementation."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Database connection parameters from environment variables
        cls.db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": os.getenv("DB_PORT")
        }
        
        # Initialize Google GenAI client
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        cls.client = genai.Client(api_key=api_key)
        
        # Initialize services
        cls.embedding_service = EmbeddingService(cls.client)
        cls.vector_store = VectorStore(cls.db_params)
        cls.hybrid_searcher = HybridSearcher(cls.db_params)
        
        # Clear existing data from the database
        with psycopg2.connect(**cls.db_params) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE chunks")
                conn.commit()
        
        # Test data
        cls.test_chunks = [
            DocumentChunk(
                text="The quick brown fox jumps over the lazy dog.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 1},
                chunk_index=0
            ),
            DocumentChunk(
                text="Python is a versatile programming language used in AI and machine learning.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 1},
                chunk_index=1
            ),
            DocumentChunk(
                text="Neural networks are a fundamental concept in deep learning.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 2},
                chunk_index=2
            ),
            DocumentChunk(
                text="The lazy cat sleeps all day in the sun.",
                doc_id="test_doc_1",
                metadata={"source": "test", "page": 2},
                chunk_index=3
            )
        ]
        
        # Store test chunks
        embedded_chunks = cls.embedding_service.embed_chunks(cls.test_chunks)
        cls.vector_store.store_chunks(embedded_chunks)

    def test_vector_search(self):
        """Test vector similarity search component."""
        query = "lazy animals sleeping"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        results = self.hybrid_searcher._execute_vector_search(
            query_embedding=query_embedding,
            k=2
        )
        
        self.assertEqual(len(results), 2)
        # Verify lazy dog/cat texts are found
        self.assertTrue(
            any("lazy dog" in r["text"] for r in results) or
            any("lazy cat" in r["text"] for r in results)
        )

    def test_text_search(self):
        """Test full-text search component."""
        query = "programming Python AI"
        
        results = self.hybrid_searcher._execute_text_search(
            query=query,
            k=2
        )
        
        self.assertEqual(len(results), 1)  # Should find the Python programming text
        self.assertTrue("Python" in results[0]["text"])
        self.assertTrue("programming" in results[0]["text"].lower())

    def test_hybrid_search(self):
        """Test combined hybrid search."""
        query = "lazy animals"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Test with different weights
        for weight in [0.2, 0.5, 0.8]:
            results = self.hybrid_searcher.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                k=3,
                weight=weight
            )
            
            self.assertIsInstance(results, list)
            self.assertLessEqual(len(results), 3)
            
            for result in results:
                self.assertIsInstance(result, SearchResult)
                self.assertTrue(hasattr(result, 'combined_score'))
                self.assertTrue(hasattr(result, 'vector_score'))
                self.assertTrue(hasattr(result, 'text_score'))

    def test_score_threshold(self):
        """Test score threshold filtering."""
        query = "completely unrelated query"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        results = self.hybrid_searcher.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=4,
            weight=0.5,
            score_threshold=0.8  # High threshold
        )
        
        # Should return fewer or no results due to high threshold
        self.assertLess(len(results), 4)

    def test_rrf_scoring(self):
        """Test Reciprocal Rank Fusion scoring."""
        # Create mock results
        vector_results = [
            {"doc_id": "1", "chunk_index": 0, "text": "test1", "metadata": {}, "similarity": 0.9},
            {"doc_id": "2", "chunk_index": 0, "text": "test2", "metadata": {}, "similarity": 0.8}
        ]
        
        text_results = [
            {"doc_id": "2", "chunk_index": 0, "text": "test2", "metadata": {}, "text_score": 0.9},
            {"doc_id": "3", "chunk_index": 0, "text": "test3", "metadata": {}, "text_score": 0.8}
        ]
        
        results = self.hybrid_searcher._compute_rrf_scores(
            vector_results=vector_results,
            text_results=text_results,
            k=3,
            weight=0.5
        )
        
        self.assertEqual(len(results), 3)
        # Doc 2 should be ranked highly as it appears in both result sets
        doc2_result = next((r for r in results if r.doc_id == "2"), None)
        self.assertIsNotNone(doc2_result)
        self.assertEqual(doc2_result.doc_id, "2")

    def test_empty_query(self):
        """Test behavior with empty query string."""
        query = ""
        query_embedding = self.embedding_service.generate_embedding(query)
        
        results = self.hybrid_searcher.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=2,
            weight=0.5
        )
        
        # Should handle empty query gracefully
        self.assertIsInstance(results, list)
        
    def test_metadata_preservation(self):
        """Test that metadata is preserved through the search pipeline."""
        query = "neural networks deep learning"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        results = self.hybrid_searcher.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=1,
            weight=0.5
        )
        
        self.assertEqual(len(results), 1)
        result = results[0]
        
        # Verify metadata fields are preserved
        self.assertIn("source", result.metadata)
        self.assertIn("page", result.metadata)
        self.assertEqual(result.metadata["source"], "test")
        
    def test_performance_limits(self):
        """Test search performance with larger k values."""
        query = "programming"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Test with a larger k value
        k_large = 100
        results = self.hybrid_searcher.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            k=k_large,
            weight=0.5
        )
        
        # Should not return more results than available documents
        self.assertLessEqual(len(results), len(self.test_chunks))
        
    def test_invalid_weight(self):
        """Test handling of invalid weight values."""
        query = "test query"
        query_embedding = self.embedding_service.generate_embedding(query)
        
        # Test with invalid weight values
        invalid_weights = [-0.1, 1.1]
        for weight in invalid_weights:
            with self.assertRaises(ValueError):
                self.hybrid_searcher.hybrid_search(
                    query=query,
                    query_embedding=query_embedding,
                    k=2,
                    weight=weight
                )

    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        cls.vector_store.delete_document("test_doc_1") 