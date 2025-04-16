"""
Hybrid search implementation combining vector similarity search with PostgreSQL full-text search.
Uses Reciprocal Rank Fusion (RRF) for score combination.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
import psycopg2
from psycopg2.extras import execute_values

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a single search result with combined ranking."""
    chunk_id: str
    doc_id: str
    filename: str
    text: str
    metadata: Dict[str, Any]
    team_id: str
    vector_score: float
    text_score: float
    combined_score: float

class HybridSearcher:
    """Implements hybrid search combining vector similarity and full-text search."""
    
    def __init__(self, db_params: Dict[str, str]):
        """Initialize the hybrid searcher.
        
        Args:
            db_params: Database connection parameters
        """
        self.db_params = db_params
        self._init_fts()
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_params)
    
    def _init_fts(self):
        """Initialize full-text search configuration."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Add tsvector column if it doesn't exist
                cur.execute("""
                    ALTER TABLE app_schema.chunks 
                    ADD COLUMN IF NOT EXISTS textsearch tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
                """)
                
                # Create GIN index for full-text search if it doesn't exist
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS chunks_textsearch_idx 
                    ON app_schema.chunks 
                    USING gin(textsearch)
                """)
                
                conn.commit()

    def _execute_vector_search(
        self,
        query_embedding: List[float],
        team_id: str,
        k: int = 10,
        score_threshold: Optional[float] = None,
        admin_override: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute vector similarity search with team filtering.
        
        Args:
            query_embedding: Query vector
            team_id: Team ID to filter results
            k: Number of results to return
            score_threshold: Optional similarity threshold
            admin_override: Whether to ignore team filtering
            
        Returns:
            List of results with similarity scores
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Base query with team join
                query = """
                    SELECT 
                        c.chunk_id,
                        c.doc_id,
                        d.filename,
                        c.text,
                        d.metadata,
                        d.team_id,
                        1 - (c.embedding <=> %s::vector) as similarity
                    FROM app_schema.chunks c
                    JOIN app_schema.documents d ON c.doc_id = d.doc_id
                    WHERE 1=1
                """
                
                params = [query_embedding]
                
                # Add team filter if not admin
                if not admin_override:
                    query += " AND d.team_id = %s"
                    params.append(team_id)
                
                # Add similarity threshold if specified
                if score_threshold is not None:
                    query += " AND 1 - (c.embedding <=> %s::vector) >= %s"
                    params.extend([query_embedding, score_threshold])
                
                query += " ORDER BY similarity DESC LIMIT %s"
                params.append(k)
                
                cur.execute(query, tuple(params))
                
                return [
                    {
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "filename": row[2],
                        "text": row[3],
                        "metadata": row[4],
                        "team_id": row[5],
                        "similarity": row[6]
                    }
                    for row in cur.fetchall()
                ]

    def _execute_text_search(
        self,
        query: str,
        team_id: str,
        k: int = 10,
        admin_override: bool = False
    ) -> List[Dict[str, Any]]:
        """Execute full-text search using PostgreSQL with team filtering.
        
        Args:
            query: Search query
            team_id: Team ID to filter results
            k: Number of results to return
            admin_override: Whether to ignore team filtering
            
        Returns:
            List of results with text search scores
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Use plainto_tsquery for more robust FTS
                if not query.strip():
                    return []
                
                # Debug: print the query string and team_id
                print(f"[HybridSearch] FTS search for query: '{query}' team_id: '{team_id}' k: {k} admin_override: {admin_override}")
                
                # Execute text search with ranking and team filtering
                sql = """
                    WITH RankedResults AS (
                        SELECT 
                            c.chunk_id,
                            c.doc_id,
                            d.filename,
                            c.text,
                            d.metadata,
                            d.team_id,
                            ts_rank_cd(c.textsearch, plainto_tsquery('english', %s), 32) as rank
                        FROM app_schema.chunks c
                        JOIN app_schema.documents d ON c.doc_id = d.doc_id
                        WHERE c.textsearch @@ plainto_tsquery('english', %s)
                """
                params = [query, query]
                if not admin_override:
                    sql += " AND d.team_id = %s"
                    params.append(team_id)
                sql += """
                    )
                    SELECT 
                        chunk_id,
                        doc_id,
                        filename,
                        text,
                        metadata,
                        team_id,
                        rank as text_score
                    FROM RankedResults
                    ORDER BY rank DESC
                    LIMIT %s
                """
                params.append(k)
                # Debug: print the SQL and parameters
                print("[HybridSearch] FTS SQL:")
                print(sql)
                print("[HybridSearch] FTS params:")
                print(params)
                try:
                    cur.execute(sql, tuple(params))
                    return [
                        {
                            "chunk_id": row[0],
                            "doc_id": row[1],
                            "filename": row[2],
                            "text": row[3],
                            "metadata": row[4],
                            "team_id": row[5],
                            "text_score": float(row[6] or 0.0)
                        }
                        for row in cur.fetchall()
                    ]
                except psycopg2.Error as e:
                    logger.error(f"Text search failed: {e}")
                    return []

    def _compute_rrf_scores(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        k: int,
        weight: float = 0.5,
        rrf_constant: int = 3
    ) -> List[SearchResult]:
        """Combine vector and text search results using Reciprocal Rank Fusion.
        
        Args:
            vector_results: Results from vector search
            text_results: Results from text search
            k: Number of results to return
            weight: Weight between vector and text scores (0-1), higher favors vector
            rrf_constant: RRF constant (smaller = more weight to top ranks)
        
        Returns:
            Combined and re-ranked results
        """
        c = rrf_constant
        # Create dictionaries mapping chunk IDs to their ranks and scores
        vector_ranks = {
            r["chunk_id"]: (i + 1, r)
            for i, r in enumerate(vector_results)
        }
        text_ranks = {
            r["chunk_id"]: (i + 1, r)
            for i, r in enumerate(text_results)
        }
        all_chunks = set(vector_ranks.keys()) | set(text_ranks.keys())
        results = []
        max_rank = max(len(vector_results), len(text_results))
        for chunk_id in all_chunks:
            vector_rank, vector_data = vector_ranks.get(
                chunk_id, 
                (max_rank + 1, None)
            )
            vector_rrf = 1 / (c + vector_rank) * (1 + c)
            text_rank, text_data = text_ranks.get(
                chunk_id,
                (max_rank + 1, None)
            )
            text_rrf = 1 / (c + text_rank) * (1 + c)
            data = vector_data or text_data
            raw_vector_score = vector_data["similarity"] if vector_data else 0.0
            raw_text_score = text_data["text_score"] if text_data else 0.0
            combined_score = weight * vector_rrf + (1 - weight) * text_rrf
            results.append(
                SearchResult(
                    chunk_id=data["chunk_id"],
                    doc_id=data["doc_id"],
                    filename=data["filename"],
                    text=data["text"],
                    metadata=data["metadata"],
                    team_id=data["team_id"],
                    vector_score=raw_vector_score,
                    text_score=raw_text_score,
                    combined_score=combined_score
                )
            )
        results.sort(key=lambda x: x.combined_score, reverse=True)
        k = min(k, len(results))
        return results[:k]

    def search(
        self,
        query: str,
        query_embedding: List[float],
        team_id: str,
        k: int = 4,
        weight: float = 0.5,
        score_threshold: Optional[float] = None,
        admin_override: bool = False,
        rrf_constant: int = 3
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and text search results.
        
        Args:
            query: The search query string
            query_embedding: The query embedding vector
            team_id: Team ID to filter results
            k: Number of results to return
            weight: Weight between vector (1.0) and text (0.0) search
            score_threshold: Optional minimum similarity threshold
            admin_override: Whether to ignore team filtering
            rrf_constant: RRF constant (smaller = more weight to top ranks)
        
        Returns:
            List of SearchResult objects with combined scores
        """
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
        vector_results = self._execute_vector_search(
            query_embedding, 
            team_id=team_id,
            k=k,
            score_threshold=score_threshold,
            admin_override=admin_override
        )
        print(f"[HybridSearch] Vector results (top {k}):")
        for r in vector_results:
            print(f"  chunk_id={r['chunk_id']} vector_score={r['similarity']:.6f}")

        text_results = self._execute_text_search(
            query,
            team_id=team_id,
            k=k,
            admin_override=admin_override
        )
        print(f"[HybridSearch] FTS results (top {k}):")
        for r in text_results:
            print(f"  chunk_id={r['chunk_id']} text_score={r['text_score']:.6f}")

        results = self._compute_rrf_scores(
            vector_results=vector_results,
            text_results=text_results,
            k=k,
            weight=weight,
            rrf_constant=rrf_constant
        )
        print(f"[HybridSearch] Final merged results (top {k}):")
        for r in results:
            print(f"  chunk_id={r.chunk_id} vector_score={r.vector_score:.6f} text_score={r.text_score:.6f} combined_score={r.combined_score:.6f}")
        return results 