"""
Re-ranking implementation using Gemini model.
Re-ranks retrieved chunks based on relevance to the query using LLM scoring.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class GeminiReranker:
    """Re-ranks retrieved chunks using Gemini for more accurate relevance scoring."""
    
    def __init__(self, client: genai.Client, model: str = "gemini-2.0-flash", max_retries: int = 2):
        """
        Initialize the re-ranker.
        
        Args:
            client: The Google GenAI client instance
            model: The model to use for re-ranking (default: gemini-2.0-flash)
            max_retries: Maximum number of retries for failed generations
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
    
    def _construct_prompt(self, query: str, chunk: str) -> Dict[str, Any]:
        """
        Construct a prompt for scoring chunk relevance.
        
        Args:
            query: The user query
            chunk: The text chunk to score
            
        Returns:
            A dictionary containing the prompt parts and system instruction
        """
        system_instruction = (
            "You are a relevance scoring system. Your task is to score how relevant a text chunk is "
            "to a given query. Consider:"
            "\n- Direct relevance to the query topic"
            "\n- Information completeness"
            "\n- Presence of key details"
            "\n- Factual alignment"
            "\nOutput format: Return only a single number between 0 and 10, where:"
            "\n- 0-3: Low/No relevance"
            "\n- 4-6: Moderate relevance"
            "\n- 7-10: High relevance"
        )
        
        content = f"Query: {query}\n\nText chunk to score:\n{chunk}"
        
        return {
            "contents": content,
            "system_instruction": system_instruction
        }
    
    def score_chunk(
        self,
        query: str,
        chunk: str,
        temperature: float = 0.1  # Lower temperature for more consistent scoring
    ) -> Optional[float]:
        """
        Score a single chunk's relevance to the query.
        
        Args:
            query: The user query
            chunk: The text chunk to score
            temperature: Controls randomness in generation
            
        Returns:
            Relevance score between 0-10, or None if scoring fails
        """
        prompt = self._construct_prompt(query, chunk)
        retries = 0
        
        while retries <= self.max_retries:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=8,  # Very short output needed
                    top_k=1,  # More deterministic
                    top_p=0.1,  # More focused on highest probability
                    system_instruction=prompt["system_instruction"]
                )
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt["contents"],
                    config=config
                )
                
                # Extract and validate the score
                try:
                    score = float(response.text.strip())
                    if 0 <= score <= 10:
                        return score
                    logger.warning(f"Score {score} out of range, retrying...")
                except ValueError:
                    logger.warning(f"Invalid score format: {response.text}, retrying...")
                
                retries += 1
                
            except Exception as e:
                retries += 1
                logger.error(f"Error scoring chunk (attempt {retries}/{self.max_retries}): {e}")
                
        logger.error("Max retries exceeded, returning None")
        return None

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Tuple[str, float]],  # (chunk_text, initial_score) pairs
        temperature: float = 0.1
    ) -> List[Tuple[str, float]]:
        """
        Re-rank a list of chunks based on LLM relevance scoring.
        
        Args:
            query: The user query
            chunks: List of (chunk_text, initial_score) pairs
            temperature: Controls randomness in scoring
            
        Returns:
            List of (chunk_text, final_score) pairs, sorted by final score
        """
        reranked_chunks = []
        
        for chunk_text, initial_score in chunks:
            # Get LLM relevance score
            llm_score = self.score_chunk(query, chunk_text, temperature)
            
            if llm_score is not None:
                # Combine LLM score with initial score (e.g., from vector similarity)
                # Using simple average here, but could be weighted based on tuning
                final_score = (llm_score + initial_score * 10) / 2  # Scale initial score to 0-10 range
                reranked_chunks.append((chunk_text, final_score))
            else:
                # Fallback to initial score if LLM scoring fails
                logger.warning("Using fallback scoring for chunk due to LLM scoring failure")
                reranked_chunks.append((chunk_text, initial_score * 10))
        
        # Sort by final score in descending order
        reranked_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return reranked_chunks 