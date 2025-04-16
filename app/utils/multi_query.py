"""
Multi-query generation implementation using Gemini model.
Generates multiple search queries from a single user query to improve retrieval coverage.
"""

from typing import List, Optional, Dict, Any
import logging
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class MultiQueryGenerator:
    """Generates multiple search queries from a single query using Gemini."""
    
    def __init__(self, client: genai.Client, model: str = "gemini-2.0-flash", max_retries: int = 2):
        """
        Initialize the multi-query generator.
        
        Args:
            client: The Google GenAI client instance
            model: The model to use for generation (default: gemini-2.0-flash)
            max_retries: Maximum number of retries for failed generations
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
    
    def _construct_prompt(self, query: str) -> Dict[str, Any]:
        """
        Construct a prompt for generating multiple search queries.
        
        Args:
            query: The original user query
            
        Returns:
            A dictionary containing the prompt parts and system instruction
        """
        system_instruction = (
            "You are a search query generator. Your task is to generate 3-5 different search queries "
            "that capture different aspects and phrasings of the user's original query. The queries should:"
            "\n- Be diverse in their phrasing and focus"
            "\n- Maintain the original intent"
            "\n- Include both broad and specific formulations"
            "\n- Be clear and concise"
            "\nOutput format: Return only the queries, one per line, without numbering or additional text."
        )
        
        content = f"Generate alternative search queries for: {query}"
        
        return {
            "contents": content,
            "system_instruction": system_instruction
        }
    
    def generate_queries(
        self,
        query: str,
        temperature: float = 0.7,
    ) -> Optional[List[str]]:
        """
        Generate multiple search queries from a single query.
        
        Args:
            query: The original user query
            temperature: Controls randomness in generation
            
        Returns:
            List of generated queries, or None if generation fails
        """
        prompt = self._construct_prompt(query)
        retries = 0
        
        while retries <= self.max_retries:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=256,  # Shorter output is sufficient
                    top_k=40,
                    top_p=0.95,
                    system_instruction=prompt["system_instruction"]
                )
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt["contents"],
                    config=config
                )
                
                # Split response into individual queries and clean them
                queries = [
                    q.strip() 
                    for q in response.text.split('\n') 
                    if q.strip()
                ]
                
                # Filter out any empty queries and limit to 5
                queries = [q for q in queries if q][:5]
                
                # Always include the original query as the first one
                if query not in queries:
                    queries.insert(0, query)
                
                return queries
                
            except Exception as e:
                retries += 1
                logger.error(f"Error generating queries (attempt {retries}/{self.max_retries}): {e}")
                if retries > self.max_retries:
                    logger.error("Max retries exceeded, returning None")
                    return None 