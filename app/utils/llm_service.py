"""
LLM service for generating responses using Google's Gemini model.
Handles prompt construction, system instructions, and error handling.
"""

from typing import List, Dict, Any, Optional, Generator
import logging
from dataclasses import dataclass
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

@dataclass
class RetrievedContext:
    """Represents a chunk of retrieved context with metadata."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    score: float

class LLMService:
    """
    A service for interacting with Google's Generative AI models.
    Provides enhanced functionality for content generation with proper
    context handling, error recovery, and configurable parameters.
    """
    
    def __init__(self, client: genai.Client, model: str = "gemini-2.0-flash", max_retries: int = 2):
        """
        Initialize the LLM service.
        
        Args:
            client: The Google GenAI client instance
            model: The model to use for generation (default: gemini-2.0-flash)
            max_retries: Maximum number of retries for failed generations
        """
        self.client = client
        self.model = model
        self.max_retries = max_retries
        
    def _construct_prompt(self, query: str, contexts: Optional[List[RetrievedContext]] = None) -> Dict[str, Any]:
        """
        Construct a prompt using the query and retrieved contexts.
        
        Args:
            query: The user's query
            contexts: Optional list of retrieved context chunks
            
        Returns:
            A dictionary containing the prompt parts and system instruction
        """
        # System instruction
        system_instruction = (
            "You are a helpful AI assistant. When using information from the provided context:"
            "\n- Always cite your sources using [Source #] format, where # is the source number"
            "\n- If multiple sources support a statement, cite all relevant sources: [Source #, #]"
            "\n- If you're unsure about something, say so"
            "\n- If the context doesn't contain relevant information, say so"
            "\n- Keep responses clear and concise"
            "\n- Do not modify or make up source numbers"
        )
        
        # Build content parts
        content_parts = []
        
        # Add context sections if provided
        if contexts:
            content_parts.append("Relevant context:")
            for i, ctx in enumerate(contexts, 1):
                # Extract filename from metadata or use doc_id
                source_name = ctx.metadata.get('filename', ctx.doc_id)
                # Add source with number and metadata
                content_parts.append(f"[Source {i}] {source_name}:")
                content_parts.append(ctx.content)
        
        # Add the user's query
        content_parts.append(f"\nUser query: {query}\n\nResponse:")
        
        return {
            "contents": "\n".join(content_parts),
            "system_instruction": system_instruction
        }
    
    def generate_response(
        self,
        query: str,
        contexts: Optional[List[RetrievedContext]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> Optional[str]:
        """
        Generate a response based on the query and optional contexts.
        
        Args:
            query: The user's query
            contexts: Optional list of retrieved context chunks
            temperature: Controls randomness in generation
            max_output_tokens: Maximum number of tokens to generate
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability cutoff for token selection
            
        Returns:
            The generated response text, or None if generation fails
        """
        prompt = self._construct_prompt(query, contexts)
        retries = 0
        
        while retries <= self.max_retries:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    system_instruction=prompt["system_instruction"]
                )
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt["contents"],
                    config=config
                )
                return response.text
                
            except Exception as e:
                retries += 1
                logger.error(f"Error generating content (attempt {retries}/{self.max_retries}): {e}")
                if retries > self.max_retries:
                    logger.error("Max retries exceeded, returning None")
                    return None
    
    def generate_response_stream(
        self,
        query: str,
        contexts: Optional[List[RetrievedContext]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024,
        top_k: int = 40,
        top_p: float = 0.95,
    ) -> Optional[Generator[str, None, None]]:
        """
        Generate a streaming response based on the query and optional contexts.
        
        Args:
            query: The user's query
            contexts: Optional list of retrieved context chunks
            temperature: Controls randomness in generation
            max_output_tokens: Maximum number of tokens to generate
            top_k: Number of highest probability tokens to consider
            top_p: Cumulative probability cutoff for token selection
            
        Returns:
            A generator yielding response chunks, or None if generation fails
        """
        prompt = self._construct_prompt(query, contexts)
        retries = 0
        
        while retries <= self.max_retries:
            try:
                config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    system_instruction=prompt["system_instruction"]
                )
                
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt["contents"],
                    config=config
                )
                
                def response_generator():
                    try:
                        for chunk in response:
                            if chunk.text:
                                yield chunk.text
                    except Exception as e:
                        logger.error(f"Error in stream generation: {e}")
                        return
                        
                return response_generator()
                
            except Exception as e:
                retries += 1
                logger.error(f"Error starting content stream (attempt {retries}/{self.max_retries}): {e}")
                if retries > self.max_retries:
                    logger.error("Max retries exceeded, returning None")
                    return None 