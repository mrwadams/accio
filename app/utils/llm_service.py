"""
LLM service for generating responses using a configured Generative AI client.
Handles prompt construction, system instructions, and error handling.
"""

from typing import List, Dict, Any, Optional, Generator
import logging
from dataclasses import dataclass
# Use types from google.genai
from google.genai import types as google_types
from .base_genai_client import BaseGenAIClient # Import the base class

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
    A service for interacting with Generative AI models via a BaseGenAIClient.
    Provides enhanced functionality for content generation with proper
    context handling, error recovery, and configurable parameters.
    """
    
    def __init__(self, client: BaseGenAIClient, max_retries: int = 2):
        """
        Initialize the LLM service.
        
        Args:
            client: An instance of a class implementing BaseGenAIClient
            max_retries: Maximum number of retries for failed generations
        """
        if not isinstance(client, BaseGenAIClient):
             raise TypeError("client must be an instance of BaseGenAIClient")
        self.client = client
        # Model name is now handled by the specific client implementation
        # self.model = model 
        self.max_retries = max_retries
        logger.info(f"LLMService initialized with client: {type(client).__name__}")
        
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
        temperature: float = 0.3,
        max_output_tokens: int = 4096,
        top_k: int = 40,
        top_p: float = 0.95,
        **kwargs
    ) -> Optional[str]:
        """
        Generate a response based on the query and optional contexts using the configured client.
        
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
        # Need to get safety_settings from the method arguments
        safety_settings: Optional[List[google_types.SafetySetting]] = kwargs.get("safety_settings")

        prompt_data = self._construct_prompt(query, contexts)
        # DEBUG: Print the constructed prompt to the CLI
        print("\n[DEBUG] Prompt sent to LLM (including context):\n" + prompt_data["contents"] + "\n")
        retries = 0
        
        while retries <= self.max_retries:
            try:
                # Use google_types.GenerateContentConfig as per example
                generation_config = google_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    safety_settings=safety_settings,
                    system_instruction=prompt_data["system_instruction"]
                )
                
                # Construct contents, potentially including system prompt if appropriate for the model/client
                # For now, keep system prompt separate and handled by client if possible.
                contents = prompt_data["contents"] 
                # TODO: Consider adding system_instruction as a separate message/part if the underlying API supports it better.

                # Call the client's generate_content method
                response = self.client.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                    # Remove stream=False as it's no longer a parameter
                    # stream=False 
                    # safety_settings can be added here if needed and handled by client impl.
                    **kwargs # Pass other args like safety_settings if needed
                )
                
                # Extract text - Assuming response object has a .text attribute or similar
                # The Vertex client adapter ensures a compatible response object.
                if hasattr(response, 'text'):
                     return response.text
                elif response.candidates: # Fallback check for common structure
                     # Handle potential multiple candidates or parts if necessary
                     first_candidate = response.candidates[0]
                     if first_candidate.content and first_candidate.content.parts:
                         return "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))
                
                logger.warning("Could not extract text from response object.")
                return None # Or handle differently

            except Exception as e:
                retries += 1
                logger.error(f"Error generating content via {type(self.client).__name__} (attempt {retries}/{self.max_retries+1}): {e}", exc_info=True)
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
        **kwargs
    ) -> Optional[Generator[str, None, None]]:
        """
        Generate a streaming response using the configured client.
        
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
        # Need to get safety_settings from the method arguments
        safety_settings: Optional[List[google_types.SafetySetting]] = kwargs.get("safety_settings")

        prompt_data = self._construct_prompt(query, contexts)
        # DEBUG: Print the constructed prompt to the CLI
        print("\n[DEBUG] Prompt sent to LLM (including context):\n" + prompt_data["contents"] + "\n")
        retries = 0
        
        while retries <= self.max_retries:
            try:
                # Use google_types.GenerateContentConfig as per example
                generation_config = google_types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    top_k=top_k,
                    top_p=top_p,
                    safety_settings=safety_settings,
                    system_instruction=prompt_data["system_instruction"]
                )
                
                contents = prompt_data["contents"]

                # Call the client's stream method
                response_stream = self.client.generate_content_stream(
                    contents=contents, # Pass constructed contents
                    generation_config=generation_config
                )
                
                # Define the generator to yield text from chunks
                def response_generator():
                    try:
                        for chunk in response_stream:
                           # Extract text similar to the non-streaming version
                           chunk_text = None
                           if hasattr(chunk, 'text'):
                               chunk_text = chunk.text
                           elif chunk.candidates:
                               first_candidate = chunk.candidates[0]
                               if first_candidate.content and first_candidate.content.parts:
                                   chunk_text = "".join(part.text for part in first_candidate.content.parts if hasattr(part, 'text'))

                           if chunk_text:
                               yield chunk_text
                               
                    except Exception as e:
                        # Log error within the generator if stream breaks mid-way
                        logger.error(f"Error during stream processing by {type(self.client).__name__}: {e}", exc_info=True)
                        # Decide if we should raise or just stop yielding
                        return 
                        
                return response_generator()
                
            except Exception as e:
                retries += 1
                logger.error(f"Error starting content stream via {type(self.client).__name__} (attempt {retries}/{self.max_retries+1}): {e}", exc_info=True)
                if retries > self.max_retries:
                    logger.error("Max retries exceeded for starting stream, returning None")
                    return None 