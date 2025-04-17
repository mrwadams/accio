import os
import logging
from dotenv import load_dotenv
from typing import List, Optional, Generator

# Import the new client implementations and base class
from .base_genai_client import BaseGenAIClient
from .google_genai_client import GoogleGenAIClient
from .vertex_genai_client import VertexGenAIClient
from .llm_service import LLMService, RetrievedContext
# Import google types for helper function return types
from google.genai import types as google_types

# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Client Configuration --- 
# Determine which client to use based on environment variable
GENAI_CLIENT_TYPE = os.getenv("GENAI_CLIENT_TYPE", "google").lower()

# Instantiate the chosen client
client: BaseGenAIClient

try:
    if GENAI_CLIENT_TYPE == "vertex":
        logger.info("Using VertexGenAIClient")
        # Configuration for Vertex comes from env vars within the class
        client = VertexGenAIClient()
    elif GENAI_CLIENT_TYPE == "google":
        logger.info("Using GoogleGenAIClient")
        # Requires GOOGLE_API_KEY env var
        client = GoogleGenAIClient() 
    else:
        raise ValueError(f"Unsupported GENAI_CLIENT_TYPE: {GENAI_CLIENT_TYPE}. Choose 'google' or 'vertex'.")
except Exception as e:
    logger.exception(f"Failed to initialize GenAI client '{GENAI_CLIENT_TYPE}'. Please check configuration and credentials.")
    # Depending on the application, you might want to exit or provide a dummy client
    raise RuntimeError("GenAI Client initialization failed") from e

# --- Service Initialization --- 
# Initialize LLMService with the chosen client instance
llm_service = LLMService(client)

# --- Public Helper Functions --- 

def generate_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
    """
    Generate an embedding for a single text using the configured client.
    
    Args:
        text: The text to generate an embedding for.
        task_type: The intended task for the embedding (e.g., RETRIEVAL_DOCUMENT).
        
    Returns:
        List of floats representing the text embedding, or None on error.
    """
    try:
        # client.embed_content now returns List[List[float]] directly
        embeddings_list = client.embed_content([text], task_type=task_type)
        # Extract the first embedding's values
        if embeddings_list:
            return embeddings_list[0]
        else:
            logger.warning("Embeddings response was empty.")
            return None
    except Exception as e:
        logger.error(f"Error generating embedding using {type(client).__name__}: {e}", exc_info=True)
        return None

def generate_embeddings(texts: List[str], task_type: str = "RETRIEVAL_DOCUMENT") -> Optional[List[List[float]]]:
    """
    Generate embeddings for a list of texts using the configured client.
    
    Args:
        texts: The list of texts to generate embeddings for.
        task_type: The intended task for the embeddings.
        
    Returns:
        A list of embedding lists, or None on error.
    """
    try:
        # client.embed_content now returns List[List[float]] directly
        return client.embed_content(texts, task_type=task_type)
        # No need to extract further, return the result directly
    except Exception as e:
        logger.error(f"Error generating batch embeddings using {type(client).__name__}: {e}", exc_info=True)
        return None

# Updated generate_content and generate_content_stream to use LLMService

def generate_content(prompt: str, temperature: float = 0.7, **kwargs) -> Optional[str]:
    """
    Generate content based on a simple prompt using the LLMService.
    Does not use RAG context.
    
    Args:
        prompt: The prompt to generate content from.
        temperature: Controls randomness in generation (default: 0.7).
        **kwargs: Additional arguments passed to LLMService (e.g., max_output_tokens).
        
    Returns:
        Generated content as a string, or None on error.
    """
    # LLMService handles the underlying client call
    return llm_service.generate_response(
        query=prompt,
        contexts=None,  # No RAG context for this helper
        temperature=temperature,
        **kwargs
    )

def generate_content_stream(prompt: str, temperature: float = 0.7, **kwargs) -> Optional[Generator[str, None, None]]:
    """
    Generate content as a stream based on a simple prompt using the LLMService.
    Does not use RAG context.
    
    Args:
        prompt: The prompt to generate content from.
        temperature: Controls randomness in generation (default: 0.7).
        **kwargs: Additional arguments passed to LLMService (e.g., max_output_tokens).
        
    Returns:
        Generator yielding content chunks, or None on error.
    """
    # LLMService handles the underlying client call
    return llm_service.generate_response_stream(
        query=prompt,
        contexts=None,  # No RAG context for this helper
        temperature=temperature,
        **kwargs
    )

# Expose the configured client and service instances if needed elsewhere
# (Use with caution, prefer using the helper functions)
configured_client = client
configured_llm_service = llm_service 