import os
import logging
from typing import List, Dict, Any, Optional, Generator, Union
from google import genai
from google.genai import types as google_types
from .base_genai_client import BaseGenAIClient

logger = logging.getLogger(__name__)

class GoogleGenAIClient(BaseGenAIClient):
    """Implementation of BaseGenAIClient using the google.genai library."""
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash", embedding_model_name: str = "text-embedding-004"):
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided or found in environment variables.")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.embedding_model_name = embedding_model_name
        # Don't fetch models eagerly, pass name string directly to methods
        # try:
        #     self._generative_model = self.client.models.get_model(self.model_name)
        #     self._embedding_model = self.client.models.get_model(self.embedding_model_name)
        logger.info(f"Initialized GoogleGenAIClient with model: {self.model_name} and embedding model: {self.embedding_model_name}")
        # except Exception as e:
        #      logger.error(f"Failed to get models '{self.model_name}' or '{self.embedding_model_name}' from google.genai: {e}", exc_info=True)
        #      raise ValueError(f"Could not initialize google.genai models. Check API key and model names.") from e

    def generate_content(
        self,
        contents: Union[str, List[Union[str, google_types.Part]]],
        generation_config: Optional[google_types.GenerateContentConfig] = None,
        safety_settings: Optional[List[google_types.SafetySetting]] = None,
        **kwargs
    ) -> google_types.GenerateContentResponse:
        """Generates content using the configured Google GenAI model (non-streaming)."""
        try:
            # Pass model name string directly
            # Use the correct keyword 'config' and pass the generation_config object
            response = self.client.models.generate_content(
                model=self.model_name, 
                contents=contents,
                config=generation_config, # Use 'config' keyword 
                # safety_settings is now passed inside config
                # safety_settings=safety_settings, 
                **kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error during Google GenAI content generation: {e}", exc_info=True)
            raise

    def embed_content(
        self,
        texts: List[str],
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        **kwargs
    ) -> List[List[float]]:
        """Generates embeddings using the configured Google GenAI embedding model."""
        try:
            # Pass embedding model name string directly
            result = self.client.models.embed_contents(
                model=self.embedding_model_name,
                requests=[{
                    'content': text,
                    'task_type': task_type,
                    'title': title,
                    'output_dimensionality': output_dimensionality
                } for text in texts],
                **kwargs
            )
            # Extract and return the list of embedding values
            return [embedding.values for embedding in result.embeddings]
            
        except Exception as e:
            logger.error(f"Error during Google GenAI embedding generation: {e}", exc_info=True)
            raise

    def generate_content_stream(
        self,
        contents: Union[str, List[Union[str, google_types.Part]]],
        generation_config: Optional[google_types.GenerateContentConfig] = None,
        safety_settings: Optional[List[google_types.SafetySetting]] = None,
        **kwargs
    ) -> Generator[google_types.GenerateContentResponse, None, None]:
        """Generates content as a stream using the configured Google GenAI model."""
        # Call the dedicated stream method
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=contents,
                config=generation_config,
                # safety_settings handled within config
                **kwargs
            )
            return response_stream
        except Exception as e:
            logger.error(f"Error during Google GenAI stream generation: {e}", exc_info=True)
            raise 