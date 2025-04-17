from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union
# Use types from google.genai
from google.genai import types as google_types

class BaseGenAIClient(ABC):
    """Abstract base class for Generative AI clients."""

    @abstractmethod
    def generate_content(
        self,
        contents: Union[str, List[Union[str, google_types.Part]]], # Use google_types.Part
        generation_config: Optional[google_types.GenerateContentConfig] = None,
        safety_settings: Optional[List[google_types.SafetySetting]] = None, # Use google_types.SafetySetting
        **kwargs
    ) -> google_types.GenerateContentResponse:
        """Generates content based on the provided input (non-streaming)."""
        pass

    @abstractmethod
    def embed_content(
        self,
        texts: List[str],
        task_type: Optional[str] = "RETRIEVAL_DOCUMENT",
        title: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
        **kwargs
        # The return type should be the actual list of embeddings
    ) -> List[List[float]]: # Return List[List[float]] directly
        """Generates embeddings for a list of texts."""
        pass

    @abstractmethod
    def generate_content_stream(
        self,
        contents: Union[str, List[Union[str, google_types.Part]]], # Use google_types.Part
        generation_config: Optional[google_types.GenerateContentConfig] = None,
        safety_settings: Optional[List[google_types.SafetySetting]] = None, # Use google_types.SafetySetting
        **kwargs
    ) -> Generator[google_types.GenerateContentResponse, None, None]:
        """Generates content as a stream."""
        pass 