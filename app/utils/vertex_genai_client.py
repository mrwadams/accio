from .base_genai_client import BaseGenAIClient

class VertexGenAIClient(BaseGenAIClient):
    def __init__(self, *args, **kwargs):
        raise ImportError("VertexGenAIClient is not available in this repository. Please provide the internal implementation.")

    def generate_content(self, *args, **kwargs):
        raise NotImplementedError("VertexGenAIClient is not available.")

    def embed_content(self, *args, **kwargs):
        raise NotImplementedError("VertexGenAIClient is not available.")

    def generate_content_stream(self, *args, **kwargs):
        raise NotImplementedError("VertexGenAIClient is not available.") 