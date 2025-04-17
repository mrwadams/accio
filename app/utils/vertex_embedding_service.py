class VertexEmbeddingService:
    def __init__(self, vertex_client):
        self.client = vertex_client
        self.embedding_dimension = 768  # Default for text-embedding-004

    def generate_embeddings(self, texts):
        result = self.client.embed_content(texts)
        if result is None:
            # Return zero vectors if embedding fails
            return [[0.0] * self.embedding_dimension for _ in texts]
        return result 