from langchain_core.embeddings import Embeddings

class LangChainVertexEmbeddings(Embeddings):
    def __init__(self, vertex_embedding_service):
        self.vertex_embedding_service = vertex_embedding_service

    def embed_documents(self, texts):
        return self.vertex_embedding_service.generate_embeddings(texts)

    def embed_query(self, text):
        return self.vertex_embedding_service.generate_embeddings([text])[0] 