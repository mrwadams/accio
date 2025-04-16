# Accio: RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that uses Google's Gemini models to answer questions based on team-specific knowledge bases.

## Features

- **RAG Architecture**: Combines retrieval of relevant information with generation for accurate responses
- **Multiple Document Types**: Support for PDF, DOCX, CSV, and TXT files using Langchain document loaders
- **Vector Database**: Uses PostgreSQL with pgvector for efficient similarity search
- **Admin Panel**: Interface for managing documents and configurations
- **Streaming Responses**: Real-time streaming of model responses

## Project Structure

```
accio/
├── app/
│   ├── components/       # Reusable UI components
│   └──  utils/            # Core utility functions (production)
├── app_pages/            # Streamlit multi-page UI (chat, admin, docs, etc.)
├── context/              # Project plans, specs, and context docs
├── examples/             # Example scripts (not used in production)
├── app.py                # Main Streamlit app entry point
├── requirements.txt      # Project dependencies
├── requirements-dev.txt  # Dev dependencies
├── tests/                # Unit tests
├── .env                  # Environment variables
├── .env.example          # Example environment file
└── README.md             # This file
```

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install system dependencies:
   ```bash
   # On macOS (using Homebrew):
   brew install libmagic

   # On Ubuntu/Debian:
   sudo apt-get install libmagic1

   # On Windows:
   # libmagic is included in the python-magic-bin package
   ```
4. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up environment variables in `.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   DB_NAME=your_db_name
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_HOST=localhost
   DB_PORT=5432
   ADMIN_CODE=your_admin_code_here
   ```
6. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## RAG Pipeline

Our RAG pipeline consists of two main flows: **Document Ingestion** and **Chat Retrieval/Generation**.

### Document Ingestion

- **Upload & Format Detection:**  
  Users upload PDF, DOCX, CSV, TXT, or MD files via the admin panel. File type is detected automatically.
- **Text Splitting:**  
  - Supports three strategies:  
    - **Paragraph-based (recursive):** Splits on paragraphs, then sentences, then words.  
    - **Fixed Size:** Splits by character count.  
    - **Semantic:** Uses AI to split based on meaning (no fixed size).
  - Chunk size and overlap are configurable for non-semantic strategies.
- **Metadata Extraction:**  
  Extracts filename, file type, size, chunking strategy, and more.
- **Chunking & Embedding:**  
  - Each chunk is embedded using Google's `text-embedding-004` model.
  - Chunks and embeddings are stored in PostgreSQL with pgvector.
- **Error Handling:**  
  - Errors and warnings are surfaced in the UI.
  - Failed documents can be retried or deleted.

### Chat Retrieval & Generation

- **Authentication:**  
  - Users log in with a team ID and access code.
  - Admins can view and manage all documents.
- **Retrieval Methods:**  
  Users can select from:
  - **Vector Search:** Semantic search using embeddings.
  - **Hybrid Search:** Combines vector and keyword (FTS) search with configurable weighting.
  - **Multi-Query:** Generates multiple query variants for broader recall.
  - **Reranked:** Uses LLM to rerank top vector results for deeper relevance.
  - **Agentic (ReAct):** Experimental mode with iterative reasoning and clarification.
- **Processing Steps:**  
  - Recent chat history is included for context.
  - Retrieval is performed according to the selected method.
  - For agentic mode, the agent can assess sufficiency, ask clarifying questions, and retry with expanded parameters.
- **Streaming Generation:**  
  - Gemini 2.0 Flash generates answers, streaming responses in real time.
  - Source citations are included and displayed with tooltips.
- **User Feedback:**  
  - Users can mark responses as helpful, not helpful, or report issues.
  - Feedback is stored for future improvement.
- **Transparency:**  
  - Users can view the sources used, scores, and agentic reasoning steps in the UI.

## Advanced RAG Techniques

### Hybrid Search

- Combines vector similarity (pgvector) and PostgreSQL full-text search (FTS).
- Uses Reciprocal Rank Fusion (RRF) to merge results.
- Weighting between vector and text search is user-configurable.
- Scores and sources are visualized in the UI.

### Multi-Query Retrieval

- Generates multiple query variants using Gemini.
- Runs vector search for each variant in parallel.
- Deduplicates and ranks results by score.
- Increases recall for ambiguous or broad queries.

### Reranking

- Retrieves an expanded set of candidates via vector search.
- Uses Gemini to score each chunk for relevance (0–10 scale).
- Combines LLM and vector scores for final ranking.
- Shows both initial and reranked scores in the UI.

### Agentic Retrieval

- Iteratively assesses if retrieved context is sufficient.
- Can ask clarifying questions if needed.
- Retries retrieval with expanded parameters if initial results are insufficient.
- All steps and reasoning are shown to the user.

## Example Usage

```python
# Vector Search
results = vector_store.similarity_search(query_embedding, k=4)

# Hybrid Search
results = hybrid_searcher.search(
    query="performance metrics",
    query_embedding=query_embedding,
    k=4,
    weight=0.7
)

# Multi-Query
queries = multi_query_generator.generate_queries("What is a Transformer?")
all_results = []
for q in queries:
    q_embedding = embedding_service.generate_embedding(q)
    all_results.extend(vector_store.similarity_search(q_embedding, k=3))
# Deduplicate and rank...

# Reranked
initial_results = vector_store.similarity_search(query_embedding, k=8)
reranked = reranker.rerank_chunks(query, [(r["text"], r["similarity"]) for r in initial_results])
```

## Document Loading

The project uses Langchain document loaders for robust document processing:

- **Supported Formats**:
  - PDF files (using `PyPDFLoader`)
  - DOCX files (using `Docx2txtLoader`)
  - CSV files (using `CSVLoader`)
  - Text files (using `UnstructuredLoader`)

- **Features**:
  - Automatic format detection
  - Metadata extraction
  - Batch processing
  - Error handling
  - Recursive directory support

- **Usage Example**:
  ```python
  from app.utils.document_loader import DocumentLoader

  # Load a single document
  result = DocumentLoader.load_document("path/to/document.pdf")

  # Load all supported documents from a directory
  results = DocumentLoader.load_documents_from_directory("path/to/docs", recursive=True)
  ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 