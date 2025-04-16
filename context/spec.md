### **LLM RAG Chatbot Specification v2.7**

**1. Objective:**
   To create an LLM-powered chatbot utilizing a Retrieval-Augmented Generation (RAG) architecture to answer questions based on team-specific knowledge bases, deployed internally on OpenShift.

**2. Core Components:**
   * **Frontend:** Streamlit application.
   * **Backend/Orchestration:** Python application logic using the `google.genai` library (imported via `from google import genai`) and `langchain` for document processing. Requires initializing a client: `client = genai.Client(api_key=...)`.
   * **LLM:** Google Gemini model `gemini-2.0-flash` accessed via the client.
   * **Embedding Model:** Google embedding model `text-embedding-004` accessed via the client.
   * **Vector Database:** PostgreSQL with `pgvector` extension.
   * **Deployment:** Streamlit app containerized on OpenShift, PostgreSQL on VM.

**3. Data Management & Configuration:**
   * **Supported Sources:** PDF, Microsoft Word (.docx), CSV.
   * **Document Processing:** Utilize Langchain document loaders for robust text extraction from various formats.
   * **Ingestion Workflow:** Manual via Admin Panel initially, potential S3 automation later.
   * **Preprocessing:** Text extraction via Langchain, guidance for clean data, error handling flagged in Admin UI.
   * **Chunking:** Utilize Langchain text splitters with multiple strategies, configurable, offline benchmarking planned.
   * **Data Model:**
      * Teams table: team_id (PK), access_code, team_name, configuration (JSONB)
      * Documents table: doc_id (PK), team_id (FK), filename, metadata (JSONB), status
      * Chunks table: chunk_id (PK), doc_id (FK), text, embedding (vector)
   * **Document Management:**
      * List all documents with metadata (filename, team, chunks, status)
      * View document details including processing history
      * Filter/search documents by team or metadata
      * Bulk operations (delete)
      * Note: To update document processing settings, delete existing document and re-upload with new settings
   * **Vectorization:** Use `text-embedding-004` via the `genai` client.
   * **Document Deletion:** Admin panel mechanism using Document ID.
   * **Configuration Management:** Store team-specific configurations (prompts, model settings, retrieval settings) in the teams table as JSONB. This allows tuning based on specific team needs and benchmark outcomes.

**4. RAG Pipeline & LLM Interaction:**
   * **Retrieval Strategy:** Hybrid Search (Vector + Keyword [PG FTS recommended]), RRF score fusion. Parameters tunable per schema.
   * **Query Pre-processing:** Multi-Query Generation using `gemini-2.0-flash`.
   * **Agentic Behavior (Framework-less):**
        * **Retrieval Assessment:** Use `gemini-2.0-flash` to review chunks.
        * **Self-Correction/Clarification:** Use `gemini-2.0-flash` to re-write query or ask questions.
        * **Retry Limits:** Max retries (e.g., 2-3).
        * **Transparency:** Expose steps in Streamlit expander.
   * **Re-ranking:** Use `gemini-2.0-flash` to re-rank candidates.
   * **Context Management:** Pass top K re-ranked chunks (K tunable per schema).
   * **Answer Generation:**
        * Construct prompt for `gemini-2.0-flash`. Prompts potentially tunable per schema.
        * Include system instructions via `config=types.GenerateContentConfig(...)`.
        * Instruct model to answer based on context and cite sources.
   * **Streaming:** Use `client.models.generate_content_stream()`.

**5. Frontend & User Experience (Streamlit):**
   * **User Interface:** Chat, streamed responses with citations, agent "thinking" expander, source doc list, feedback buttons.
   * **Admin Panel:** Separate section/auth, schema management, upload, deletion, error flags, interface for managing per-schema configurations.

**6. Access Control & Security:**
   * **Authentication:** 
      * All authentication is handled via access codes stored in environment variables.
      * Teams authenticate using their team-specific access code.
      * Admin authentication uses a separate admin access code.
      * No per-user management or user management UI in the current design.
   * **Authorization:**
      * Team access is restricted to documents tagged with their team_id.
      * All database queries automatically filter by team_id after authentication.
      * Admins have access to all documents and configuration settings.
   * **Data Isolation:**
      * Single application schema containing all tables.
      * Team isolation achieved through application-level filtering.
      * All queries include team_id filters to ensure data separation.
      * Database indexes on team_id columns to optimize filtered queries.

**7. Deployment, Operations & Documentation:**
   * **Infrastructure:** Streamlit on OpenShift, PostgreSQL on VM.
   * **Scalability:** PoC scale initially.
   * **Logging:** Queries, responses, agent steps, feedback. Standard OpenShift logging.
   * **Runtime Error Handling:** User-friendly errors, retry option.
   * **Onboarding Documentation:** Create clear documentation outlining the process for onboarding a new team, including schema setup, data requirements, and initial configuration steps (Onboarding Checklist).
   * **User Training Materials:** Develop short, targeted training materials (e.g., quick reference guide, example prompts) for end-users to help them interact effectively with the chatbot.

**8. Non-Functional Requirements:**
    * **Code Reusability & Maintainability:** The backend Python code should be designed in a modular and well-structured way to promote reusability of components (e.g., data loaders, retrieval logic, LLM interaction wrappers) and ease of maintenance.