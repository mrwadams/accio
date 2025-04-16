### **RAG Chatbot Implementation Plan v1.7**

This plan outlines the steps to implement the RAG chatbot based on `rag_chatbot_spec_v2.6`. *Emphasis added on modular design throughout backend phases.*

**Phase 0: Setup & Foundation**

1.  **Environment Setup:** Install dependencies, import `genai`, `types`, `langchain`.
2.  **Client Initialization:** `client = genai.Client(...)`.
3.  **Database Setup:**
     * Create application schema and enable pgvector
     * Create teams, documents, and chunks tables with appropriate indexes
     * Add team_id foreign key constraints and indexes
4.  **Basic Streamlit App:** Minimal structure.
5.  **API Connection Test:** Test `client`, `generate_content` (`gemini-2.0-flash`), `embed_content` (`text-embedding-004`).

**Phase 1: Core RAG Pipeline - MVP** *(Focus on modular functions)*

1.  **Document Loading:** Implement using Langchain document loaders for various formats (CSV, PDF, DOCX).
2.  **Text Splitting:** Implement using Langchain text splitters with configurable strategies.
3.  **Embedding & Storage:** Generate embeddings (`text-embedding-004`), store in DB. Wrap DB interactions.
4.  **Basic Vector Retrieval:** Implement `pgvector` search function.
5.  **Simple Generation:** Construct prompt, call `generate_content` (`gemini-2.0-flash`). Wrap LLM call.
6.  **Streamlit Integration:** Connect modular components into the basic chat UI.

**Phase 2: Enhancing Data Handling & Admin Panel** *(Focus on modularity)*

1.  **Multi-Format Parsing:** Leverage Langchain document loaders for robust parsing of PDF, DOCX, CSV.
2.  **Full Metadata Extraction:** Implement reusable metadata extraction logic.
3.  **Admin Panel UI:** Build multi-page/section app. Include placeholders/basic UI for configuration management.
4.  **Ingestion Logic:** Connect UI to modular backend ingestion pipeline (`embed_content`). Handle errors.
5.  **Document Management:**
    * Implement database queries to list/filter documents
    * Add UI components for viewing document details
    * Add document search/filter functionality
    * Implement bulk operations interface
6.  **Document Deletion:** Implement backend logic and UI integration.
    * Note: Document updates/reprocessing should be handled by deleting and re-uploading with new settings.

**Phase 3: Advanced RAG Features** *(Focus on modularity)*

1.  **Hybrid Search Implementation:** Implement reusable keyword search (PG FTS) + vector search module. Implement RRF score fusion.
2.  **Multi-Query Generation:** Implement reusable module using `generate_content`. Integrate into retrieval flow.
3.  **Re-ranking Implementation:** Implement reusable module using `generate_content`. Integrate after retrieval.

**Phase 4: Agentic Behavior & User Experience** *(Focus on modularity)*

1.  **Agentic Logic Implementation:** Develop modular functions/classes for assessment, re-writing, clarification using `generate_content`. Integrate flow with retry limits.
2.  **UI Enhancements:** Implement expander, streaming (`generate_content_stream`), citation display.
3.  **Feedback Mechanism:** Implement UI and DB storage.
4.  **Runtime Error Handling:** Implement robust error handling.

**Phase 5: Access Control, Configuration & Deployment**

1.  **Team Management (Admin):**
     * Implement team CRUD operations
     * Add team configuration management UI
     * Create initial teams with access codes
2.  **Authentication:**
     * Implement shared login flow using access codes
     * Add team context to session state
     * Implement admin access flow
3.  **Query Filtering:**
     * Create query wrapper to automatically filter by team_id
     * Implement admin override for full access
     * Add team-specific configuration loading
4.  **Containerization:** Create `Dockerfile`.
5.  **Deployment:** Deploy to OpenShift, configure connections and environment variables.
6.  **Logging Setup:** Configure application logging.

**Phase 6: Testing, Documentation & Refinement**

1.  **Benchmarking Setup:** Create evaluation script. Use results to inform per-schema configurations.
2.  **End-to-End Testing:** Test thoroughly.
3.  **Prompt Tuning:** Refine prompts, potentially storing variations in the new configuration system. Consider using `types.GenerateContentConfig`.
4.  **Parameter Tuning:** Adjust parameters via the configuration system.
5.  **User Acceptance Testing (UAT):** Conduct PoCs.
6.  **Create Onboarding Documentation:** Develop the checklist/guide for adding new teams.
7.  **Create User Training Materials:** Develop the quick reference guide/examples for end-users.
8.  **Iterate:** Refine based on testing, feedback, and UAT.