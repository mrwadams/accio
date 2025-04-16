import streamlit as st
import os
from pathlib import Path
import tempfile
from dotenv import load_dotenv
from google import genai
from app.utils.database import store_document, get_team_documents, delete_document, get_all_documents, get_document_chunks
from app.utils.embeddings import EmbeddingService, VectorStore
from app.utils.ingestion import IngestionPipeline
import pandas as pd
import sys
import nltk

# Load environment variables
load_dotenv()

# Ensure NLTK uses bundled data in project root
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'nltk_data')
if NLTK_DATA_PATH not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_PATH)

# Ensure 'punkt' is available (download if missing)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', download_dir=NLTK_DATA_PATH)
    except Exception as e:
        print(f"Error downloading NLTK 'punkt': {e}", file=sys.stderr)
        raise

def check_auth():
    """Check if user is authenticated"""
    if not st.session_state.authenticated:
        st.error("Please login first")
        st.stop()
    return True

def initialize_services():
    """Initialize required services."""
    if 'genai_client' not in st.session_state:
        st.session_state.genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        st.session_state.embedding_service = EmbeddingService(st.session_state.genai_client)
    
    # Initialize vector store
    vector_store = VectorStore({
        "dbname": os.getenv("DB_NAME"),
        "user": os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host": os.getenv("DB_HOST"),
        "port": os.getenv("DB_PORT")
    })
    
    # Initialize ingestion pipeline
    st.session_state.ingestion_pipeline = IngestionPipeline(
        embedding_service=st.session_state.embedding_service,
        vector_store=vector_store
    )

def process_uploaded_file(uploaded_file, team_id, chunk_strategy, chunk_size, overlap):
    """Process an uploaded file through the ingestion pipeline."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        # Write uploaded file content to temp file
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # First store the document to get a doc_id
        metadata = {
            "filename": uploaded_file.name,
            "file_type": uploaded_file.type,
            "file_size": uploaded_file.size,
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size if chunk_strategy != "semantic" else None,
            "chunk_overlap": overlap if chunk_strategy != "semantic" else None,
            "num_chunks": None  # Will be updated after processing
        }
        doc_id = store_document(team_id, uploaded_file.name, metadata)
        if not doc_id:
            st.error("Failed to store document metadata")
            return
        
        # Process the file
        loader_config = {
            "chunk_strategy": chunk_strategy,
            "original_filename": uploaded_file.name,
            "team_id": team_id
        }
        
        # Only add chunk_size and overlap if not using semantic chunking
        if chunk_strategy != "semantic":
            loader_config["chunk_size"] = chunk_size
            loader_config["chunk_overlap"] = overlap
        
        result = st.session_state.ingestion_pipeline.process_file(
            file_path=tmp_file_path,
            doc_id=doc_id,
            schema_name="app_schema",
            loader_config=loader_config
        )
        
        # Update metadata with chunk count
        if not result.errors:
            metadata["num_chunks"] = result.num_chunks
            store_document(team_id, uploaded_file.name, metadata, doc_id=doc_id)
        
        # Display results
        if result.errors:
            st.error("❌ Processing failed:")
            for error in result.errors:
                st.error(f"  • {error}")
        else:
            st.success("✅ Document processed successfully!")
            st.write(f"Document ID: {result.doc_id}")
            st.write(f"Number of chunks: {result.num_chunks}")
            
            if result.warnings:
                st.warning("⚠️ Warnings:")
                for warning in result.warnings:
                    st.warning(f"  • {warning}")
            
    finally:
        # Clean up temp file
        os.unlink(tmp_file_path)

def delete_document_and_flag(doc_id, team_id):
    delete_document(doc_id, team_id)
    st.session_state['rerun_after_delete'] = True

def main():
    # Check authentication
    check_auth()
    
    # Get team ID from session
    team_id = st.session_state.team_id
    is_admin = st.session_state.is_admin
    
    st.title("Document Management")
    st.markdown("""
    Upload, process, and manage documents in the knowledge base.
    """)
    
    # Initialize services
    initialize_services()
    
    # Document upload section
    st.header("Upload Documents")
    
    # Chunking options in columns
    col1, col2 = st.columns(2)
    
    with col1:
        chunk_strategy = st.selectbox(
            "Chunking Strategy",
            ["paragraph", "fixed_size", "semantic"],
            help="""Choose how to split your document:
- paragraph: Recursively splits on paragraph breaks, then sentences, then words (recommended for most texts)
- fixed_size: Splits by fixed character count only (no regard for paragraphs or sentences)
- semantic: Uses AI to split based on meaning and context (no fixed size)""",
            format_func=lambda x: {
                "paragraph": "Paragraph-based (Recursive)",
                "fixed_size": "Fixed Size (by Character Count)",
                "semantic": "Semantic (AI-based)"
            }.get(x, x)
        )
    
    with col2:
        if chunk_strategy != "semantic":
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=1000,
                help="Target size of each chunk in characters (for fixed_size, this is the exact chunk size)"
            )
            
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=500,
                value=200,
                help="Number of characters to overlap between chunks"
            )
        else:
            st.info("Semantic chunking automatically determines optimal chunk sizes based on content meaning.")
            chunk_size = None
            chunk_overlap = None
    
    # File uploader with clear visual separation
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["txt", "md", "pdf", "docx", "csv"],
        help="Select a document to process. For CSVs, each row will be treated as a chunk. See https://python.langchain.com/docs/how_to/document_loader_csv/ for details."
    )
    
    if uploaded_file:
        # Show file info before processing
        st.write(f"File: {uploaded_file.name}")
        with st.expander("View file details"):
            st.write(f"File type: {uploaded_file.type}")
            st.write(f"File size: {uploaded_file.size} bytes")
        
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                process_uploaded_file(uploaded_file, team_id, chunk_strategy, chunk_size, chunk_overlap)
    
    # Document list section with clear separation
    st.markdown("---")
    st.header("Uploaded Documents")
    
    # Get documents for the team or all if admin
    if is_admin:
        documents = get_all_documents()
        # Extract unique team IDs for filter
        team_ids = sorted({doc['team_id'] for doc in documents})
        team_filter = st.selectbox("Filter by Team ID", options=["All Teams"] + team_ids, index=0)
        if team_filter != "All Teams":
            documents = [doc for doc in documents if doc['team_id'] == team_filter]
    else:
        documents = get_team_documents(team_id)
    
    if not documents:
        st.info("No documents uploaded yet.")
    else:
        # Prepare document data for the table
        doc_rows = []
        for doc in documents:
            doc_rows.append({
                "Filename": doc["filename"],
                "Document ID": doc["doc_id"],
                **({"Team": doc["team_id"]} if is_admin else {}),
                "Chunks": doc.get("metadata", {}).get("num_chunks", "Processing..."),
                "Uploaded": doc.get("created_at", "")
            })
        df_docs = pd.DataFrame(doc_rows)
        st.dataframe(df_docs, use_container_width=True, hide_index=True)
        # Select a document for actions
        if len(df_docs) > 0:
            selected_idx = st.selectbox(
                "Select a document to view actions",
                options=df_docs.index,
                format_func=lambda i: f"{df_docs.loc[i, 'Filename']} ({df_docs.loc[i, 'Document ID'][:8]}...)"
            )
            selected_doc = documents[selected_idx]
            colA, colB = st.columns(2)
            with colA:
                if st.button("📑 View Chunks", key="view_chunks_main"):
                    st.session_state['view_chunks_doc_id'] = selected_doc["doc_id"]
            with colB:
                if st.button("🗑️ Delete Document", key="delete_doc_main"):
                    delete_document_and_flag(selected_doc["doc_id"], selected_doc.get("team_id"))

    # After rendering the document list, check for rerun flag
    if st.session_state.get('rerun_after_delete'):
        st.session_state['rerun_after_delete'] = False
        st.rerun()

    # Show chunk viewer below the document list if a document is selected
    doc_id = st.session_state.get('view_chunks_doc_id')
    if doc_id:
        st.markdown("---")
        st.header("Document Chunks")
        chunks = get_document_chunks(doc_id)
        if not chunks:
            st.info("No chunks found for this document.")
        else:
            chunk_rows = [
                {
                    "Index": idx,
                    "Chunk ID": chunk["chunk_id"],
                    "Preview": chunk["text"][:100] + ("..." if len(chunk["text"]) > 100 else "")
                }
                for idx, chunk in enumerate(chunks)
            ]
            df = pd.DataFrame(chunk_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            selected_idx = st.selectbox(
                "Select a chunk to view details",
                options=df["Index"],
                format_func=lambda x: f"Chunk {x}"
            )
            selected_chunk = chunks[selected_idx]
            st.markdown(f"**Chunk ID:** `{selected_chunk['chunk_id']}`")
            st.code(selected_chunk["text"], language=None, line_numbers=True)
        if st.button("Close Chunk Viewer"):
            st.session_state['view_chunks_doc_id'] = None

if __name__ == "__main__":
    main() 