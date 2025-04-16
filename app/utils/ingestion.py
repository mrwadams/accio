"""
Ingestion pipeline for processing and storing documents in the RAG system.
Coordinates document loading, text splitting, embedding generation, and storage.
"""

import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from app.utils.document_loader import DocumentLoader
from app.utils.text_splitter import TextSplitter
from app.utils.embeddings import EmbeddingService, VectorStore, DocumentChunk

logger = logging.getLogger(__name__)

@dataclass
class IngestionResult:
    """Result of document ingestion process."""
    doc_id: str
    filename: str
    num_chunks: int
    metadata: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class IngestionPipeline:
    """Coordinates the document ingestion process."""
    
    # Map UI-friendly names to TextSplitter types
    STRATEGY_MAP = {
        "paragraph": "recursive",
        "fixed_size": "character",
        "semantic": "semantic"
    }
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "paragraph"
    ):
        """Initialize the ingestion pipeline.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Storage for document chunks and embeddings
            chunk_size: Target size for text chunks (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            chunk_strategy: Text splitting strategy ('paragraph', 'fixed_size', 'semantic')
        """
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunk_strategy = chunk_strategy
    
    def process_file(
        self,
        file_path: str,
        schema_name: str,
        doc_id: Optional[str] = None,
        loader_config: Optional[Dict[str, Any]] = None
    ) -> IngestionResult:
        """Process a single file through the ingestion pipeline.
        
        Args:
            file_path: Path to the document file
            schema_name: Name of the schema this document belongs to
            doc_id: Optional document ID (generated if not provided)
            loader_config: Optional configuration for the document loader
            
        Returns:
            IngestionResult containing processing details and any errors/warnings
        """
        errors = []
        warnings = []
        
        try:
            # Generate document ID if not provided
            if not doc_id:
                doc_id = str(uuid.uuid4())
            
            # Extract team_id from loader_config
            team_id = loader_config.get('team_id') if loader_config else None
            if not team_id:
                raise ValueError("team_id is required in loader_config")
            
            # Load and validate document
            doc_result = DocumentLoader.load_document(
                file_path=file_path,
                schema_name=schema_name,
                loader_config=loader_config
            )
            
            if doc_result.get("error"):
                return IngestionResult(
                    doc_id=doc_id,
                    filename=Path(file_path).name,
                    num_chunks=0,
                    metadata=doc_result.get("metadata", {}),
                    errors=[doc_result["error"]],
                    warnings=[]
                )
            
            # Check for validation warnings
            if "validation_warning" in doc_result["metadata"]:
                warnings.append(doc_result["metadata"]["validation_warning"])
            
            # Map UI strategy to TextSplitter type
            splitter_type = self.STRATEGY_MAP.get(
                loader_config.get("chunk_strategy", self.chunk_strategy),
                "recursive"
            )
            
            # Get chunk size from loader_config if provided, else use default
            chunk_size = loader_config.get("chunk_size", self.chunk_size)
            chunk_overlap = loader_config.get("chunk_overlap", self.chunk_overlap)
            
            # Split text into chunks
            if Path(file_path).suffix.lower() == ".csv":
                # Use the chunks from the loader directly for CSVs
                chunks = [
                    {
                        "content": chunk["content"],
                        "metadata": {
                            **chunk["metadata"],
                            "chunk_index": i
                        }
                    }
                    for i, chunk in enumerate(doc_result["chunks"])
                ]
            else:
                chunks = TextSplitter.split_text(
                    text=doc_result["content"],
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    splitter_type=splitter_type,
                    embeddings=self.embedding_service if splitter_type == "semantic" else None
                )
            
            # Create DocumentChunk objects
            doc_chunks = []
            for chunk in chunks:
                # Ensure chunk has required metadata
                chunk_metadata = {
                    **doc_result["metadata"],
                    "schema": schema_name,
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "team_id": team_id
                }
                
                # Add any additional chunk-specific metadata
                if "source_range" in chunk["metadata"]:
                    chunk_metadata["source_range"] = chunk["metadata"]["source_range"]
                
                doc_chunks.append(DocumentChunk(
                    text=chunk["content"],
                    doc_id=doc_id,
                    metadata=chunk_metadata,
                    chunk_index=chunk_metadata["chunk_index"]
                ))
            
            # Generate embeddings and store
            embedded_chunks = self.embedding_service.embed_chunks(doc_chunks)
            self.vector_store.store_chunks(embedded_chunks, team_id)
            
            return IngestionResult(
                doc_id=doc_id,
                filename=Path(file_path).name,
                num_chunks=len(embedded_chunks),
                metadata=doc_result["metadata"],
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return IngestionResult(
                doc_id=doc_id or "error",
                filename=Path(file_path).name,
                num_chunks=0,
                metadata={},
                errors=[str(e)],
                warnings=warnings
            )
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its chunks from storage.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            self.vector_store.delete_document(doc_id)
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False 