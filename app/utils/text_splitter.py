from typing import List, Dict, Any, Optional, Union
import logging
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
    HTMLHeaderTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class TextSplitter:
    """
    A modular text splitter that supports various splitting strategies using Langchain text splitters.
    """
    
    # Define available splitter types
    SPLITTER_TYPES = {
        "recursive": RecursiveCharacterTextSplitter,
        "character": CharacterTextSplitter,
        "token": TokenTextSplitter,
        "markdown": MarkdownTextSplitter,
        "html_header": HTMLHeaderTextSplitter,
        "semantic": SemanticChunker,
    }
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean text by removing problematic characters and normalizing whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
            
        # Remove null characters
        text = text.replace('\0', '')
        
        # Replace various types of whitespace with standard spaces
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Normalize unicode whitespace
        text = ' '.join(text.split())
        
        # Remove other potential problematic characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Normalize line endings
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        
        return text
    
    @staticmethod
    def split_text(
        text: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = "recursive",
        embeddings: Optional[Embeddings] = None
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks using the specified splitter type and parameters.
        
        Args:
            text (str): The text to split
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            splitter_type (str): Type of splitter to use ('recursive', 'character', 'token', 'semantic')
            embeddings (Optional[Embeddings]): Required for semantic chunking - embeddings model to use
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - content: str, the text chunk
                - metadata: Dict containing chunk index and any other metadata
        """
        try:
            # Clean the text first
            text = TextSplitter._clean_text(text)
            if not text:
                logger.warning("Text was empty after cleaning")
                return []
                
            # Initialize the appropriate splitter
            if splitter_type == "semantic":
                if embeddings is None:
                    logger.warning("Embeddings model required for semantic chunking. Falling back to recursive.")
                    splitter_type = "recursive"
                else:
                    # Use more balanced settings for semantic chunking
                    splitter = SemanticChunker(
                        embeddings=embeddings,
                        breakpoint_threshold_type="percentile",
                        breakpoint_threshold_amount=60.0,  # Lower threshold for more balanced chunks
                        buffer_size=3,  # Smaller buffer for more precise splits
                        sentence_split_regex=r"(?<=[.!?])\s+",  # Split on sentence endings
                        add_start_index=True,  # Track position in original text
                        number_of_chunks=None  # Let it determine based on content
                    )
            
            if splitter_type == "recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    # Use default separators that preserve semantic units
                    separators=["\n\n", "\n", " ", ""]
                )
            elif splitter_type == "character":
                splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    separator="",  # Pure character-based splitting
                    is_separator_regex=False
                )
            elif splitter_type == "token":
                splitter = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            
            # Split the text into documents
            documents = splitter.create_documents([text])
            
            # Convert to our expected format
            chunks = []
            for i, doc in enumerate(documents):
                chunks.append({
                    "content": doc.page_content,
                    "metadata": {
                        "chunk_index": i,
                        **doc.metadata
                    }
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text: {str(e)}")
            # Return the original text as a single chunk in case of error
            return [{
                "content": text,
                "metadata": {
                    "chunk_index": 0,
                    "error": str(e)
                }
            }]
    
    @staticmethod
    def split_documents(
        documents: List[Dict[str, Any]],
        splitter_type: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        length_function: Optional[callable] = None,
        is_separator_regex: bool = False,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Split multiple documents into chunks.
        
        Args:
            documents (List[Dict[str, Any]]): List of document dictionaries
            splitter_type (str): Type of splitter to use
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            separator (str): Separator to use for splitting
            length_function (callable, optional): Function to calculate length of text
            is_separator_regex (bool): Whether the separator is a regex pattern
            **kwargs: Additional arguments to pass to the splitter
            
        Returns:
            List[Dict[str, Any]]: List of document chunks
        """
        all_chunks = []
        
        for doc in documents:
            if "content" not in doc:
                logger.warning(f"Skipping document without content: {doc}")
                continue
                
            # Split the document
            chunks = TextSplitter.split_text(
                text=doc["content"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                splitter_type=splitter_type
            )
            
            # Add document metadata to each chunk
            for chunk in chunks:
                chunk["metadata"].update({
                    "source_document": doc.get("metadata", {}).get("filename", "unknown"),
                    **doc.get("metadata", {})
                })
            
            all_chunks.extend(chunks)
        
        return all_chunks 