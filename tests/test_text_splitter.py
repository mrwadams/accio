import sys
import os
import unittest
from pathlib import Path

# Add the parent directory to the path so we can import the app modules
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.text_splitter import TextSplitter
from app.utils.document_loader import DocumentLoader

class TestTextSplitter(unittest.TestCase):
    """Test cases for the TextSplitter class."""
    
    def setUp(self):
        """Set up test data."""
        # Create a longer text with clear paragraph breaks
        self.sample_text = """This is a sample text for testing the text splitter.
It contains multiple paragraphs that should be split into chunks.

This is the second paragraph with some more text.
It should be split along with the first paragraph.

This is the third paragraph with even more text.
It should also be split into appropriate chunks.

This is the fourth paragraph with additional content.
It provides more text to ensure we get multiple chunks.

This is the fifth paragraph with even more content.
It adds more text to make sure we have enough to split."""
        
        # Create a temporary test file
        self.test_file_path = Path(__file__).parent / "test_document.txt"
        with open(self.test_file_path, "w") as f:
            f.write(self.sample_text)
            
        print("\nSample text length:", len(self.sample_text))
        print("Sample text:")
        print(self.sample_text)
    
    def tearDown(self):
        """Clean up test data."""
        if self.test_file_path.exists():
            self.test_file_path.unlink()
    
    def test_split_text_recursive(self):
        """Test splitting text using the recursive splitter."""
        print("\nTesting recursive splitter:")
        chunks = TextSplitter.split_text(
            text=self.sample_text,
            splitter_type="recursive",
            chunk_size=150,  # Smaller chunk size to ensure multiple chunks
            chunk_overlap=20
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}:")
            print(f"  Length: {len(chunk['content'])}")
            print(f"  Content: {chunk['content']}")
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk has content and metadata
        for chunk in chunks:
            self.assertIn("content", chunk)
            self.assertIn("metadata", chunk)
            self.assertIn("chunk_index", chunk["metadata"])
    
    def test_split_text_character(self):
        """Test splitting text using the character splitter."""
        print("\nTesting character splitter:")
        chunks = TextSplitter.split_text(
            text=self.sample_text,
            splitter_type="character",
            chunk_size=150,  # Smaller chunk size to ensure multiple chunks
            chunk_overlap=20
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}:")
            print(f"  Length: {len(chunk['content'])}")
            print(f"  Content: {chunk['content']}")
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
    
    def test_split_documents(self):
        """Test splitting multiple documents."""
        print("\nTesting document splitting:")
        # Load a document
        doc_result = DocumentLoader.load_document(str(self.test_file_path))
        self.assertIsNone(doc_result["error"])
        
        print("Document content length:", len(doc_result["content"]))
        
        # Split the document
        chunks = TextSplitter.split_documents(
            documents=[doc_result],
            splitter_type="recursive",
            chunk_size=150,  # Smaller chunk size to ensure multiple chunks
            chunk_overlap=20
        )
        
        print(f"Number of chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i}:")
            print(f"  Length: {len(chunk['content'])}")
            print(f"  Content: {chunk['content']}")
        
        # Check that we got multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that each chunk has the source document metadata
        for chunk in chunks:
            self.assertIn("source_document", chunk["metadata"])
            self.assertEqual(chunk["metadata"]["source_document"], "test_document.txt")

if __name__ == "__main__":
    unittest.main() 