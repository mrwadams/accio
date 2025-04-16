import os
import sys
import unittest
from pathlib import Path
import json

# Add the parent directory to the path so we can import the app module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):
    """Test the DocumentLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test directory
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Create test files
        self.create_test_files()
        
        # Path to sample files
        self.sample_dir = Path("tests/data")
        self.pdf_file = self.sample_dir / "sample.pdf"
        self.docx_file = self.sample_dir / "sample.docx"
        self.csv_file = self.sample_dir / "sample.csv"
        
        # Test schema name
        self.test_schema = "test_knowledge_base"
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove test files
        for file in self.test_dir.glob("*"):
            if file.is_file():
                file.unlink()
        
        # Remove test directory
        self.test_dir.rmdir()
    
    def create_test_files(self):
        """Create test files for testing."""
        # Create a simple text file
        with open(self.test_dir / "test.txt", "w") as f:
            f.write("This is a test text file.\nIt has multiple lines.\nThird line here.")
        
        # Create a simple CSV file with more data
        with open(self.test_dir / "test.csv", "w") as f:
            f.write("name,age,city\nJohn,30,New York\nJane,25,Boston\nBob,35,Chicago")
    
    def verify_base_metadata(self, metadata, expected_file_type, schema_name=None):
        """Helper to verify base metadata fields."""
        self.assertIn("document_id", metadata)
        self.assertIsInstance(metadata["document_id"], str)
        self.assertEqual(metadata["file_type"], expected_file_type)
        self.assertIn("file_size", metadata)
        self.assertIn("created_time", metadata)
        self.assertIn("modified_time", metadata)
        self.assertIn("source_range_type", metadata)
        if schema_name:
            self.assertEqual(metadata["schema_name"], schema_name)
    
    def verify_chunk_metadata(self, chunk, doc_id, schema_name=None):
        """Helper to verify chunk metadata fields."""
        self.assertIn("content", chunk)
        self.assertIn("metadata", chunk)
        chunk_metadata = chunk["metadata"]
        self.assertEqual(chunk_metadata["document_id"], doc_id)
        self.assertIn("chunk_index", chunk_metadata)
        self.assertIn("source_range_type", chunk_metadata)
        self.assertIn("source_range", chunk_metadata)
        if schema_name:
            self.assertEqual(chunk_metadata["schema_name"], schema_name)
    
    def test_load_txt_file_with_schema(self):
        """Test loading a text file with schema."""
        result = DocumentLoader.load_document(
            str(self.test_dir / "test.txt"),
            schema_name=self.test_schema
        )
        
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("chunks", result)
        self.assertIsNone(result["error"])
        
        # Verify content
        self.assertIn("This is a test text file", result["content"])
        
        # Verify metadata
        self.verify_base_metadata(result["metadata"], "txt", self.test_schema)
        self.assertEqual(result["metadata"]["source_range_type"], "line")
        
        # Verify chunks
        self.assertGreater(len(result["chunks"]), 0)
        for chunk in result["chunks"]:
            self.verify_chunk_metadata(
                chunk, 
                result["metadata"]["document_id"],
                self.test_schema
            )
    
    def test_load_csv_file_with_schema(self):
        """Test loading a CSV file with schema and verifying detailed metadata."""
        result = DocumentLoader.load_document(
            str(self.test_dir / "test.csv"),
            schema_name=self.test_schema
        )
        
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("chunks", result)
        self.assertIsNone(result["error"])
        
        # Verify metadata
        metadata = result["metadata"]
        self.verify_base_metadata(metadata, "csv", self.test_schema)
        self.assertEqual(metadata["source_range_type"], "row")
        self.assertEqual(metadata["num_rows"], 3)
        self.assertEqual(metadata["num_columns"], 3)
        self.assertEqual(set(metadata["column_names"]), {"name", "age", "city"})
        self.assertEqual(metadata["source_range"], "rows 1-3")
        
        # Verify chunks
        self.assertEqual(len(result["chunks"]), 3)  # One chunk per row
        for chunk in result["chunks"]:
            self.verify_chunk_metadata(
                chunk,
                metadata["document_id"],
                self.test_schema
            )
    
    def test_load_pdf_file_with_schema(self):
        """Test loading a PDF file with schema and verifying detailed metadata."""
        if not self.pdf_file.exists():
            self.skipTest(f"Sample PDF file not found: {self.pdf_file}")
            
        result = DocumentLoader.load_document(
            str(self.pdf_file),
            schema_name=self.test_schema
        )
        
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("chunks", result)
        self.assertIsNone(result["error"])
        
        # Verify metadata
        metadata = result["metadata"]
        self.verify_base_metadata(metadata, "pdf", self.test_schema)
        self.assertEqual(metadata["source_range_type"], "page")
        self.assertIn("num_pages", metadata)
        self.assertIn("pdf_info", metadata)
        self.assertRegex(metadata["source_range"], r"pages 1-\d+")
        
        # Verify chunks
        self.assertGreater(len(result["chunks"]), 0)
        for chunk in result["chunks"]:
            self.verify_chunk_metadata(
                chunk,
                metadata["document_id"],
                self.test_schema
            )
            # Verify page-specific metadata
            self.assertRegex(chunk["metadata"]["source_range"], r"page \d+")
    
    def test_load_docx_file_with_schema(self):
        """Test loading a DOCX file with schema and verifying detailed metadata."""
        if not self.docx_file.exists():
            self.skipTest(f"Sample DOCX file not found: {self.docx_file}")
            
        result = DocumentLoader.load_document(
            str(self.docx_file),
            schema_name=self.test_schema
        )
        
        self.assertIsNotNone(result)
        self.assertIn("content", result)
        self.assertIn("metadata", result)
        self.assertIn("chunks", result)
        self.assertIsNone(result["error"])
        
        # Verify metadata
        metadata = result["metadata"]
        self.verify_base_metadata(metadata, "docx", self.test_schema)
        self.assertEqual(metadata["source_range_type"], "paragraph")
        self.assertIn("num_paragraphs", metadata)
        self.assertIn("num_sections", metadata)
        self.assertIn("word_properties", metadata)
        self.assertRegex(metadata["source_range"], r"paragraphs 1-\d+")
        
        # Verify chunks
        self.assertGreater(len(result["chunks"]), 0)
        for chunk in result["chunks"]:
            self.verify_chunk_metadata(
                chunk,
                metadata["document_id"],
                self.test_schema
            )
    
    def test_load_with_custom_config(self):
        """Test loading with custom configuration."""
        custom_config = {
            'min_chars_per_chunk': 100,
            'max_chars_per_chunk': 5000,
            'csv_args': {
                'delimiter': ',',
                'encoding': 'utf-8'
            }
        }
        
        result = DocumentLoader.load_document(
            str(self.test_dir / "test.csv"),
            schema_name=self.test_schema,
            loader_config=custom_config
        )
        
        self.assertIsNotNone(result)
        self.assertIsNone(result["error"])
        
        # The CSV should still load correctly with custom config
        self.assertIn("John", result["content"])
        self.assertIn("Jane", result["content"])
    
    def test_content_validation(self):
        """Test content validation warnings."""
        # Create a very short file
        with open(self.test_dir / "short.txt", "w") as f:
            f.write("Too short")
        
        result = DocumentLoader.load_document(str(self.test_dir / "short.txt"))
        
        self.assertIsNotNone(result)
        self.assertIsNone(result["error"])
        self.assertIn("validation_warning", result["metadata"])
        self.assertIn("too short", result["metadata"]["validation_warning"].lower())
    
    def test_load_documents_from_directory_with_schema(self):
        """Test loading documents from a directory with schema support."""
        results = DocumentLoader.load_documents_from_directory(
            str(self.test_dir),
            schema_name=self.test_schema
        )
        
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # test.txt and test.csv
        
        for result in results:
            self.assertIn("content", result)
            self.assertIn("metadata", result)
            self.assertIn("chunks", result)
            self.assertEqual(result["metadata"]["schema_name"], self.test_schema)
            
            # Verify chunks have schema
            for chunk in result["chunks"]:
                self.assertEqual(
                    chunk["metadata"]["schema_name"],
                    self.test_schema
                )
    
    def test_fallback_loader(self):
        """Test fallback loader functionality."""
        # This would be better with a corrupted PDF that fails primary loader
        # For now, we'll just verify the configuration exists
        self.assertIn('fallback_loader', DocumentLoader.LOADER_CONFIGS['.pdf'])
        self.assertIn('fallback_loader', DocumentLoader.LOADER_CONFIGS['.docx'])

if __name__ == "__main__":
    unittest.main() 