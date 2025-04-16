"""
Test document loader script to populate the vector store with sample documents
for testing RAG capabilities.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.utils.embeddings import EmbeddingService, VectorStore, DocumentChunk
from app.utils.text_splitter import TextSplitter

# Load environment variables
load_dotenv()

# Initialize services
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
embedding_service = EmbeddingService(client)
vector_store = VectorStore({
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
})

# Sample test documents
TEST_DOCS = [
    {
        'doc_id': 'python_basics',
        'content': """
Python is a high-level, interpreted programming language. It was created by Guido van Rossum
and released in 1991. Python's design philosophy emphasizes code readability with its notable
use of significant whitespace. Its language constructs and object-oriented approach aim to
help programmers write clear, logical code for small and large-scale projects.

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms,
including structured (particularly procedural), object-oriented, and functional programming.
Python is often described as a "batteries included" language due to its comprehensive standard library.
        """.strip(),
        'metadata': {
            'title': 'Python Programming Basics',
            'source': 'test_docs',
            'type': 'programming'
        }
    },
    {
        'doc_id': 'machine_learning_intro',
        'content': """
Machine learning is a subset of artificial intelligence that focuses on developing systems
that can learn from and make decisions based on data. Instead of following strictly static
program instructions, machine learning algorithms build a model based on sample data, known
as training data, to make predictions or decisions without being explicitly programmed to do so.

The process of machine learning involves several key steps:
1. Data Collection: Gathering relevant data for the problem
2. Data Preparation: Cleaning and preprocessing the data
3. Model Selection: Choosing the appropriate algorithm
4. Training: Teaching the model using the prepared data
5. Evaluation: Testing the model's performance
6. Deployment: Using the model in real-world applications
        """.strip(),
        'metadata': {
            'title': 'Introduction to Machine Learning',
            'source': 'test_docs',
            'type': 'ai'
        }
    },
    {
        'doc_id': 'streamlit_guide',
        'content': """
Streamlit is an open-source Python library that makes it easy to create custom web apps
for machine learning and data science. It allows you to turn data scripts into shareable
web apps in minutes, not weeks. All you need to do is write Python scripts.

Key features of Streamlit include:
- Simple and intuitive API
- Support for charts and interactive widgets
- Caching mechanism for performance
- Live reloading during development
- Easy deployment options
- Integration with popular data science libraries

To create a basic Streamlit app, you start by importing the library:
import streamlit as st

Then you can add elements like:
st.title("My App")  # Adds a title
st.write("Hello World")  # Writes text
st.dataframe(data)  # Displays a dataframe
        """.strip(),
        'metadata': {
            'title': 'Streamlit Development Guide',
            'source': 'test_docs',
            'type': 'development'
        }
    }
]

def load_test_documents():
    """Load test documents into the vector store."""
    print("Loading test documents...")
    
    for doc in TEST_DOCS:
        print(f"\nProcessing document: {doc['doc_id']}")
        
        # Split the document into chunks using TextSplitter
        chunks = TextSplitter.split_text(
            text=doc['content'],
            chunk_size=500,  # Smaller chunks for testing
            chunk_overlap=50
        )
        
        # Create DocumentChunk objects
        doc_chunks = [
            DocumentChunk(
                text=chunk['content'],
                doc_id=doc['doc_id'],
                metadata={**doc['metadata'], **chunk['metadata']},
                chunk_index=chunk['metadata']['chunk_index']
            )
            for chunk in chunks
        ]
        
        # Generate embeddings and store in vector store
        embedded_chunks = embedding_service.embed_chunks(doc_chunks)
        vector_store.store_chunks(embedded_chunks)
        
        print(f"Successfully processed {len(embedded_chunks)} chunks for {doc['doc_id']}")
    
    print("\nAll test documents loaded successfully!")

if __name__ == "__main__":
    load_test_documents() 