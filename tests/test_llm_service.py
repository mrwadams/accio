"""
Tests for the LLM service implementation.
Verifies prompt construction, generation config, and error handling.
"""

import os
import pytest
from unittest.mock import Mock, patch
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.utils.llm_service import LLMService, RetrievedContext

# Load environment variables
load_dotenv()

@pytest.fixture
def real_client():
    """Create a real Google GenAI client."""
    return genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

@pytest.fixture
def llm_service(real_client):
    """Create an LLMService instance with the real client."""
    return LLMService(client=real_client)

@pytest.fixture
def mock_client():
    """Create a mock client for error testing."""
    return Mock(spec=genai.Client)

@pytest.fixture
def mock_llm_service(mock_client):
    """Create an LLMService instance with a mock client for error testing."""
    return LLMService(client=mock_client)

def test_construct_prompt_without_context(llm_service):
    """Test prompt construction without context."""
    query = "What is the capital of France?"
    prompt = llm_service._construct_prompt(query)
    
    assert isinstance(prompt, dict)
    assert "contents" in prompt
    assert "system_instruction" in prompt
    assert "User query: What is the capital of France?" in prompt["contents"]
    assert "You are a helpful AI assistant" in prompt["system_instruction"]

def test_construct_prompt_with_context(llm_service):
    """Test prompt construction with context."""
    query = "What is the capital of France?"
    contexts = [
        RetrievedContext(
            doc_id="doc1",
            content="Paris is the capital of France.",
            metadata={"source": "geography.txt"},
            score=0.95
        )
    ]
    
    prompt = llm_service._construct_prompt(query, contexts)
    
    assert isinstance(prompt, dict)
    assert "[doc1]: Paris is the capital of France." in prompt["contents"]
    assert "Relevant context:" in prompt["contents"]
    assert "User query:" in prompt["contents"]

def test_generate_response_real(llm_service):
    """Test response generation with the real client."""
    query = "What is 2+2? Answer with just the number."
    response = llm_service.generate_response(query)
    
    assert response is not None
    assert "4" in response  # Simple arithmetic test

def test_generate_response_with_context_real(llm_service):
    """Test response generation with context using the real client."""
    query = "What is the capital mentioned in the text?"
    contexts = [
        RetrievedContext(
            doc_id="doc1",
            content="Paris is the capital of France.",
            metadata={"source": "geography.txt"},
            score=0.95
        )
    ]
    
    response = llm_service.generate_response(query, contexts=contexts)
    
    assert response is not None
    assert "Paris" in response
    assert "[doc1]" in response  # Should cite the source

def test_generate_response_stream_real(llm_service):
    """Test streaming response generation with the real client."""
    query = "Count from 1 to 3."
    stream_gen = llm_service.generate_response_stream(query)
    
    # Collect the streamed response
    full_response = ""
    for part in stream_gen:
        assert isinstance(part, str)
        full_response += part
    
    assert full_response is not None
    assert len(full_response) > 0
    assert any(str(i) in full_response for i in range(1, 4))

def test_custom_generation_params_real(llm_service):
    """Test generation with custom parameters using the real client."""
    query = "Write a very short story about a cat."
    response = llm_service.generate_response(
        query,
        temperature=0.1,  # Low temperature for more focused output
        max_output_tokens=50  # Limit length
    )
    
    assert response is not None
    assert len(response.split()) < 50  # Rough check for length limit
    # Check for cat-related words since the story might use pronouns or names
    assert any(word in response.lower() for word in ["cat", "kitten", "feline", "paw", "whisker", "meow", "purr"])

# Error handling tests with mock client
def test_generate_response_with_retry(mock_llm_service, mock_client):
    """Test response generation with retry on failure."""
    mock_response = Mock()
    mock_response.text = "Success after retry"
    mock_client.models.generate_content.side_effect = [
        Exception("First attempt failed"),
        mock_response
    ]
    
    response = mock_llm_service.generate_response("test query")
    
    assert response == "Success after retry"
    assert mock_client.models.generate_content.call_count == 2

def test_generate_response_max_retries_exceeded(mock_llm_service, mock_client):
    """Test response generation when max retries are exceeded."""
    mock_client.models.generate_content.side_effect = Exception("Failed")
    
    response = mock_llm_service.generate_response("test query")
    
    assert response is None
    assert mock_client.models.generate_content.call_count == 3  # Initial + 2 retries

def test_custom_model_name_real(real_client):
    """Test LLMService initialization with custom model name."""
    service = LLMService(client=real_client, model="gemini-2.0-flash")
    
    response = service.generate_response("What is 1+1? Answer with just the number.")
    
    assert response is not None
    assert "2" in response 