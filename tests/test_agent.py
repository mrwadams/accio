import pytest
from app.components.agent import Agent, RetrievalAssessment, QueryRewrite
from app.utils import genai_client
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load environment variables and initialize client
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@pytest.fixture
def agent():
    return Agent(max_retries=2, temperature=0.2)

@pytest.mark.asyncio
async def test_assess_retrieval_sufficient(agent):
    query = "What is RAG?"
    chunks = [
        "RAG (Retrieval Augmented Generation) is a technique that enhances large language models by retrieving relevant information from a knowledge base before generating responses.",
        "It combines retrieval with generation to provide more accurate and factual responses based on specific knowledge sources."
    ]
    
    result = await agent.assess_retrieval(query, chunks)
    
    assert isinstance(result, RetrievalAssessment)
    assert result.is_sufficient == True
    assert len(result.reasoning) > 0
    assert result.needs_clarification == False

@pytest.mark.asyncio
async def test_assess_retrieval_needs_clarification(agent):
    query = "What were the key developments?"
    chunks = [
        "There were several developments across different time periods.",
        "Many changes occurred in various areas of the system."
    ]
    
    result = await agent.assess_retrieval(query, chunks)
    
    assert isinstance(result, RetrievalAssessment)
    # The model should recognize this needs clarification
    assert result.needs_clarification == True
    assert len(result.reasoning) > 0
    # The clarification should ask for more specifics about either time, system, or area
    assert result.suggested_clarification is not None
    assert any(word in result.suggested_clarification.lower() 
              for word in ["time", "period", "system", "area", "specific"])

@pytest.mark.asyncio
async def test_rewrite_query(agent):
    query = "How does RAG work?"
    feedback = "Previous retrieval was too general and didn't cover the technical implementation details"
    
    result = await agent.rewrite_query(query, feedback)
    
    assert isinstance(result, QueryRewrite)
    assert len(result.rewritten_query) > len(query)  # Should be more specific
    assert len(result.reasoning) > 0
    assert result.original_query == query

@pytest.mark.asyncio
async def test_process_query_successful(agent):
    # Define test retrieval function
    async def test_retrieval(query):
        return [
            "RAG (Retrieval Augmented Generation) is a technique that enhances LLMs by retrieving relevant information before generating responses.",
            "The key components of RAG include: 1) A retrieval system that finds relevant documents, 2) An embedding model that converts text to vectors, 3) A vector database for storing and searching embeddings, and 4) An LLM that generates responses using the retrieved context."
        ]
    
    # Define test generation function
    async def test_generation(query, chunks):
        return "RAG is a technique that combines retrieval and generation to provide more accurate responses."

    result = await agent.process_query(
        query="What is RAG and what are its components?",
        retrieval_func=test_retrieval,
        generation_func=test_generation,
        schema_config={}
    )
    
    assert result["success"] == True
    assert len(result["response"]) > 0
    assert len(result["process_log"]) > 0
    assert any(log["step"] == "assessment" for log in result["process_log"])

@pytest.mark.asyncio
async def test_process_query_needs_clarification(agent):
    # Define test retrieval function
    async def test_retrieval(query):
        return [
            "The system underwent various changes and improvements.",
            "Multiple features were added across different versions."
        ]
    
    # Define test generation function that shouldn't be called
    async def test_generation(query, chunks):
        return "This shouldn't be called"

    result = await agent.process_query(
        query="What changes were made to the system?",
        retrieval_func=test_retrieval,
        generation_func=test_generation,
        schema_config={}
    )
    
    assert result["success"] == False
    assert result.get("needs_clarification") == True
    assert "clarification_question" in result
    assert len(result["process_log"]) > 0

@pytest.mark.asyncio
async def test_process_query_max_retries(agent):
    # Define test retrieval function that always returns insufficient info
    async def test_retrieval(query):
        return [
            "Some very general information about systems.",
            "More general information without specific details."
        ]

    # Define test generation function that shouldn't be called
    async def test_generation(query, chunks):
        return "This shouldn't be called"

    result = await agent.process_query(
        query="Tell me about the system architecture",
        retrieval_func=test_retrieval,
        generation_func=test_generation,
        schema_config={}
    )

    assert result["success"] == False
    # The model should either hit max retries or ask for clarification
    assert (
        ("max" in result.get("error", "").lower() and "retries" in result.get("error", "").lower()) or
        (result.get("needs_clarification") == True and result.get("clarification_question") is not None)
    )
    assert len(result["process_log"]) > 0
    # Should have attempted at least once
    assert any(log["step"] == "attempt" for log in result["process_log"])

@pytest.mark.asyncio
async def test_error_handling(agent):
    # Define test retrieval function that raises an exception
    async def test_retrieval(query):
        raise Exception("Simulated retrieval error")
    
    # Define test generation function
    async def test_generation(query, chunks):
        return "This shouldn't be called"

    result = await agent.process_query(
        query="What is RAG?",
        retrieval_func=test_retrieval,
        generation_func=test_generation,
        schema_config={}
    )
    
    assert result["success"] == False
    assert len(result["process_log"]) > 0
    assert any(log["step"] == "error" for log in result["process_log"])
    assert "retrieval error" in str(result["process_log"][-1]["error"]).lower() 