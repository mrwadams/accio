import os
from dotenv import load_dotenv
from google import genai
from .llm_service import LLMService

# Load environment variables
load_dotenv()

# Initialize Google GenAI client
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize services
llm_service = LLMService(client)

def generate_embedding(text: str) -> list:
    """
    Generate an embedding for the given text using the text-embedding-004 model.
    
    Args:
        text: The text to generate an embedding for
        
    Returns:
        List of floats representing the text embedding
    """
    try:
        model = client.models.get_model("text-embedding-004")
        embedding = model.embed_content(text)
        return embedding.values
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def generate_content(prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.7) -> str:
    """
    Generate content based on the prompt using the specified model.
    Note: model parameter is kept for backward compatibility but ignored as model is configured in LLMService.
    
    Args:
        prompt: The prompt to generate content from
        model: The model to use (default: gemini-2.0-flash) - ignored, configured in LLMService
        temperature: Controls randomness in generation (default: 0.7)
        
    Returns:
        Generated content as string
    """
    return llm_service.generate_response(
        query=prompt,
        contexts=None,  # No context needed for direct generation
        temperature=temperature
    )

def generate_content_stream(prompt: str, model: str = "gemini-2.0-flash", temperature: float = 0.7):
    """
    Generate content using streaming, returning a generator for the generated content.
    Note: model parameter is kept for backward compatibility but ignored as model is configured in LLMService.
    
    Args:
        prompt: The prompt to generate content from
        model: The model to use (default: gemini-2.0-flash) - ignored, configured in LLMService
        temperature: Controls randomness in generation (default: 0.7)
        
    Returns:
        Generator yielding content chunks
    """
    return llm_service.generate_response_stream(
        query=prompt,
        contexts=None,  # No context needed for direct generation
        temperature=temperature
    ) 