#!/usr/bin/env python3
"""
API Connection Test Script for RAG Chatbot

This script tests the connection to the Gemini API and verifies that the required
models (gemini-2.0-flash and text-embedding-004) are accessible.
"""

import os
from dotenv import load_dotenv
from google import genai
import time

def test_client_initialization():
    """Test that the client can be initialized with the API key."""
    print("\n=== Testing Client Initialization ===")
    try:
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            print("❌ ERROR: GOOGLE_API_KEY not found in .env file")
            return False
        
        client = genai.Client(api_key=api_key)
        print("✅ Client initialized successfully")
        return client
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize client: {str(e)}")
        return False

def test_generate_content(client):
    """Test the generate_content method with gemini-2.0-flash model."""
    print("\n=== Testing generate_content with gemini-2.0-flash ===")
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash", 
            contents="Explain how AI works in a few words"
        )
        print(f"✅ generate_content successful")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to generate content: {str(e)}")
        return False

def test_embed_content(client):
    """Test the embed_content method with text-embedding-004 model."""
    print("\n=== Testing embed_content with text-embedding-004 ===")
    try:
        response = client.models.embed_content(
            model="text-embedding-004",
            contents="This is a test sentence for embedding."
        )
        # Get the first embedding's values
        embedding_values = response.embeddings[0].values
        print(f"✅ embed_content successful")
        print(f"Embedding dimension: {len(embedding_values)}")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to embed content: {str(e)}")
        return False

def test_streaming(client):
    """Test the streaming functionality with gemini-2.0-flash model."""
    print("\n=== Testing streaming with gemini-2.0-flash ===")
    try:
        response = client.models.generate_content_stream(
            model="gemini-2.0-flash",
            contents="Write a short poem about technology."
        )
        
        print("Streaming response: ", end="")
        for chunk in response:
            print(chunk.text, end="", flush=True)
            time.sleep(0.1)  # Small delay to simulate streaming
        print("\n✅ Streaming successful")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to stream content: {str(e)}")
        return False

def main():
    """Run all tests and report overall status."""
    print("=== RAG Chatbot API Connection Test ===")
    
    client = test_client_initialization()
    if not client:
        print("\n❌ API Connection Test Failed: Could not initialize client")
        return
    
    generate_success = test_generate_content(client)
    embed_success = test_embed_content(client)
    stream_success = test_streaming(client)
    
    print("\n=== Test Summary ===")
    print(f"Client Initialization: {'✅ Passed' if client else '❌ Failed'}")
    print(f"generate_content: {'✅ Passed' if generate_success else '❌ Failed'}")
    print(f"embed_content: {'✅ Passed' if embed_success else '❌ Failed'}")
    print(f"Streaming: {'✅ Passed' if stream_success else '❌ Failed'}")
    
    if client and generate_success and embed_success and stream_success:
        print("\n✅ All API connection tests passed successfully!")
    else:
        print("\n❌ Some API connection tests failed. Please check the errors above.")

if __name__ == "__main__":
    main() 