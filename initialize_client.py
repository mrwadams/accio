from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the client with your API key
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Example usage
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text) 