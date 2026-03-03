import os
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")

response = client.models.generate_content(model="gemma-3-27b-it",contents="Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.")

print(response.text)

usage = response.usage_metadata
print(f"\n Prompt tokens: {usage.prompt_token_count}")
print(f"Response tokens: {usage.candidates_token_count}")
