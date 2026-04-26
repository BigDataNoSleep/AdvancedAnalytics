import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

try:
    response = client.models.embed_content(
        model="models/gemini-embedding-2",
        contents=["test string"]
    )
    print("Success! Embeddings size:", len(response.embeddings[0].values))
except Exception as e:
    print("Error:", e)
