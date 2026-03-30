import os
from typing import List

import chromadb
import requests
from dotenv import load_dotenv
from fastembed import TextEmbedding



BASE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(ENV_PATH):
    print(f"Loading .env file from: {ENV_PATH}")
    # Remove BOM if present in .env file
    with open(ENV_PATH, 'r', encoding='utf-8-sig') as env_file:
        content = env_file.read()
    with open(ENV_PATH, 'w', encoding='utf-8') as env_file:
        env_file.write(content)
    load_dotenv(ENV_PATH)
else:
    print(".env file not found at expected path.")
    load_dotenv()
    Api_key = os.getenv("GEMINI_API_KEY")

# Debugging: Read .env file content directly
with open(ENV_PATH, 'r') as env_file:
    print(".env file content:")
    print(env_file.read())

# Debugging: Check if GEMINI_API_KEY is loaded correctly
print(f"Loaded GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

# Must match rag/ingest.py (lightweight ONNX model)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Point to the same Chroma DB created by rag/ingest.py
CHROMA_DB_DIR = os.path.join(
    BASE_DIR,
    "..",
    "rag",
    "chroma_db",
)
COLLECTION_NAME = "resume_chunks"
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)


_embedding_model: TextEmbedding | None = None
_chroma_collection = None


def _get_embedding_model() -> TextEmbedding:
    global _embedding_model
    if _embedding_model is None:
        # Explicitly set the device to CPU to avoid GPU-related issues in environments without GPU support
        _embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, device="cpu")
    return _embedding_model


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR, 
            settings=Settings(anonymized_telemetry=False)
        )
        _chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _chroma_collection


def retrieve_context(question: str, top_k: int = 3) -> List[str]:
    """Embed the question and retrieve the most similar resume chunks."""
    if not question.strip():
        return []

    model = _get_embedding_model()
    collection = _get_chroma_collection()

    query_embedding = next(model.embed([question]))
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    documents = result.get("documents") or []
    if not documents:
        return []
    return documents[0]


def build_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No context."
    prompt = (
        "You are an assistant that answers questions about a person's resume.\n"
        "Use only the information provided in the CONTEXT section to answer.\n"
        "If the answer is not contained in the context, say you do not know.\n\n"
        "CONTEXT:\n"
        f"{context_text}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer in a clear, concise paragraph."
    )
    return prompt


def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before running the application."
        )

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }
    body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt,
                    }
                ]
            }
        ]
    }

    print("--- GEMINI INPUT (prompt sent to API) ---")
    print("--- END GEMINI INPUT ---")

    response = requests.post(GEMINI_API_URL, headers=headers, json=body, timeout=30)
    response.raise_for_status()
    data = response.json()

    try:
        candidates = data.get("candidates", [])
        first = candidates[0]
        content = first.get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "")
        if not text:
            raise KeyError
        return text.strip()
    except (KeyError, IndexError, TypeError):
        return f"Unexpected response format from Gemini API: {data}"


def answer_question(question: str) -> str:
    """High-level RAG pipeline: retrieve context, build prompt, call Gemini."""
    context_chunks = retrieve_context(question, top_k=3)
    if not context_chunks:
        return (
            "I could not find any resume context to answer from. "
            "Make sure the vector store has been built by running rag/ingest.py."
        )

    prompt = build_prompt(question, context_chunks)
    return call_gemini(prompt)