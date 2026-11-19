import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# API Keys Provider Eksternal
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Model konfigurasi default
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Embedding: Google AI Studio Gemini Embedding 1
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# Reranker: cross-encoder via OpenRouter /rerank (Cohere Rerank 3.5)
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", "cohere/rerank-v3.5")
RERANKER_TIMEOUT = int(os.getenv("RERANKER_TIMEOUT", "8"))
