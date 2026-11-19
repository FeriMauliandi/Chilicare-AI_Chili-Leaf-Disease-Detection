from langchain_chroma import Chroma
from src.config import (
    CHROMA_DB_DIR,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    GOOGLE_API_KEY,
)
from src.ingestion.embedder import get_embedding_model

COLLECTION_NAME = "chilicare_kb"

def get_embedder():
    return get_embedding_model(
        model_name=EMBEDDING_MODEL_NAME,
        dimensions=EMBEDDING_DIMENSIONS,
        api_key=GOOGLE_API_KEY,
    )

def get_vector_store():
    embedder = get_embedder()
    return Chroma(
        persist_directory=str(CHROMA_DB_DIR), 
        embedding_function=embedder,
        collection_name=COLLECTION_NAME
    )
