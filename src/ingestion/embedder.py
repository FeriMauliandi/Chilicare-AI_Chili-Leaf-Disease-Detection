from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIMENSIONS,
    GOOGLE_API_KEY,
)

def get_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
    dimensions: int = EMBEDDING_DIMENSIONS,
    api_key: str = None,
):
    key = api_key or GOOGLE_API_KEY
    if not key:
        print("Peringatan: GOOGLE_API_KEY atau GEMINI_API_KEY tidak ditemukan. Menggunakan dummy key untuk inisialisasi.")
        key = "dummy_key_for_test"

    print(f"Mempersiapkan model embedding Google Gemini: {model_name} (dimensi: {dimensions})")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=key,
        output_dimensionality=dimensions,
    )
    return embeddings
