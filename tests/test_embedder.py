from src.ingestion.embedder import get_embedding_model
from src.retrieval.vector_store import get_embedder
from src.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSIONS

def test_embedder_initialization():
    emb = get_embedding_model(api_key="dummy_test_key")
    assert emb.model == EMBEDDING_MODEL_NAME
    assert emb.output_dimensionality == EMBEDDING_DIMENSIONS

def test_vector_store_embedder():
    emb = get_embedder()
    assert emb.model == EMBEDDING_MODEL_NAME
    assert emb.output_dimensionality == EMBEDDING_DIMENSIONS
