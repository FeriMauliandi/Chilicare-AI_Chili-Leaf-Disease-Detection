from pathlib import Path
from src import config

def test_config_variables_exist():
    """Verify that all required config variables are defined with appropriate defaults."""
    assert hasattr(config, "BASE_DIR")
    assert hasattr(config, "DATA_DIR")
    assert hasattr(config, "CHROMA_DB_DIR")
    assert hasattr(config, "GOOGLE_API_KEY")
    assert hasattr(config, "OPENROUTER_API_KEY")
    assert hasattr(config, "OPENROUTER_BASE_URL")
    assert hasattr(config, "LLM_MODEL_NAME")
    assert hasattr(config, "LLM_TEMPERATURE")
    assert hasattr(config, "EMBEDDING_MODEL_NAME")
    assert hasattr(config, "EMBEDDING_DIMENSIONS")
    assert hasattr(config, "RERANKER_MODEL_NAME")
    assert hasattr(config, "RERANKER_TIMEOUT")

def test_config_defaults():
    """Verify default configurations match requirements."""
    assert config.OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"
    assert config.LLM_MODEL_NAME == "openai/gpt-oss-20b"
    assert config.LLM_TEMPERATURE == 0.1
    assert config.EMBEDDING_MODEL_NAME == "models/gemini-embedding-001"
    assert config.EMBEDDING_DIMENSIONS == 768
    assert config.RERANKER_MODEL_NAME == "cohere/rerank-v3.5"
    assert config.RERANKER_TIMEOUT == 8

def test_config_paths():
    """Verify BASE_DIR, DATA_DIR and CHROMA_DB_DIR are valid Path objects pointing to project folders."""
    assert isinstance(config.BASE_DIR, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.CHROMA_DB_DIR, Path)
    assert config.DATA_DIR.name == "data"
    assert config.CHROMA_DB_DIR.name == "chroma_db"
    assert config.DATA_DIR == config.BASE_DIR / "data"
    assert config.CHROMA_DB_DIR == config.BASE_DIR / "chroma_db"
    assert config.DATA_DIR.exists()
