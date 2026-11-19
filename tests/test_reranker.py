import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from src.retrieval.reranker import OpenRouterReranker

def test_reranker_empty_documents():
    reranker = OpenRouterReranker(api_key="test_key")
    result = reranker.compress_documents([], "query cabai")
    assert result == []

def test_reranker_no_api_key():
    reranker = OpenRouterReranker(api_key=None, top_n=2)
    docs = [
        Document(page_content="Dokumen 1"),
        Document(page_content="Dokumen 2"),
        Document(page_content="Dokumen 3"),
    ]
    result = reranker.compress_documents(docs, "query cabai")
    assert len(result) == 2
    assert result[0].page_content == "Dokumen 1"

@patch("httpx.Client.post")
def test_reranker_success(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "cohere/rerank-v3.5",
        "results": [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.82},
        ]
    }
    mock_post.return_value = mock_response

    reranker = OpenRouterReranker(api_key="test_key", top_n=2)
    docs = [
        Document(page_content="Dokumen 0", metadata={"source": "A"}),
        Document(page_content="Dokumen 1", metadata={"source": "B"}),
        Document(page_content="Dokumen 2", metadata={"source": "C"}),
    ]

    result = reranker.compress_documents(docs, "cabai kuning")
    assert len(result) == 2
    assert result[0].page_content == "Dokumen 2"
    assert result[0].metadata["relevance_score"] == 0.95
    assert result[0].metadata["source"] == "C"
    assert result[1].page_content == "Dokumen 0"
    assert result[1].metadata["relevance_score"] == 0.82

@patch("httpx.Client.post")
def test_reranker_api_failure_fallback(mock_post):
    mock_post.side_effect = Exception("OpenRouter connection error")

    reranker = OpenRouterReranker(api_key="test_key", top_n=2)
    docs = [
        Document(page_content="Dokumen 0"),
        Document(page_content="Dokumen 1"),
        Document(page_content="Dokumen 2"),
    ]

    result = reranker.compress_documents(docs, "cabai kuning")
    assert len(result) == 2
    assert result[0].page_content == "Dokumen 0"

@pytest.mark.asyncio
async def test_reranker_async_empty():
    reranker = OpenRouterReranker(api_key="test_key")
    result = await reranker.acompress_documents([], "query")
    assert result == []
