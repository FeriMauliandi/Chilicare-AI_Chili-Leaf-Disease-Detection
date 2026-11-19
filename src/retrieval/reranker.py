from typing import Sequence, Optional
import httpx
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from src.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    RERANKER_MODEL_NAME,
    RERANKER_TIMEOUT,
)

class OpenRouterReranker(BaseDocumentCompressor):
    """
    Reranker compressor yang memanfaatkan OpenRouter /rerank endpoint
    dengan model cross-encoder seperti Cohere Rerank 3.5 (cohere/rerank-v3.5).
    """
    model_name: str = Field(default_factory=lambda: RERANKER_MODEL_NAME)
    api_key: Optional[str] = Field(default_factory=lambda: OPENROUTER_API_KEY)
    base_url: str = Field(default_factory=lambda: OPENROUTER_BASE_URL)
    top_n: int = 3
    timeout: int = Field(default_factory=lambda: RERANKER_TIMEOUT)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        doc_list = list(documents)
        if not self.api_key:
            print("Peringatan: OPENROUTER_API_KEY tidak ditemukan. Melewati proses reranking.")
            return doc_list[:self.top_n]

        endpoint = f"{self.base_url.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in doc_list],
            "top_n": self.top_n,
        }

        try:
            with httpx.Client(timeout=float(self.timeout)) as client:
                response = client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            if not results:
                return doc_list[:self.top_n]

            reranked_docs = []
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score")
                if idx is not None and 0 <= idx < len(doc_list):
                    original_doc = doc_list[idx]
                    reranked_docs.append(
                        Document(
                            page_content=original_doc.page_content,
                            metadata={
                                **original_doc.metadata,
                                "relevance_score": score,
                            },
                        )
                    )

            return reranked_docs[:self.top_n] if reranked_docs else doc_list[:self.top_n]

        except Exception as e:  # noqa: BLE001
            print(f"Peringatan: Gagal memanggil OpenRouter rerank ({e}). Menggunakan dokumen retrieval awal.")
            return doc_list[:self.top_n]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []

        doc_list = list(documents)
        if not self.api_key:
            print("Peringatan: OPENROUTER_API_KEY tidak ditemukan. Melewati proses reranking.")
            return doc_list[:self.top_n]

        endpoint = f"{self.base_url.rstrip('/')}/rerank"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": [doc.page_content for doc in doc_list],
            "top_n": self.top_n,
        }

        try:
            async with httpx.AsyncClient(timeout=float(self.timeout)) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()

            results = data.get("results", [])
            if not results:
                return doc_list[:self.top_n]

            reranked_docs = []
            for item in results:
                idx = item.get("index")
                score = item.get("relevance_score")
                if idx is not None and 0 <= idx < len(doc_list):
                    original_doc = doc_list[idx]
                    reranked_docs.append(
                        Document(
                            page_content=original_doc.page_content,
                            metadata={
                                **original_doc.metadata,
                                "relevance_score": score,
                            },
                        )
                    )

            return reranked_docs[:self.top_n] if reranked_docs else doc_list[:self.top_n]

        except Exception as e:  # noqa: BLE001
            print(f"Peringatan: Gagal memanggil async OpenRouter rerank ({e}). Menggunakan dokumen retrieval awal.")
            return doc_list[:self.top_n]
