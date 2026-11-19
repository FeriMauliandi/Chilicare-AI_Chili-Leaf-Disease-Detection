# ruff: noqa: E402
import sys
import os
from dotenv import load_dotenv

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(root_dir)

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_classic.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from src.retrieval.vector_store import get_vector_store
from src.retrieval.retriever import get_retriever
from src.retrieval.reranker import OpenRouterReranker
from src.chains.prompt import (
    get_contextualize_q_prompt,
    get_conversational_rag_prompt,
)
from src.config import (
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    RERANKER_MODEL_NAME,
)

load_dotenv()

def create_rag_chain(disease_label=None):
    llm = ChatGroq(
        model=LLM_MODEL_NAME, 
        streaming=True,
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    ) 
    
    vs = get_vector_store()
    base_retriever = get_retriever(
        vector_store=vs, 
        search_type="similarity", 
        k=5,
        filter_label=disease_label 
    )
    
    mq_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    print(f"Memuat model Re-ranker via OpenRouter ({RERANKER_MODEL_NAME})...")
    # Kita hanya ambil 3 dokumen paling relevan (top_n=3) setelah di-rerank
    compressor = OpenRouterReranker(top_n=3)
    
    # BUNGKUS MENJADI COMPRESSION RETRIEVER
    rerank_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=mq_retriever
    )
    
    # History-aware retriever (mengubah percakapan multi-turn menjadi query mandiri jika ada riwayat)
    contextualize_q_prompt = get_contextualize_q_prompt()
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=rerank_retriever,
        prompt=contextualize_q_prompt,
    )
    
    qa_prompt = get_conversational_rag_prompt()
    
    def format_docs(docs):
        print("\n" + "="*50)
        print("🎯 [DEBUG] DOKUMEN TERBAIK SETELAH DI-RERANK & DIVERIFIKASI:")
        for i, doc in enumerate(docs):
            sumber = doc.metadata.get('label') or doc.metadata.get('source') or 'Sumber tidak diketahui'
            skor = doc.metadata.get('relevance_score', 'N/A')
            print(f"  [{i+1}] Topik: {sumber} | Skor Relevansi: {skor}")
        print("="*50 + "\n")
        
        # Gabungkan teks dokumen final
        return "\n\n".join(doc.page_content for doc in docs)

    def normalize_input(x):
        """Mendukung input berupa string langsung maupun dictionary conversational."""
        if isinstance(x, str):
            return {"input": x, "chat_history": []}
        if isinstance(x, dict):
            return {
                "input": x.get("input") or x.get("question", ""),
                "chat_history": x.get("chat_history", []) or []
            }
        return {"input": str(x), "chat_history": []}
        
    rag_chain = (
        RunnableLambda(normalize_input)
        | {
            "context": history_aware_retriever | format_docs,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
