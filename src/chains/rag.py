import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.chains.prompt import DISEASE_PROMPT_TEMPLATE
from src.retrieval.vector_store import get_vector_store
from src.config import LLM_MODEL_NAME, LLM_TEMPERATURE

load_dotenv()

def get_llm():
    return ChatGroq(
        model=LLM_MODEL_NAME, 
        temperature=LLM_TEMPERATURE,
        api_key=os.getenv("GROQ_API_KEY"),
    )

def generate_narrative(disease_name):
    print(f"Mencari data untuk label: {disease_name}...")
    
    search_query = f"Penjelasan lengkap mengenai penyebab, ciri-ciri gejala, dan cara mengatasi penyakit {disease_name} pada tanaman cabai."
    
    results = []
    try:
        vectorstore = get_vector_store()
        results = vectorstore.similarity_search(
            query=search_query, 
            k=3,
            filter={"label": disease_name}
        )
        if not results:
            results = vectorstore.similarity_search(
                query=search_query,
                k=3,
                filter={"label": disease_name.capitalize()}
            )
    except Exception as e:  # noqa: BLE001
        print(f"Peringatan: Gagal melakukan pencarian vectorstore: {e}")
        results = []

    retrieved_context = ""
    if results:
        retrieved_context = "\n\n".join([doc.page_content for doc in results])
    else:
        # Fallback langsung ke data/data.json jika vectorstore kosong/dimensi mismatch
        json_path = os.path.join("data", "data.json")
        if os.path.exists(json_path):
            try:
                import json
                with open(json_path, "r", encoding="utf-8") as f:
                    all_data = json.load(f)
                matched_key = next((k for k in all_data if k.lower() == disease_name.lower()), None)
                if matched_key:
                    details = all_data[matched_key]
                    ctx_lines = [f"Penyakit/Kondisi: {matched_key}"]
                    if details.get("penyebab"):
                        ctx_lines.append("Penyebab:\n- " + "\n- ".join(details["penyebab"]))
                    if details.get("gejala"):
                        ctx_lines.append("Gejala:\n- " + "\n- ".join(details["gejala"]))
                    if details.get("penanganan"):
                        ctx_lines.append("Penanganan:\n- " + "\n- ".join(details["penanganan"]))
                    if details.get("pencegahan"):
                        ctx_lines.append("Pencegahan:\n- " + "\n- ".join(details["pencegahan"]))
                    retrieved_context = "\n".join(ctx_lines)
            except Exception as e:  # noqa: BLE001
                print(f"Peringatan: Gagal membaca data.json fallback: {e}")

    if not retrieved_context:
        return f"Data penyakit '{disease_name}' tidak ditemukan di database."

    print("Data ditemukan. Menghasilkan narasi dengan LLM...")

    llm = get_llm()
    chain = DISEASE_PROMPT_TEMPLATE | llm

    response = chain.invoke({
        "disease_name": disease_name,
        "context": retrieved_context
    })
    
    return response.content