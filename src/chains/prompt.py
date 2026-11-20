from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)

def get_rag_prompt():
    template = """Anda adalah asisten AI spesialis pertanian cabai.
    Gunakan konteks berikut untuk menjawab pertanyaan pengguna.
    Jika jawabannya tidak ada di dalam konteks, katakan dengan jujur bahwa Anda tidak tahu. Jangan mengarang informasi.

    Konteks: {context}

    Pertanyaan: {input}

    Hindari membuat tabel lebih dari 2 kolom. Jika tabel lebih dari 2 kolom, ubah menjadi format daftar.
    Jawaban yang detail dan ramah:"""
    
    return PromptTemplate.from_template(template)


def get_contextualize_q_prompt():
    """
    Prompt untuk merumuskan ulang pertanyaan terbaru dengan mempertimbangkan riwayat percakapan
    menjadi pertanyaan mandiri (standalone question).
    """
    system_prompt = (
        "Mengingat riwayat obrolan dan pertanyaan terbaru pengguna "
        "yang mungkin merujuk pada konteks dalam riwayat obrolan, "
        "rumuskan pertanyaan mandiri (standalone question) yang dapat dipahami "
        "tanpa melihat riwayat obrolan. JANGAN menjawab pertanyaan tersebut, "
        "cukup rumuskan kembali jika perlu, atau kembalikan apa adanya jika sudah jelas."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


def get_conversational_rag_prompt():
    """
    Prompt QA interaktif yang menggabungkan konteks retrieval referensi,
    riwayat percakapan, dan pertanyaan pengguna.
    """
    qa_system_prompt = (
        "Anda adalah asisten AI spesialis pertanian cabai (ChiliCare AI).\n"
        "Gunakan potongan konteks referensi berikut untuk menjawab pertanyaan pengguna.\n"
        "Gunakan juga riwayat obrolan sebelumnya untuk memahami rujukan atau konteks pertanyaan.\n"
        "Jika jawabannya tidak ada di dalam konteks atau di luar topik pertanian cabai, "
        "katakan dengan jujur dan ramah bahwa Anda tidak memiliki informasi tersebut. "
        "Jangan mengarang informasi.\n\n"
        "Konteks Referensi:\n{context}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])


DISEASE_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["disease_name", "context"],
    template="""
Anda adalah penyuluh pertanian cabai.

Gunakan HANYA informasi referensi berikut. Jangan menambahkan fakta baru di luar referensi.

--- REFERENSI DATABASE ---
{context}
--------------------------

Tugas:
Jelaskan tentang kondisi/penyakit "{disease_name}" berdasarkan referensi di atas.
Gunakan bahasa yang sederhana, ramah, dan mudah dipahami oleh petani cabai.
Jika kondisinya adalah "Sehat", berikan apresiasi dan jelaskan bahwa tanaman dalam kondisi sehat (jangan jawab jika hasilnya bukan "sehat").
"""
)
