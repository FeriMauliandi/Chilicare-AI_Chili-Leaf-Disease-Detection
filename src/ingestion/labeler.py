"""Pemberian label otomatis untuk chunk knowledge base.

Model YOLO hanya mengembalikan lima label: Bercak, Keriting, Kuning, Sehat,
dan Whitefly. Entri kurasi di ``data/data.json`` sudah membawa label yang
benar, tetapi dokumen PDF dan halaman web tidak. Tanpa metadata ``label``,
pencarian dengan filter per-penyakit (dipakai saat menyusun narasi hasil
deteksi) tidak akan pernah menyentuh isi PDF.

Modul ini menandai chunk tersebut lewat skoring kata kunci. Chunk yang tidak
bisa dipetakan dengan yakin diberi label ``Umum``: tetap bisa ditemukan pada
chat bebas (yang tidak memfilter label), tetapi tidak mengotori pencarian
per-penyakit.
"""

import re
from typing import Dict, Iterable, List

from langchain_core.documents import Document

GENERAL_LABEL = "Umum"

# Skor minimum agar sebuah chunk dianggap cukup yakin untuk diberi label
# penyakit tertentu. Di bawah nilai ini chunk dianggap konteks umum.
MIN_SCORE = 2

# Bobot lebih tinggi untuk istilah yang spesifik (nama patogen/gejala),
# bobot rendah untuk istilah yang sering muncul sebagai konteks sekunder.
# Urutan label menentukan pemenang saat skor seri (yang lebih spesifik dulu).
LABEL_KEYWORDS: Dict[str, Dict[str, int]] = {
    "Bercak": {
        "bercak": 3,
        "cercospora": 3,
        "bercak daun": 3,
        "antraknosa": 2,
        "xanthomonas": 3,
        "mankozeb": 2,
        "karbendazim": 2,
        "spora": 1,
        "jamur": 1,
    },
    "Keriting": {
        "keriting": 3,
        "mengeriting": 3,
        "thrips": 3,
        "tungau": 3,
        "polyphagotarsonemus": 3,
        "mites": 2,
        "kutu daun": 2,
        "aphis": 2,
        "akarikida": 2,
    },
    "Kuning": {
        "kuning": 3,
        "menguning": 3,
        "virus gemini": 3,
        "begomovirus": 3,
        "klorosis": 2,
        "bule": 2,
    },
    "Whitefly": {
        "whitefly": 3,
        "kutu kebul": 3,
        "bemisia": 3,
        "embun madu": 2,
        "jelaga": 2,
        "yellow sticky": 2,
        "spirotetramat": 1,
    },
    "Sehat": {
        "pemupukan": 2,
        "penyiraman": 2,
        "sanitasi": 2,
        "budidaya": 2,
        "sehat": 2,
        "benih": 1,
        "bibit": 1,
        "pupuk": 1,
        "panen": 1,
        "mulsa": 1,
        "jarak tanam": 1,
        "drainase": 1,
    },
}


def _normalize(text: str) -> str:
    """Huruf kecil dan rapikan spasi agar pencocokan kata kunci konsisten."""
    return re.sub(r"\s+", " ", (text or "").lower())


def score_labels(text: str) -> Dict[str, int]:
    """Hitung skor kemiripan teks terhadap setiap label penyakit."""
    normalized = _normalize(text)
    return {
        label: sum(normalized.count(keyword) * weight for keyword, weight in keywords.items())
        for label, keywords in LABEL_KEYWORDS.items()
    }


def infer_label(text: str) -> str:
    """Tentukan satu label untuk sebuah teks, atau ``Umum`` jika ambigu."""
    scores = score_labels(text)
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score < MIN_SCORE:
        return GENERAL_LABEL
    return best_label


def assign_labels(documents: Iterable[Document]) -> List[Document]:
    """Tambahkan metadata ``label`` ke dokumen yang belum memilikinya.

    Label yang sudah ada (misalnya dari ``data.json``) tidak pernah ditimpa,
    karena entri kurasi lebih akurat daripada hasil heuristik.
    """
    docs = list(documents)
    for doc in docs:
        if doc.metadata.get("label"):
            continue
        doc.metadata["label"] = infer_label(doc.page_content)
    return docs
