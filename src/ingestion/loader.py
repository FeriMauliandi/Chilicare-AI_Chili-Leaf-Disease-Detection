import json
import os

from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document

# Batas waktu permintaan web agar satu situs yang lambat tidak menggagalkan
# seluruh pipeline ingest.
WEB_TIMEOUT_SECONDS = 20


def _format_json_entry(label: str, details: dict) -> str:
    """Ratakan satu entri data.json menjadi teks yang siap di-embed."""
    lines = [f"Penyakit/Kondisi: {label}"]

    for key, heading in (
        ("penyebab", "Penyebab"),
        ("gejala", "Gejala"),
        ("penanganan", "Penanganan"),
        ("pencegahan", "Pencegahan"),
    ):
        values = details.get(key) or []
        if values:
            lines.append(f"{heading}:\n- " + "\n- ".join(values))

    return "\n".join(lines) + "\n"


def _load_json(source: str):
    """Muat basis pengetahuan terkurasi dari data.json."""
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    documents = []
    for label, details in data.items():
        documents.append(
            Document(
                page_content=_format_json_entry(label, details),
                metadata={
                    "label": label,
                    "source": source,
                    "format": "json",
                },
            )
        )
    return documents


def load_data(source: str):
    print(f"Mencoba memuat data dari: {source}")

    if source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source, requests_kwargs={"timeout": WEB_TIMEOUT_SECONDS})
        documents = loader.load()

    elif source.lower().endswith(".pdf"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = PyPDFLoader(source)
        documents = loader.load()

    elif source.lower().endswith(".json"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File tidak ditemukan: {source}")
        documents = _load_json(source)

    elif source.lower().endswith(".md"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = TextLoader(source, encoding="utf-8")
        documents = loader.load()

    elif source.lower().endswith(".csv"):
        if not os.path.exists(source):
            raise FileNotFoundError(f"File tidak ditemukan: {source}")
        loader = CSVLoader(source, encoding="utf-8")
        documents = loader.load()

    else:
        raise ValueError("Format tidak didukung. Harap masukkan URL web, .pdf, .json, .md, atau .csv")

    return documents
