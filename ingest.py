"""Bangun ulang basis pengetahuan ChiliCare di ChromaDB.

Pipeline ini idempoten: setiap kali dijalankan, koleksi lama diganti dengan
hasil yang dibangun ulang dari daftar ``SOURCES``. Jadi menjalankannya
berkali-kali tidak akan menggandakan chunk.

Embedding dihitung lebih dulu, baru koleksi lama dihapus. Dengan urutan ini,
kegagalan API (mis. kuota Gemini habis) tidak akan mengosongkan basis
pengetahuan yang sedang dipakai aplikasi.

Sumber yang diproses:
  * tiga halaman web agronomi,
  * tiga PDF budidaya/penyakit cabai,
  * basis pengetahuan terkurasi ``data/data.json``.

Setiap chunk diberi metadata ``label`` (Bercak, Keriting, Kuning, Sehat,
Whitefly, atau Umum) agar pencarian per-penyakit benar-benar menjangkau
seluruh isi PDF, bukan hanya entri JSON.
"""

import hashlib
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import chromadb

from src.config import CHROMA_DB_DIR, GOOGLE_API_KEY
from src.ingestion.chunker import split_documents
from src.ingestion.labeler import GENERAL_LABEL, assign_labels
from src.ingestion.loader import load_data
from src.retrieval.vector_store import COLLECTION_NAME, get_embedder, get_vector_store

SOURCES = [
    "https://www.dgwfertilizer.co.id/8-hama-dan-penyakit-penting-pada-tanaman-cabai/",
    "https://mitrabertani.com/artikel/detail/Budidaya-Cabai-Sederhana-tapi-Penting-Cara-Tepat-Tanam-Cabai",
    "https://digitani.ipb.ac.id/bagaimana-langkah-langkah-budidaya-cabai/",
    "data/cabai.pdf",
    "data/budidaya_cabai_rawit.pdf",
    "data/penyakit.pdf",
    "data/data.json",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MIN_CHUNK_CHARS = 80
# Gemini free tier menghitung kuota per teks (bukan per HTTP request), dengan
# batas 100 teks/menit. Batch 32 + jeda 25 detik menjaga laju di ~77 teks/menit.
EMBED_BATCH_SIZE = 32
EMBED_BATCH_DELAY = 25
MAX_RETRIES = 5

_RETRY_DELAY_PATTERN = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)

# Penanda boilerplate yang tidak relevan secara agronomi: menu situs, halaman
# daftar isi/gambar PDF, dan blok komentar.
NOISE_MARKERS = (
    "menu toggle",
    "skip to content",
    "berita terkait",
    "leave a reply",
    "add a comment",
    "daftar isi",
    "daftar gambar",
    "daftar tabel",
    "kata pengantar",
)


def _looks_like_noise(text: str) -> bool:
    """Deteksi chunk yang isinya navigasi situs atau daftar isi, bukan materi."""
    lowered = text.lower()
    if any(marker in lowered for marker in NOISE_MARKERS):
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 5:
        short_lines = sum(1 for line in lines if len(line.split()) <= 4)
        if short_lines / len(lines) > 0.6:
            return True

    return False


def _reset_database() -> None:
    """Hapus koleksi lama agar hasil ingest selalu bersih dan deterministik."""
    client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    existing = [getattr(collection, "name", str(collection)) for collection in client.list_collections()]

    for name in existing:
        client.delete_collection(name)
        print(f"Menghapus koleksi lama: {name}")

    # delete_collection() tidak selalu menghapus direktori segmen HNSW lama,
    # sehingga sisa file bisa menumpuk dan ikut ter-commit. Karena seluruh
    # koleksi sudah dihapus, semua subdirektori di sini sudah tidak terpakai.
    for entry in Path(CHROMA_DB_DIR).iterdir():
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
            print(f"Membersihkan berkas segmen usang: {entry.name}")


def _collect_chunks():
    """Muat semua sumber, potong menjadi chunk, lalu beri label."""
    all_chunks = []

    for source in SOURCES:
        try:
            print(f"\nMembaca sumber: {source}")
            raw_docs = load_data(source)
            chunks = split_documents(raw_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            all_chunks.extend(chunks)
            print(f"Berhasil memproses menjadi {len(chunks)} chunk.\n")
        except Exception as error:  # noqa: BLE001
            # Satu sumber gagal (mis. situs down) tidak boleh menggagalkan
            # keseluruhan pipeline.
            print(f"[DILEWATI] Gagal memproses {source}: {error}\n")

    return assign_labels(all_chunks)


def _deduplicate(chunks):
    """Buang chunk terlalu pendek dan chunk yang isinya identik."""
    seen = set()
    unique = []

    for doc in chunks:
        raw_text = doc.page_content or ""
        text = " ".join(raw_text.split())
        if len(text) < MIN_CHUNK_CHARS or _looks_like_noise(raw_text):
            continue

        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue

        seen.add(digest)
        doc.page_content = text
        doc.metadata["chunk_id"] = digest[:24]
        unique.append(doc)

    return unique


def _retry_delay_seconds(error: Exception, attempt: int) -> float:
    """Hormati saran ``retryDelay`` dari API, jika ada; jika tidak, backoff."""
    match = _RETRY_DELAY_PATTERN.search(str(error))
    if match:
        return float(match.group(1)) + 2
    return min(60.0, 10.0 * (2 ** (attempt - 1)))


def _embed_in_batches(texts):
    """Hitung embedding seluruh chunk SEBELUM database disentuh.

    Memisahkan tahap jaringan dari tahap destruktif membuat kegagalan API
    (mis. kuota habis) tidak pernah meninggalkan basis pengetahuan kosong.
    """
    embedder = get_embedder()
    vectors = []
    total = len(texts)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                vectors.extend(embedder.embed_documents(batch))
                print(f"  Embedding {min(start + EMBED_BATCH_SIZE, total)}/{total} chunk")
                break
            except Exception as error:  # noqa: BLE001
                if attempt == MAX_RETRIES:
                    raise SystemExit(
                        f"Gagal menghitung embedding setelah {MAX_RETRIES} percobaan: {error}"
                    ) from error
                delay = _retry_delay_seconds(error, attempt)
                print(f"  Percobaan {attempt}/{MAX_RETRIES} gagal. Ulangi dalam {delay:.0f}s...")
                time.sleep(delay)

        if start + EMBED_BATCH_SIZE < total:
            time.sleep(EMBED_BATCH_DELAY)

    return vectors


def _clean_metadata(metadata):
    """Chroma menolak nilai metadata None."""
    return {key: value for key, value in metadata.items() if value is not None}


def _store_embeddings(vectors, chunks) -> int:
    """Simpan chunk beserta embedding yang sudah dihitung ke koleksi baru."""
    store = get_vector_store()
    collection = store._collection  # noqa: SLF001 - butuh jalur embedding precomputed

    ids = [
        f"{str(doc.metadata.get('label', GENERAL_LABEL)).lower()}-{doc.metadata['chunk_id']}"
        for doc in chunks
    ]

    for start in range(0, len(chunks), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        collection.add(
            ids=ids[start:end],
            embeddings=vectors[start:end],
            documents=[doc.page_content for doc in chunks[start:end]],
            metadatas=[_clean_metadata(doc.metadata) for doc in chunks[start:end]],
        )

    return collection.count()


def run_ingestion_pipeline() -> None:
    if not GOOGLE_API_KEY:
        raise SystemExit(
            "GOOGLE_API_KEY (atau GEMINI_API_KEY) belum diisi. "
            "Embedding ChromaDB membutuhkan kredensial Google AI Studio."
        )

    print("Memulai Data Ingestion Pipeline (rebuild penuh)\n")

    chunks = _deduplicate(_collect_chunks())

    if not chunks:
        raise SystemExit("Tidak ada chunk yang berhasil diproses. Basis pengetahuan tidak diubah.")

    distribution = Counter(doc.metadata.get("label", GENERAL_LABEL) for doc in chunks)
    print("Komposisi chunk per label:")
    for label, count in distribution.most_common():
        print(f"  - {label:<10} {count}")

    print(f"\nMenghitung embedding untuk {len(chunks)} chunk...")
    vectors = _embed_in_batches([doc.page_content for doc in chunks])

    if len(vectors) != len(chunks):
        raise SystemExit(
            f"Jumlah embedding ({len(vectors)}) tidak sama dengan jumlah chunk ({len(chunks)}). "
            "Basis pengetahuan lama tidak diubah."
        )

    # Baru setelah semua embedding siap, koleksi lama dihapus dan diganti.
    _reset_database()
    stored = _store_embeddings(vectors, chunks)

    print(f"\nSelesai. Koleksi '{COLLECTION_NAME}' sekarang berisi {stored} embedding.")


if __name__ == "__main__":
    run_ingestion_pipeline()
