"""DEPRECATED: gunakan ``python ingest.py``.

Dulu skrip ini hanya memasukkan ``data/data.json`` ke ChromaDB. Sekarang
seluruh sumber (PDF, web, dan JSON) diproses oleh pipeline di ``ingest.py``,
sehingga menjalankan skrip ini secara terpisah berisiko menggandakan data.
Skrip ini dipertahankan sebagai alias agar perintah lama tetap bekerja.
"""

from ingest import run_ingestion_pipeline

if __name__ == "__main__":
    print("db_setup.py sudah digabung ke ingest.py. Menjalankan rebuild penuh...\n")
    run_ingestion_pipeline()
