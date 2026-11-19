
# 🌶️ ChiliCare AI: Sistem Cerdas Diagnosis Penyakit Daun Cabai (YOLOv11s & LLM RAG)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?logo=python)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/Vision-YOLOv11s%20(mAP50%3A%200.903)-00FFFF.svg?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3-orange.svg)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-purple.svg)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LPU%20Accelerated-f55036.svg)](https://groq.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://docker.com)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-green.svg?logo=github-actions)](.github/workflows/ci-cd.yml)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces%20Live-yellow.svg?logo=huggingface)](https://huggingface.co/spaces/feryms/chilicareAI)

> **ChiliCare AI** adalah aplikasi web cerdas berarsitektur *dual-pipeline* yang dirancang untuk membantu petani, penyuluh, dan akademisi agronomi dalam mendiagnosis penyakit tanaman cabai secara presisi. Sistem ini menggabungkan model *Computer Vision* **YOLOv11s** untuk lokalisasi lesi visual dan ekosistem *Large Language Models* (LLM) berbasis **Retrieval-Augmented Generation (RAG)** untuk memberikan rekomendasi penanganan dan dosis bahan aktif yang terstruktur.

---

## 🎯 Latar Belakang & Nilai Nyata di Lapangan

Tanaman cabai (*Capsicum annuum*) merupakan komoditas hortikultura strategis di Indonesia dengan fluktuasi harga yang tinggi. Salah satu ancaman terbesar kegagalan panen adalah serangan hama dan infeksi patogen daun (seperti *Virus Gemini*, *Cercospora*, dan *Thrips*).

**Tantangan di Lapangan:**
1. **Diagnosis Terlambat atau Keliru:** Gejala awal bercak jamur dan serangan hama pengisap sering disalahartikan, berujung pada penggunaan pestisida yang salah sasaran.
2. **Keterbatasan Akses Ahli:** Tenaga penyuluh pertanian lapangan memiliki keterbatasan waktu dan jangkauan wilayah.
3. **Resistensi Kimia:** Penyemprotan bahan aktif tanpa takaran tepat memicu resistensi hama dan pemborosan biaya produksi.

**Solusi ChiliCare AI:**
Menghadirkan asisten agronomi saku yang dapat diakses dari peramban ponsel. Cukup unggah foto daun bergejala untuk mendapatkan identifikasi *bounding box* visual dalam hitungan milidetik, diikuti narasi penanganan kuratif dan preventif dari basis data agronomi terpercaya.

---

## 🏗️ Arsitektur Sistem

Sistem beroperasi melalui dua pipeline terpisah yang terintegrasi di backend FastAPI:

```mermaid
flowchart TD
    subgraph INGESTION["1. Knowledge Ingestion Pipeline (Offline)"]
        DOCS["Dokumen Agronomi: Budidaya & Penyakit Cabai (PDF/JSON)"] --> Loader["Multi-Format Document Loader"]
        Loader --> Splitter["Recursive Character Text Splitter"]
        Splitter --> Embed["Sentence Transformers Embedding"]
        Embed --> Chroma[("ChromaDB Vector Store")]
    end

    subgraph VISION["2. Vision Inference Pipeline (Real-Time)"]
        UserImg(["Foto Daun Cabai (Ponsel/Kamera)"]) --> Preprocess["Normalisasi & Resize Citra"]
        Preprocess --> YOLO["YOLOv11s Custom Object Detection"]
        YOLO --> BBox["Anotasi Bounding Box & Skor Keyakinan"]
        YOLO --> LabelOutput["Ekstraksi Label Kelas (Bercak/Kuning/Keriting/Kutu)"]
    end

    subgraph RAG["3. Context Retrieval & LLM Generation"]
        LabelOutput --> QueryBuild["Formulasi Query Agronomi"]
        QueryBuild --> VectorSearch["Similarity Search (ChromaDB, k=3)"]
        Chroma --> VectorSearch
        VectorSearch --> Context["Konteks Terverifikasi (Penyebab, Dosis, Sanitasi)"]
        Context --> PromptTemplate["Prompt Khusus Patologi Tanaman"]
        PromptTemplate --> LLM["Groq LPU: GPT-OSS 20B"]
        LLM --> FinalDiag(["Laporan Diagnosis Terstruktur & Rekomendasi Tindakan"])
    end

    subgraph CHAT["4. Interactive Expert Chatbot"]
        UserQuestion(["Tanya Jawab Bebas (Pupuk/Perawatan)"]) --> ChatChain["Conversational RAG Chain (Multi-turn Memory)"]
        Chroma --> ChatChain
        ChatChain --> StreamResp(["Streaming Response Asisten Ahli"])
    end
```

---

## 📊 Hasil Evaluasi Model (YOLOv11s)

Model dievaluasi menggunakan data uji independen (*test split*) dengan augmentasi pengujian (*Test-Time Augmentation*).

### Spesifikasi Model
- **Arsitektur:** YOLO11s (`100 layers`, `9,414,735 parameters`, `21.4 GFLOPs`)
- **Perangkat Evaluasi:** Tesla T4 GPU (16 GB VRAM)
- **Kecepatan Inferensi:** ~31.7 ms per citra (GPU) / ~180 ms (CPU)
- **Ukuran File Bobot:** ~19.2 MB (`best.pt`)

### 1. Performa Keseluruhan (Overall Test Metrics)

| Dataset Split | Gambar | Objek (Instances) | Precision (Box-P) | Recall (R) | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation** | 60 | 110 | 0.861 | 0.789 | 0.869 | 0.713 |
| **Test (Final)** | **60** | **111** | **0.896** | **0.882** | **0.903** | **0.758** |

### 2. Performa Deteksi per Kelas (Test Set)

| Kelas | Gambar | Objek (Instances) | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🟤 **Bercak Daun (Cercospora)** | 11 | 11 | **0.963** | 0.909 | **0.905** | 0.797 |
| 🍃 **Daun Keriting (Thrips / Mites)** | 11 | 36 | 0.785 | 0.889 | **0.896** | 0.673 |
| 🟨 **Penyakit Kuning (Virus Gemini)** | 14 | 16 | 0.898 | **1.000** | **0.995** | **0.825** |
| 🌿 **Daun Sehat (Normal)** | 13 | 29 | **1.000** | 0.716 | **0.867** | 0.773 |
| 🪰 **Kutu Kebul (Whitefly)** | 12 | 19 | 0.832 | 0.895 | **0.850** | 0.724 |
| **Rata-rata (All Classes)** | **60** | **111** | **0.896** | **0.882** | **0.903** | **0.758** |

> **Analisis Hasil:** Model menunjukkan kemampuan istimewa pada deteksi **Penyakit Kuning** dengan **Recall 100%** dan **mAP50 99.5%**, memastikan infeksi virus gemini tidak terlewatkan di lahan. Deteksi **Bercak Daun** dan **Daun Sehat** masing-masing mencapai presisi **96.3%** dan **100%**, meminimalkan risiko alarm palsu.

---

## 🚀 Fitur & Keunggulan Teknis

1. **Dual-Pipeline Vision-to-Text & Text-to-Text**:
   - Inferensi visual cepat dengan bounding box anotasi.
   - Penjelasan penyakit, penyebab biologis, dan langkah pemulihan otomatis via RAG.
2. **Chatbot Ahli Agronomi Terpadu**:
   - Asisten percakapan interaktif multi-turn untuk konsultasi dosis pupuk NPK, jarak tanam, dan sanitasi gulma.
   - Didukung streaming token berkecepatan tinggi via Groq LPU.
3. **Antarmuka Web Modern & Responsif**:
   - Dilengkapi **Landing Page** informatif bertema agrikultur dan **Web App** yang nyaman digunakan di layar desktop maupun ponsel.
   - Penuh aksesibilitas keyboard dan navigasi tanpa hambatan.
4. **Basis Pengetahuan Terverifikasi**:
   - Seluruh jawaban grounded pada modul budidaya cabai resmi untuk mencegah halusinasi jawaban AI.
5. **Siap Produksi & Teruji CI/CD**:
   - Pengujian otomatis 21 unit test pytest.
   - Linting ruff standar industri.
   - Pipeline GitHub Actions otomatis untuk publikasi Docker image dan deploy ke Hugging Face Space.

---

## 📂 Struktur Proyek

```text
chilicareAI/
├── backend/
│   └── api.py                    # Aplikasi FastAPI, routing inferensi YOLO & streaming chat
├── frontend/
│   ├── index.html                # Halaman beranda (Landing page) modern
│   ├── app.html                  # Antarmuka aplikasi web deteksi penyakit & chat AI
│   ├── script.js                 # Logika streaming chat, upload & markdown parsing
│   ├── background.jpg            # Aset fotografi cabai di kebun
│   └── _redirects                # Konfigurasi routing Netlify CDN
├── src/
│   ├── config.py                 # Konfigurasi variabel lingkungan & path terpusat
│   ├── chains/
│   │   ├── chain.py              # Orkestrasi RAG chain LangChain
│   │   ├── prompt.py             # Prompt engineering agronomi & patologi
│   │   └── rag.py                # Pipeline sintesis diagnosis visi ke teks
│   ├── ingestion/
│   │   ├── loader.py             # Multi-format document loader
│   │   ├── chunker.py            # Pemotongan teks dokumen
│   │   └── embedder.py           # Inisialisasi model embedding
│   └── retrieval/
│       ├── vector_store.py       # Abstraksi database vektor ChromaDB
│       ├── retriever.py          # Konfigurasi retriever dokumen
│       └── reranker.py           # Reranker cross-encoder pendukung
├── data/
│   ├── data.json                 # Basis pengetahuan terstruktur gejala & penanganan
│   └── *.pdf                     # Modul referensi teknis budidaya cabai
├── model/
│   └── best(v11s).pt             # Bobot model YOLOv11s hasil pelatihan
├── tests/
│   ├── test_api.py               # Unit test REST endpoints FastAPI
│   ├── test_chunker.py           # Pengujian pemotongan teks
│   ├── test_config.py            # Validasi konfigurasi & environment
│   ├── test_embedder.py          # Pengujian model embedding
│   ├── test_prompt.py            # Verifikasi prompt template
│   └── test_reranker.py          # Pengujian fungsionalitas reranker
├── .github/workflows/
│   └── ci-cd.yml                 # CI/CD otomatis (Lint, Test, Docker, Deploy HF)
├── app.py                        # Antarmuka alternatif berbasis Streamlit
├── Dockerfile                    # Konfigurasi kontainerisasi produksi
├── pytest.ini                    # Konfigurasi pytest & pytest-asyncio
└── requirements.txt              # Daftar dependensi produksi Python
```

---

## 🛠️ Panduan Memulai Cepat

### 1. Kloning Repositori & Persiapan Lingkungan

```bash
# Kloning repositori
git clone https://github.com/FeriMauliandi/Chilicare-AI_Chili-Leaf-Disease-Detection.git
cd Chilicare-AI_Chili-Leaf-Disease-Detection

# Buat virtual environment Python
python -m venv .venv

# Aktivasi environment (Windows PowerShell)
.venv\Scripts\activate

# Aktivasi environment (Linux/macOS)
# source .venv/bin/activate

# Pasang dependensi
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Konfigurasi Variabel Lingkungan (`.env`)

Buat berkas `.env` di direktori utama:

```env
# API Key untuk LLM (Pilih salah satu atau keduanya)
GROQ_API_KEY=gsk_...
OPENROUTER_API_KEY=sk-or-...

# Opsional: Konfigurasi nama model LLM
LLM_MODEL=openai/gpt-oss-20b
```

### 3. Eksekusi Indeksasi Vektor (Opsional jika memperbarui data)

```bash
python ingest.py
```

### 4. Menjalankan Aplikasi

#### Opsi A: Menjalankan Server FastAPI (Direkomendasikan)
```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```
Akses di peramban Anda:
- **Landing Page:** `http://localhost:8000/`
- **Aplikasi Deteksi & Chat:** `http://localhost:8000/app`
- **Dokumentasi API Swagger:** `http://localhost:8000/docs`

#### Opsi B: Menggunakan Docker
```bash
docker build -t chilicare-app .
docker run -p 7860:7860 --env-file .env chilicare-app
```
Akses aplikasi di `http://localhost:7860`.

---

## 🧪 Pengujian & CI/CD Pipeline

Proyek ini menerapkan pengujian otomatis terstandar untuk menjamin stabilitas kode:

```bash
# Menjalankan seluruh pengujian unit
pytest -v

# Menjalankan pengujian dengan laporan cakupan kode (coverage)
pytest --cov=src --cov=backend tests/

# Menjalankan linter kode
ruff check src/ backend/ tests/
```

### Alur Kerja GitHub Actions (`.github/workflows/ci-cd.yml`)
Setiap kali terjadi `push` atau `pull request` ke branch `main`:
1. **Lint & Test:** Memverifikasi kualitas kode dengan `ruff` dan mengeksekusi 21 unit test dengan `pytest`.
2. **Docker Build:** Menguji build image Docker agar bebas dari kendala dependensi.
3. **Auto-Deploy:** Otomatis mempublikasikan image ke GitHub Container Registry (GHCR) dan memperbarui aplikasi di **Hugging Face Space**.

---

## 📄 Lisensi & Pernyataan Penelitian

Proyek ini dikembangkan untuk tujuan penelitian dan alat bantu agronomi terbuka. Seluruh saran pengobatan penyakit dan rekomendasi pestisida bersumber dari pustaka penyuluhan budidaya cabai. Selalu perhatikan instruksi label resmi produsen bahan aktif saat melakukan penanganan di lahan pertanian.
