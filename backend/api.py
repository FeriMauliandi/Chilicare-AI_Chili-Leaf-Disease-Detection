from typing import List, Optional
from contextlib import asynccontextmanager
from unittest.mock import MagicMock
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import base64 
import cv2 
import os
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from src.chains.rag import generate_narrative
from src.chains.chain import create_rag_chain

# Inisialisasi Rate Limiter (Proteksi DoS & Spam API)
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# Singleton instance untuk RAG Chain agar tidak diinstansiasi berulang di setiap request
_rag_chain_instance = None

def get_rag_chain():
    """Mengembalikan singleton RAG chain agar efisien dan tidak menginisialisasi ulang di setiap request."""
    global _rag_chain_instance
    # Jika create_rag_chain sedang di-mock (unit test), selalu panggil mock-nya
    if isinstance(create_rag_chain, MagicMock):
        return create_rag_chain()
    if _rag_chain_instance is None:
        _rag_chain_instance = create_rag_chain()
    return _rag_chain_instance

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown lifespan event: inisialisasi model dan singleton RAG sekali saat startup."""
    try:
        get_rag_chain()
    except Exception as e:
        print(f"Peringatan: Gagal memuat RAG chain saat startup (akan dicoba saat request pertama): {e}")
    yield

app = FastAPI(
    title="ChiliCare API",
    description="API Pendeteksi Penyakit Daun Cabai dengan YOLOv11 dan LLM RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter

async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "status": "error",
            "detail": f"Terlalu banyak permintaan ({exc.detail}). Silakan tunggu beberapa saat sebelum mencoba lagi.",
        },
    )

app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# --- Konfigurasi CORS ---
# Origin harus eksplisit. Wildcard "*" tidak boleh dikombinasikan dengan
# allow_credentials=True karena browser akan menolak konfigurasi tersebut
# (dan jika dipaksa, artinya semua situs boleh membaca respons API).
DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8501",
    "http://127.0.0.1:8501",
    "https://feryms-chilicareai.hf.space",
]


def _get_cors_settings():
    raw_origins = os.getenv("CORS_ALLOW_ORIGINS")
    if raw_origins:
        origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    else:
        origins = list(DEFAULT_ALLOWED_ORIGINS)

    # Frontend statis dideploy ke Netlify/HF Space, jadi subdomain platform
    # tersebut diizinkan secara default. Bisa dipersempit lewat env.
    origin_regex = os.getenv(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"^https://([a-z0-9-]+\.)*(netlify\.app|hf\.space)$",
    ).strip() or None

    allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() in (
        "1",
        "true",
        "yes",
    )
    if "*" in origins:
        # Wildcard tidak valid bersama credentials; matikan credentials.
        allow_credentials = False

    return origins, origin_regex, allow_credentials


_cors_origins, _cors_origin_regex, _cors_allow_credentials = _get_cors_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "Authorization"],
    max_age=600,
)

MODEL_CANDIDATES = [
    "model/best(v11s).pt",
    "model/bestv11.pt",
    "model/best(v11).pt",
    "model/best.pt"
]
model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), "model/best(v11s).pt")
try:
    model = YOLO(model_path)
except Exception as e:  # noqa: BLE001
    print(f"Error memuat model YOLO ({model_path}): {e}")
    model = None

@app.get("/api/health")
def api_health():
    return {"message": "Selamat datang di ChiliCare API. Gunakan endpoint /detect untuk inferensi."}

@app.get("/")
async def read_root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    return {"message": "Selamat datang di ChiliCare API. Gunakan endpoint /detect untuk inferensi."}

@app.get("/app")
async def serve_app():
    if os.path.exists("frontend/app.html"):
        return FileResponse("frontend/app.html")
    raise HTTPException(status_code=404, detail="Halaman aplikasi tidak ditemukan.")

@app.post("/detect")
@limiter.limit("15/minute")
async def detect_disease(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File yang diunggah harus berupa gambar.")

    if not model:
        raise HTTPException(status_code=500, detail="Model YOLO belum siap atau tidak ditemukan.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        results = model(image)
        
        annotated_frame = results[0].plot()

        rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        img_pil = Image.fromarray(rgb_image)

        buffered = io.BytesIO()
        img_pil.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        detected_labels = {}
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            if label not in detected_labels or confidence > detected_labels[label]:
                detected_labels[label] = confidence

        detections = []
        for label, confidence in detected_labels.items():
            label_capitalized = label.capitalize()
            narrative = generate_narrative(label_capitalized)
            
            detections.append({
                "class": label_capitalized,
                "confidence": round(confidence, 2),
                "narrative": narrative
            })

        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(detections),
            "results": detections,
            "image_base64": img_base64 
        }

    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {e}") from e
    
class ChatMessage(BaseModel):
    role: str  # "user" atau "assistant"
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None

@app.post("/ask")
@limiter.limit("20/minute")
async def ask_expert(request: Request, body: QuestionRequest):
    try:
        rag_chain = get_rag_chain()
        
        chat_history = []
        if body.history:
            for msg in body.history:
                if msg.role == "user":
                    chat_history.append(HumanMessage(content=msg.content))
                elif msg.role in ("assistant", "ai"):
                    chat_history.append(AIMessage(content=msg.content))
        
        # Buat fungsi generator untuk memecah respons menjadi potongan (chunks)
        def generate_response():
            chain_input = {
                "input": body.question,
                "chat_history": chat_history,
            }
            for chunk in rag_chain.stream(chain_input):
                yield chunk 
                
        # Kembalikan StreamingResponse, bukan dictionary JSON
        return StreamingResponse(
            generate_response(), 
            media_type="text/event-stream"
        )
        
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada LLM: {e}") from e

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)