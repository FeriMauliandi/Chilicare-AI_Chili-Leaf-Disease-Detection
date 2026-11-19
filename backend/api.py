from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import uvicorn
import base64 
import cv2 
import numpy as np 
import os
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage, AIMessage
from src.chains.rag import generate_narrative
from src.chains.chain import create_rag_chain

app = FastAPI(
    title="ChiliCare API",
    description="API Pendeteksi Penyakit Daun Cabai dengan YOLOv11 dan LLM RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = YOLO("model/bestv11.pt")
except Exception as e:
    print(f"Error memuat model YOLO: {e}")
    model = None

@app.get("/api/health")
def api_health():
    return {"message": "Selamat datang di ChiliCare API. Gunakan endpoint /detect untuk inferensi."}

@app.get("/")
async def read_root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept and os.path.exists("frontend/landing.html"):
        return FileResponse("frontend/landing.html")
    return {"message": "Selamat datang di ChiliCare API. Gunakan endpoint /detect untuk inferensi."}

@app.get("/app")
async def serve_app():
    if os.path.exists("frontend/index.html"):
        return FileResponse("frontend/index.html")
    raise HTTPException(status_code=404, detail="Halaman aplikasi tidak ditemukan.")

@app.post("/detect")
async def detect_disease(file: UploadFile = File(...)):
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

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")
    
class ChatMessage(BaseModel):
    role: str  # "user" atau "assistant"
    content: str

class QuestionRequest(BaseModel):
    question: str
    history: Optional[List[ChatMessage]] = None

@app.post("/ask")
async def ask_expert(request: QuestionRequest):
    try:
        rag_chain = create_rag_chain()
        
        chat_history = []
        if request.history:
            for msg in request.history:
                if msg.role == "user":
                    chat_history.append(HumanMessage(content=msg.content))
                elif msg.role in ("assistant", "ai"):
                    chat_history.append(AIMessage(content=msg.content))
        
        # Buat fungsi generator untuk memecah respons menjadi potongan (chunks)
        def generate_response():
            chain_input = {
                "input": request.question,
                "chat_history": chat_history,
            }
            for chunk in rag_chain.stream(chain_input):
                yield chunk 
                
        # Kembalikan StreamingResponse, bukan dictionary JSON
        return StreamingResponse(
            generate_response(), 
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan pada LLM: {str(e)}")

if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)