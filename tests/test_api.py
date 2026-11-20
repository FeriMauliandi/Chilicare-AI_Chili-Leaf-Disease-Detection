from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import io
from PIL import Image

from backend.api import app

client = TestClient(app)

def test_read_root():
    """Test root endpoint / returns 200 and expected message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ChiliCare API" in response.json()["message"]

@patch("backend.api.model")
def test_detect_disease_invalid_file_type(mock_model):
    """Test /detect rejects non-image file uploads with 400 Bad Request."""
    files = {"file": ("test.txt", b"dummy content", "text/plain")}
    response = client.post("/detect", files=files)
    assert response.status_code == 400
    assert "gambar" in response.json()["detail"]

@patch("backend.api.model")
@patch("backend.api.generate_narrative")
def test_detect_disease_success(mock_generate_narrative, mock_model):
    """Test /detect endpoint with a valid image file and mocked YOLO model."""
    mock_generate_narrative.return_value = "Saran penanganan penyakit kuning."
    
    # Mock YOLO model box output
    mock_box = MagicMock()
    mock_box.cls = [0]
    mock_box.conf = [0.95]
    
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    # plot() returns a dummy numpy array representing RGB image
    import numpy as np
    mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    
    mock_model.return_value = [mock_result]
    mock_model.names = {0: "kuning"}
    
    # Create a small dummy image in bytes
    img = Image.new("RGB", (50, 50), color="green")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()
    
    files = {"file": ("chili_leaf.jpg", img_bytes, "image/jpeg")}
    response = client.post("/detect", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_detections"] == 1
    assert data["results"][0]["class"] == "Kuning"
    assert data["results"][0]["narrative"] == "Saran penanganan penyakit kuning."

@patch("backend.api.create_rag_chain")
def test_ask_expert_endpoint(mock_create_rag_chain):
    """Test /ask endpoint streaming response from RAG chain."""
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Tanaman ", "cabai ", "membutuhkan ", "air."]
    mock_create_rag_chain.return_value = mock_chain
    
    payload = {"question": "Bagaimana penyiraman cabai?"}
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 200
    assert "Tanaman cabai membutuhkan air." in response.text

@patch("backend.api.create_rag_chain")
def test_ask_expert_with_history(mock_create_rag_chain):
    """Test /ask endpoint handles multi-turn conversation history."""
    mock_chain = MagicMock()
    mock_chain.stream.return_value = ["Gunakan ", "fungisida ", "mankozeb."]
    mock_create_rag_chain.return_value = mock_chain

    payload = {
        "question": "Berapa dosisnya?",
        "history": [
            {"role": "user", "content": "Daun cabai ada bercak hitam"},
            {"role": "assistant", "content": "Itu adalah bercak daun cercospora"}
        ]
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    assert "Gunakan fungisida mankozeb." in response.text
    
    call_args = mock_chain.stream.call_args[0][0]
    assert call_args["input"] == "Berapa dosisnya?"
    assert len(call_args["chat_history"]) == 2

def test_get_rag_chain_singleton():
    """Test get_rag_chain caches and returns singleton instance."""
    import backend.api
    from src.chains.chain import create_rag_chain as orig_create_rag_chain

    backend.api._rag_chain_instance = None
    call_count = 0
    dummy_chain = object()

    def fake_create_chain():
        nonlocal call_count
        call_count += 1
        return dummy_chain

    backend.api.create_rag_chain = fake_create_chain
    try:
        res1 = backend.api.get_rag_chain()
        res2 = backend.api.get_rag_chain()
        assert res1 is dummy_chain
        assert res2 is dummy_chain
        assert call_count == 1
    finally:
        backend.api.create_rag_chain = orig_create_rag_chain
        backend.api._rag_chain_instance = None

def _cors_preflight(origin, path="/ask", method="POST"):
    return client.options(
        path,
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "content-type",
        },
    )


def test_cors_allows_local_origin():
    """Origin lokal yang terdaftar boleh melakukan preflight."""
    response = _cors_preflight("http://localhost:8000")
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8000"


def test_cors_allows_huggingface_space_origin():
    """Origin HF Space (deployment utama) diizinkan."""
    origin = "https://feryms-chilicareai.hf.space"
    response = _cors_preflight(origin)
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_rejects_unknown_origin():
    """Origin asing tidak boleh mendapat header Access-Control-Allow-Origin."""
    response = _cors_preflight("https://evil.example.com")
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_cors_credentials_not_enabled_by_default():
    """allow_credentials harus mati kecuali diaktifkan eksplisit lewat env."""
    response = _cors_preflight("http://localhost:8000")
    assert "access-control-allow-credentials" not in response.headers


@patch("backend.api.model")
def test_rate_limit_exceeded(mock_model):
    """Test rate limiting triggers HTTP 429 when threshold is exceeded."""
    from backend.api import limiter
    limiter.reset()

    try:
        files = {"file": ("test.txt", b"dummy", "text/plain")}
        responses = [client.post("/detect", files=files) for _ in range(17)]
        status_codes = [r.status_code for r in responses]
        assert 429 in status_codes
        
        rate_limited = [r for r in responses if r.status_code == 429][0]
        data = rate_limited.json()
        assert rate_limited.status_code == 429
        assert data["status"] == "error"
        assert "Terlalu banyak permintaan" in data["detail"]
    finally:
        limiter.reset()

