from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import sys
import shutil
import mimetypes
import time

# Make imports robust regardless of working directory
THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]  # .../DeepSecure
REPO_DIR = PROJECT_DIR.parent       # .../Development
for p in (str(PROJECT_DIR), str(REPO_DIR)):
    if p not in sys.path:
        sys.path.append(p)

# Support running as `backend.main` or `DeepSecure.backend.main`
try:
    from DeepSecure.detectors.cnn_lstm_detector import analyze_image_or_video
    from DeepSecure.detectors.mfcc_transformer_detector import analyze_audio
    from DeepSecure.utils.geotag import get_geo_tag
except Exception:  # noqa: BLE001 - fallback for flexible run modes
    from detectors.cnn_lstm_detector import analyze_image_or_video
    from detectors.mfcc_transformer_detector import analyze_audio
    from utils.geotag import get_geo_tag


app = FastAPI(title="DeepSecure MVP", version="0.1.0")

# Allow local Streamlit to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = Path(__file__).resolve().parents[1] / "static" / "uploads"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "static" / "results"
for d in (UPLOAD_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _detect_media_type(filename: str, content_type: str | None) -> str:
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type.startswith("video/"):
            return "video"
        if content_type.startswith("audio/"):
            return "audio"
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        if guessed.startswith("image/"):
            return "image"
        if guessed.startswith("video/"):
            return "video"
        if guessed.startswith("audio/"):
            return "audio"
    return "unknown"


@app.get("/")
def root():
    return {"status": "ok", "service": "DeepSecure MVP"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    media_type = _detect_media_type(file.filename, file.content_type)
    if media_type == "unknown":
        return JSONResponse(status_code=400, content={"error": "Unsupported file type"})

    # Save upload
    safe_name = f"{int(time.time()*1000)}_{file.filename}"
    upload_path = UPLOAD_DIR / safe_name
    with upload_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Route to detector
    if media_type in ("image", "video"):
        result = analyze_image_or_video(str(upload_path), RESULTS_DIR)
    elif media_type == "audio":
        result = analyze_audio(str(upload_path), RESULTS_DIR)
    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported media"})

    # Geo-tag (IP-based placeholder)
    geo_tag = get_geo_tag()

    payload = {
        "media_type": media_type,
        "result": result.get("label", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "explanation_path": result.get("explanation_path"),
        "geo_tag": geo_tag,
        "timestamp": timestamp,
        "model_version": result.get("model_version", "v0"),
    }

    return JSONResponse(content=payload)


