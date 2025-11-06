DeepFake Detection System
Advanced AI-Powered Detection for Text, Image, Video & Audio DeepFakes


Overview
DeepFake content is fast-evolving and poses a serious threat to security, misinformation, and digital integrity. The DeepFake Detection System is an AI-based multi-modal detection framework that identifies deepfake content from texts, images, videos, and audios by utilizing cutting-edge pre-trained models.

Key Features:

Detects AI-generated texts, images, videos, and audios

Deploys prior trained models-therefore no manual training becomes necessary

API integration for quick third-party consumption

Web-based and browser extension for real-time detection

Highly optimized for speed, accuracy, and efficiency


Stay ahead of the AI revolution. Protect digital integrity with DeepFake Detection!

# DeepSecure MVP

## Overview

DeepSecure is a comprehensive deepfake detection system that combines components from Mavericks, DeepFakeHunter, and Maya360 into a single upload-based MVP. It detects deepfakes in images, videos, and audio files with explainability features and geo-tagging capabilities.

## Features

- **Multi-modal Detection**: Supports images, videos, and audio files
- **Advanced Models**: 
  - CNN+LSTM/Transformer hybrid for image/video detection
  - MFCC+Transformer for audio detection
- **Explainability**: Grad-CAM overlays for visual explanations
- **Geo-tagging**: IP/metadata-based location tracking
- **RESTful API**: FastAPI backend with `/analyze` endpoint
- **User-friendly UI**: Streamlit-based web interface

## Project Structure

```
DeepSecure/
├── app/                    # Streamlit frontend
│   └── streamlit_app.py
├── backend/                # FastAPI backend
│   └── main.py
├── detectors/              # Detection models
│   ├── cnn_lstm_detector.py
│   ├── mfcc_transformer_detector.py
│   └── xception_wrapper.py
├── utils/                  # Utility functions
│   ├── face_detect_blazeface.py
│   ├── geotag.py
│   └── gradcam.py
├── models/                 # Trained model files
├── static/                 # Upload and result directories
│   ├── uploads/
│   └── results/
├── data/                   # Training data (not included in repo)
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd DeepSecure
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Backend (FastAPI)

1. **Start the FastAPI server**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```
   
   Or from the project root:
   ```bash
   python -m uvicorn backend.main:app --reload --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`

2. **API Documentation**
   - Interactive docs: `http://127.0.0.1:8000/docs`
   - Alternative docs: `http://127.0.0.1:8000/redoc`

### Running the Frontend (Streamlit)

1. **Start the Streamlit app**
   ```bash
   streamlit run app/streamlit_app.py
   ```

   The UI will be available at `http://localhost:8501`

2. **Configure backend URL (optional)**
   - Create `.streamlit/secrets.toml` file
   - Add: `backend_url = "http://127.0.0.1:8000"`

### API Endpoints

#### POST `/analyze`

Analyzes uploaded media files for deepfake detection.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image, video, or audio file)

**Supported formats:**
- Images: `.jpg`, `.jpeg`, `.png`
- Videos: `.mp4`, `.mov`, `.avi`
- Audio: `.wav`, `.mp3`, `.m4a`

**Response:**
```json
{
  "media_type": "image|video|audio",
  "result": "deepfake|real",
  "confidence": 85.5,
  "explanation_path": "path/to/explanation.png",
  "geo_tag": {
    "country": "India",
    "city": "Bengaluru"
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "model_version": "v0"
}
```

#### GET `/`

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "DeepSecure MVP"
}
```

## Components

- **Backend**: FastAPI with a single `/analyze` endpoint
- **Frontend**: Streamlit upload UI
- **Detectors**: 
  - CNN+LSTM/Transformer hybrid for image/video
  - MFCC+Transformer for audio
- **Explainability**: Grad-CAM overlays
- **Geo-tagging**: IP/metadata-based location detection

## Development

### Testing

Run tests to verify the installation:
```bash
python test_advanced_model.py
```

### Model Training

For training new models, refer to:
- `advanced_anti_overfitting_training.py` - Training script
- `model_documentation/` - Model architecture and performance documentation

## Notes

- The `data/` folder is excluded from the repository (see `.gitignore`)
- Model files (`.pt`, `.pth`) are excluded due to size
- Uploaded files are stored in `static/uploads/`
- Analysis results are stored in `static/results/`

