from pathlib import Path
import random
from typing import List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from utils.gradcam import generate_gradcam_overlay
from utils.face_detect_blazeface import detect_largest_face_bbox


_MOBILENET = None
_TEMP_SCALE = None  # Temperature scaling parameter for confidence calibration
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Decision threshold (optimized for balanced performance)
_THRESHOLD = 0.5


def _load_temperature_scale():
    """Load calibrated temperature scaling parameter if available."""
    global _TEMP_SCALE
    if _TEMP_SCALE is not None:
        return _TEMP_SCALE

    # Check for temperature calibration file
    models_dir = Path(__file__).resolve().parents[1] / "models"
    temp_file = models_dir / "temperature_scale.txt"

    if temp_file.exists():
        try:
            with open(temp_file, 'r') as f:
                _TEMP_SCALE = float(f.read().strip())
                print(f"[calibration] Using temperature scale: {_TEMP_SCALE}")
        except (ValueError, IOError) as e:
            print(f"[calibration] Failed to load temperature scale: {e}")
            _TEMP_SCALE = 1.0  # Default to no scaling
    else:
        _TEMP_SCALE = 1.0  # Default to no scaling
        print(f"[calibration] No temperature file found at {temp_file}, using T=1.0")

    return _TEMP_SCALE


def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
    """Return a new state dict with `prefix` removed from keys that start with it."""
    new = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_k = k[len(prefix):]
            new[new_k] = v
        else:
            new[k] = v
    return new


def _load_state_dict_robust(model: nn.Module, weights_path: str | None):
    """
    Robust loader:
     - Accepts files that are full models or dicts.
     - Accepts keys with 'module.' (DataParallel), 'backbone.' prefix, or wrapped under 'state_dict' / 'model_state'.
     - Tries to load with various adjustments and prints missing/unexpected keys summary.
    """
    if not weights_path:
        return

    wp = Path(weights_path)
    if not wp.exists():
        print(f"[model loader] Weights path doesn't exist: {weights_path}")
        return

    print(f"[model loader] Loading weights from: {weights_path}")
    # load raw object
    loaded = torch.load(str(wp), map_location="cpu")

    # If the file is a nn.Module instance (saved with torch.save(model))
    if not isinstance(loaded, dict):
        # try to extract state_dict
        try:
            state = loaded.state_dict()
            print("[model loader] Loaded a full model object; extracted state_dict()")
        except Exception:
            print("[model loader] Saved file is not a dict nor a module with state_dict(). Skipping load.")
            return
    else:
        state = loaded

    # Unwrap common keys used by some checkpoints
    # e.g. {'state_dict': {...}}, {'model_state': {...}}, {'module_state_dict': {...}}
    candidate_keys = ['state_dict', 'model_state', 'model', 'state']
    for k in candidate_keys:
        if k in state and isinstance(state[k], dict):
            print(f"[model loader] Found nested state under key '{k}', using that dict")
            state = state[k]
            break

    # If keys look like they have a top-level prefix, try to handle it.
    keys = list(state.keys())
    if len(keys) == 0:
        print("[model loader] State dict is empty. Skipping.")
        return

    def try_load(sd: dict):
        missing, unexpected = model.load_state_dict(sd, strict=False)
        return missing, unexpected

    # 1) Try direct load first
    missing, unexpected = try_load(state)
    if len(missing) == 0 and len(unexpected) == 0:
        print("[model loader] Loaded weights directly with no missing/unexpected keys.")
        return

    # 2) If keys have 'module.' prefix (from DataParallel), strip and retry
    if any(k.startswith('module.') for k in keys):
        stripped = _strip_prefix_from_state_dict(state, 'module.')
        missing, unexpected = try_load(stripped)
        if len(missing) == 0 and len(unexpected) == 0:
            print("[model loader] Stripped 'module.' prefix and loaded successfully.")
            return
        else:
            print("[model loader] Stripped 'module.' but still some mismatch.")

    # 3) If keys have 'backbone.' prefix (common when whole model wrapped), strip and retry
    if any(k.startswith('backbone.') for k in keys):
        stripped = _strip_prefix_from_state_dict(state, 'backbone.')
        missing, unexpected = try_load(stripped)
        if len(missing) == 0 and len(unexpected) == 0:
            print("[model loader] Stripped 'backbone.' prefix and loaded successfully.")
            return
        else:
            print("[model loader] Stripped 'backbone.' but still some mismatch.")

    # 4) Try removing 'backbone.' and also 'module.' if both exist
    strip_both = state
    if any(k.startswith('backbone.') for k in keys):
        strip_both = _strip_prefix_from_state_dict(strip_both, 'backbone.')
    if any(k.startswith('module.') for k in list(strip_both.keys())):
        strip_both = _strip_prefix_from_state_dict(strip_both, 'module.')
    missing, unexpected = try_load(strip_both)
    if len(missing) == 0 and len(unexpected) == 0:
        print("[model loader] Stripped 'backbone.' and/or 'module.' and loaded successfully.")
        return

    # 5) As final attempt, try stripping a small set of common wrapper prefixes like 'backbone.' or 'net.' generically
    for prefix in ('backbone.', 'net.', 'network.', 'model.'):
        if any(k.startswith(prefix) for k in keys):
            stripped = _strip_prefix_from_state_dict(state, prefix)
            missing, unexpected = try_load(stripped)
            if len(missing) == 0 and len(unexpected) == 0:
                print(f"[model loader] Stripped '{prefix}' and loaded successfully.")
                return

    # Report summary so user can inspect
    print("[model loader] Could not perfectly match keys. Summary:")
    print(f"  - total checkpoint keys: {len(keys)}")
    print(f"  - missing keys after best load attempt (count): {len(missing)}")
    if len(missing) > 0:
        print("    example missing keys:", missing[:10])
    print(f"  - unexpected keys after best load attempt (count): {len(unexpected)}")
    if len(unexpected) > 0:
        print("    example unexpected keys:", unexpected[:10])
    print("[model loader] Loaded with strict=False so available matching params were applied. "
          "You may want to inspect the missing/unexpected keys above.")


class SimpleDeepFakeModel(nn.Module):
    """Simple DeepFake detection model (89.6% accuracy)."""

    def __init__(self, num_classes=1, dropout_rate=0.3):
        super(SimpleDeepFakeModel, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Simple attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Linear(128, 1280),
            nn.Sigmoid()
        )

        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(1280, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone.features(x)

        # Apply attention
        att_weights = self.attention(features)
        features = features * att_weights.unsqueeze(-1).unsqueeze(-1)

        # Global pooling and classification
        x = nn.functional.adaptive_avg_pool2d(features, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class DeepFakeModel(nn.Module):
    """DeepFake detection model matching the retrained architecture."""

    def __init__(self, num_classes=1, dropout_rate=0.4):
        super(DeepFakeModel, self).__init__()

        # Load pretrained MobileNetV2
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 1280),
            nn.Sigmoid()
        )

        # Enhanced classifier with regularization
        in_features = 1280
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.7),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Backbone features
        features = self.backbone.features(x)

        # Apply attention
        att_weights = self.attention(features)
        features = features * att_weights.unsqueeze(-1).unsqueeze(-1)

        # Global pooling and classification
        x = nn.functional.adaptive_avg_pool2d(features, 1)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _get_mobilenet(weights_path: str | None):
    global _MOBILENET
    if _MOBILENET is not None:
        return _MOBILENET
    
    # Import the AdvancedDeepFakeModel
    from advanced_anti_overfitting_training import AdvancedDeepFakeModel
    
    # Determine which model architecture to use based on the weights file
    if weights_path and "advanced_deepfake_detector" in weights_path:
        # Use AdvancedDeepFakeModel for the new 99.8% accuracy model
        model = AdvancedDeepFakeModel(num_classes=1, dropout_rate=0.5, attention_dim=256)
        print("[model] Using AdvancedDeepFakeModel architecture (99.8% accuracy - PRIMARY MODEL)")
    elif weights_path and "simple_deepfake_detector" in weights_path:
        # Use SimpleDeepFakeModel for the 89.6% accuracy model
        model = SimpleDeepFakeModel(num_classes=1, dropout_rate=0.3)
        print("[model] Using SimpleDeepFakeModel architecture (89.6% accuracy)")
    else:
        # Use DeepFakeModel for older models
        model = DeepFakeModel(num_classes=1, dropout_rate=0.4)
        print("[model] Using DeepFakeModel architecture (legacy)")
    
    # load weights robustly (handles state_dict wrappers/prefixes)
    if weights_path:
        try:
            _load_state_dict_robust(model, weights_path)
        except Exception as e:
            print(f"[model loader] Exception while loading weights: {e}")
    model.eval()
    _MOBILENET = model
    return _MOBILENET


def _predict_image_pil(img: Image.Image, weights_path: str | None) -> float:
    model = _get_mobilenet(weights_path)
    temperature = _load_temperature_scale()

    with torch.no_grad():
        tensor = _TRANSFORM(img).unsqueeze(0)
        logits = model(tensor)
        # Apply temperature scaling and sigmoid
        # Handle both single value and tensor outputs
        if logits.dim() > 1:
            logits = logits.squeeze()
        scaled_logits = logits / temperature
        prob = torch.sigmoid(scaled_logits).item()

    return float(prob)


def get_image_logits(img: Image.Image, weights_path: str | None) -> float:
    """Get raw logits before temperature scaling (for calibration training)."""
    model = _get_mobilenet(weights_path)
    with torch.no_grad():
        tensor = _TRANSFORM(img).unsqueeze(0)
        logits = model(tensor)
        # Handle both single value and tensor outputs
        if logits.dim() > 1:
            logits = logits.squeeze()
    return float(logits.item())


def _sample_video_frames(path: str, max_frames: int = 16) -> List[Image.Image]:
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        # fallback: try reading sequentially
        frames = []
        for _ in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        return frames
    indices = np.linspace(0, max(0, total - 1), num=max_frames, dtype=int)
    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _detect_and_crop_face(image_pil: Image.Image) -> Image.Image:
    """Crop the largest detected face using OpenCV Haar cascades, fallback to original image."""
    try:
        rgb = np.array(image_pil)
        project_root = Path(__file__).resolve().parents[1]
        bbox = detect_largest_face_bbox(rgb, project_root)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
            face_rgb = Image.fromarray(rgb[y1:y2, x1:x2])
            print(f"[blazeface] face bbox used: ({x1},{y1})-({x2},{y2})")
            return face_rgb
        # Fallback to Haar if BlazeFace not available
        cv_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces: List[Tuple[int, int, int, int]] = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return image_pil
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x2, y2 = x + w, y + h
        x, y = max(0, x), max(0, y)
        face_rgb = Image.fromarray(cv2.cvtColor(cv_img[y:y2, x:x2], cv2.COLOR_BGR2RGB))
        print(f"[haar] face bbox used: ({x},{y})-({x2},{y2})")
        return face_rgb
    except Exception:
        return image_pil


def analyze_image_or_video(input_path: str, results_dir: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)

    p = Path(input_path)
    ext = p.suffix.lower()

    # Inference - Use ONLY the best model (simple_deepfake_detector.pt.best)
    models_dir = Path(__file__).resolve().parents[1] / "models"
    main_model_path = models_dir / "advanced_deepfake_detector_best.pt"

    if main_model_path.exists():
        weights_path = str(main_model_path)
        print("[model] MAIN DeepFake detector (99.8% accuracy - ADVANCED MODEL)")
    else:
        print("ERROR: Main model 'advanced_deepfake_detector_best.pt' not found!")
        print("Please ensure the main model file exists in the models/ directory.")
        return {"label": "unknown", "confidence": 0.0, "explanation_path": None, "model_version": "none"}
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        frames = _sample_video_frames(input_path, max_frames=16)
        if frames:
            probs = []
            for f in frames:
                face = _detect_and_crop_face(f)
                probs.append(_predict_image_pil(face, weights_path))
            # Median aggregation for stability
            prob = float(np.median(np.array(probs)))
        else:
            prob = 0.5
    else:
        # Image path
        img = Image.open(input_path).convert("RGB")
        face = _detect_and_crop_face(img)
        prob = _predict_image_pil(face, weights_path)

    # Convert probability to label/confidence
    # For the advanced model, use standard probability threshold
    if "advanced_deepfake_detector" in weights_path:
        # Advanced model uses standard probability threshold
        label = "deepfake" if prob >= _THRESHOLD else "real"
        confidence = round(prob * 100.0 if label == "deepfake" else (1.0 - prob) * 100.0, 2)
    elif "simple_deepfake_detector" in weights_path:
        # Get raw logits for better decision making
        if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            # For videos, use the same probability as before
            raw_logit = torch.log(torch.tensor(prob / (1 - prob + 1e-8)))
        else:
            # For images, get raw logits
            img = Image.open(input_path).convert("RGB")
            face = _detect_and_crop_face(img)
            raw_logit = get_image_logits(face, weights_path)
        
        # Use logit threshold: higher logits = more likely fake
        # Threshold of -4.0 works well based on our analysis
            label = "deepfake" if raw_logit > -3.90 else "real"
            confidence = round(min(99.9, max(0.1, abs(raw_logit + 3.90) * 20)), 1)
    else:
        # Original logic for other models
        label = "deepfake" if prob >= _THRESHOLD else "real"
        confidence = round(prob * 100.0 if label == "deepfake" else (1.0 - prob) * 100.0, 2)

    overlay_source = input_path

    # If video, extract first frame as PNG for visualization
    if ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            first_frame_path = results_dir / (p.stem + "_frame0.png")
            cv2.imwrite(str(first_frame_path), frame)
            overlay_source = str(first_frame_path)
        else:
            overlay_source = None

    # Produce Grad-CAM overlay placeholder (image only)
    explanation_path: str | None = None
    output_overlay = results_dir / (p.stem + "_gradcam.png")
    try:
        if overlay_source is not None:
            generate_gradcam_overlay(overlay_source, str(output_overlay))
            explanation_path = str(output_overlay)
        else:
            explanation_path = None
    except Exception:
        explanation_path = None

    return {
        "label": label,
        "confidence": confidence,
        "explanation_path": explanation_path,
        "model_version": "advanced-mobilenetv2-attention-v2-99.8%",
    }
