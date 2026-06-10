import os
cd import uuid
from datetime import datetime
from typing import Any
from pathlib import Path

import cv2
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

from ClassificationModels.CLS_extraction import LabColorExtractor
from ClassificationModels.SVM_Calibrator import SoilCalibratorSVM
from SOCModels.XGBoost_SOC import SOCXGBPredictor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOIL_MODEL_PATH = os.getenv(
    "SOIL_MODEL_PATH", os.path.join(ROOT_DIR, "soil_xgb_model.json")
)
SOIL_CLASSES_PATH = os.getenv(
    "SOIL_CLASSES_PATH", os.path.join(ROOT_DIR, "soil_classes.npy")
)
SOC_MODEL_PATH = os.getenv("SOC_MODEL_PATH", os.path.join(ROOT_DIR, "soc_xgb_model.json"))
SOC_META_PATH = os.getenv("SOC_META_PATH", os.path.join(ROOT_DIR, "soc_xgb_meta.json"))
CONF_THRESHOLD = float(os.getenv("SOIL_CONF_THRESHOLD", "0.75"))
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "analysis_history")

# Image storage configuration
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(ROOT_DIR, "saved_images"))
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(title="Soil Organic Carbon API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

calibrator = SoilCalibratorSVM()
extractor = LabColorExtractor()
soil_model = xgb.XGBClassifier()
soil_classes: np.ndarray | None = None
soc_predictor: SOCXGBPredictor | None = None
supabase_client = None


def _load_soil_models() -> bool:
    global soil_classes
    if not os.path.exists(SOIL_MODEL_PATH) or not os.path.exists(SOIL_CLASSES_PATH):
        return False

    soil_model.load_model(SOIL_MODEL_PATH)
    soil_classes = np.load(SOIL_CLASSES_PATH, allow_pickle=True)
    return True


def _load_soc_model() -> bool:
    global soc_predictor
    if not os.path.exists(SOC_MODEL_PATH):
        return False

    try:
        soc_predictor = SOCXGBPredictor(
            model_path=SOC_MODEL_PATH, meta_path=SOC_META_PATH
        )
    except Exception:
        soc_predictor = None
        return False

    return True


def _init_supabase() -> None:
    global supabase_client
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        supabase_client = None
        return

    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def _decode_image(data: bytes) -> np.ndarray:
    buffer = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return img


def _soc_payload(img: np.ndarray) -> dict[str, Any] | None:
    if soc_predictor is None:
        return None

    soc_value = float(soc_predictor.predict_image(img))
    soc_percent = soc_value / 10.0
    soc_gkg = soc_value * 10.0

    if soc_percent < 1.0:
        soc_cat = "Low"
        soc_explain = "Low organic carbon; fertility and structure may be limited."
    elif soc_percent <= 2.5:
        soc_cat = "Medium"
        soc_explain = "Moderate organic carbon; typical for many cultivated soils."
    else:
        soc_cat = "High"
        soc_explain = "High organic carbon; generally good structure and fertility."

    return {
        "value": soc_value,
        "percent": soc_percent,
        "g_per_kg": soc_gkg,
        "category": soc_cat,
        "note": soc_explain,
    }


def _log_history(payload: dict[str, Any], mode: str) -> None:
    if supabase_client is None:
        return

    soc = payload.get("soc") or {}
    record = {
        "mode": mode,
        "predicted_class": payload.get("predicted_class"),
        "confidence": payload.get("confidence"),
        "margin": payload.get("margin"),
        "is_soil": payload.get("is_soil"),
        "soc_value": soc.get("value"),
        "soc_percent": soc.get("percent"),
        "soc_g_per_kg": soc.get("g_per_kg"),
        "soc_category": soc.get("category"),
        "soc_note": soc.get("note"),
    }

    try:
        supabase_client.table(SUPABASE_TABLE).insert(record).execute()
    except Exception:
        pass


def _classify(img: np.ndarray) -> dict[str, Any]:
    if soil_classes is None:
        raise HTTPException(status_code=503, detail="Soil model not loaded.")

    resized = cv2.resize(img, (128, 128))
    calibrated = calibrator.calibrate(resized)
    
    # Save original and augmented images
    image_paths = _save_image_pair(resized, calibrated)
    
    features = extractor.extract_features(calibrated)
    if features is None:
        raise HTTPException(status_code=500, detail="Feature extraction failed.")

    probs = soil_model.predict_proba(features.reshape(1, -1))[0]
    top_idx = np.argsort(probs)[::-1][:3]
    top_classes = soil_classes[top_idx]
    top_probs = probs[top_idx]

    pred = str(top_classes[0])
    conf = float(top_probs[0])
    margin = float(top_probs[0] - top_probs[1]) if len(top_probs) > 1 else conf

    is_soil = pred != "Not_Soil" and conf >= CONF_THRESHOLD

    return {
        "predicted_class": pred,
        "confidence": conf,
        "margin": margin,
        "is_soil": is_soil,
        "top_classes": [str(c) for c in top_classes.tolist()],
        "top_probs": [float(p) for p in top_probs.tolist()],
        "image_id": image_paths.get("image_id", ""),
        "original_path": image_paths.get("original_path", ""),
        "augmented_path": image_paths.get("augmented_path", ""),
    }


@app.on_event("startup")
def startup() -> None:
    _load_soil_models()
    _load_soc_model()
    _init_supabase()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "soil_model_loaded": soil_classes is not None,
        "soc_model_loaded": soc_predictor is not None,
        "soil_model_path": SOIL_MODEL_PATH,
        "soc_model_path": SOC_MODEL_PATH,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _decode_image(data)
    payload = _classify(img)
    payload["soc"] = _soc_payload(img) if payload["is_soil"] else None
    _log_history(payload, "combined")
    return payload


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _decode_image(data)
    payload = _classify(img)
    _log_history(payload, "classification")
    return payload


@app.post("/soc")
async def soc(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _decode_image(data)
    payload = _classify(img)
    payload["soc"] = _soc_payload(img) if payload["is_soil"] else None
    _log_history(payload, "soc")
    return payload


@app.get("/image/{image_id}/{image_type}")
async def get_image(image_id: str, image_type: str = "original") -> dict[str, Any]:
    """Retrieve saved original or augmented image."""
    try:
        # Search for the image directory
        for dir_path in Path(IMAGES_DIR).iterdir():
            if image_id in dir_path.name:
                image_file = dir_path / f"{image_type}.jpg"
                if image_file.exists():
                    with open(image_file, "rb") as f:
                        return {"image": f.read(), "found": True}
        
        raise HTTPException(status_code=404, detail="Image not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _save_image_pair(original_img: np.ndarray, augmented_img: np.ndarray) -> dict[str, str]:
    """Save original and augmented images to disk."""
    try:
        # Create unique ID and timestamp
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create subdirectory for this analysis
        analysis_dir = os.path.join(IMAGES_DIR, f"{timestamp}_{image_id}")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save original image
        original_path = os.path.join(analysis_dir, "original.jpg")
        cv2.imwrite(original_path, original_img)
        
        # Save augmented image
        augmented_path = os.path.join(analysis_dir, "augmented.jpg")
        cv2.imwrite(augmented_path, augmented_img)
        
        return {
            "image_id": image_id,
            "original_path": original_path,
            "augmented_path": augmented_path,
            "analysis_dir": analysis_dir,
        }
    except Exception as e:
        print(f"Error saving images: {e}")
        return {}
