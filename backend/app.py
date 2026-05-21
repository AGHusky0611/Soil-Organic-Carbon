import os
from typing import Any

import cv2
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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


def _classify(img: np.ndarray) -> dict[str, Any]:
    if soil_classes is None:
        raise HTTPException(status_code=503, detail="Soil model not loaded.")

    resized = cv2.resize(img, (128, 128))
    calibrated = calibrator.calibrate(resized)
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
    }


@app.on_event("startup")
def startup() -> None:
    _load_soil_models()
    _load_soc_model()


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
    return payload


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _decode_image(data)
    return _classify(img)


@app.post("/soc")
async def soc(file: UploadFile = File(...)) -> dict[str, Any]:
    data = await file.read()
    img = _decode_image(data)
    payload = _classify(img)
    payload["soc"] = _soc_payload(img) if payload["is_soil"] else None
    return payload
