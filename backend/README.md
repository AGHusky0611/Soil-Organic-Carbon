# Soil Organic Carbon API

## Run locally

1. Create and activate a Python environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Start the server from the repo root:

```
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` returns model load status.
- `POST /analyze` combined classification + SOC.
- `POST /classify` classification only.
- `POST /soc` SOC mode (still requires soil check).

## Environment variables

- `SOIL_MODEL_PATH` (default: repo root `soil_xgb_model.json`)
- `SOIL_CLASSES_PATH` (default: repo root `soil_classes.npy`)
- `SOC_MODEL_PATH` (default: repo root `soc_xgb_model.json`)
- `SOC_META_PATH` (default: repo root `soc_xgb_meta.json`)
- `SOIL_CONF_THRESHOLD` (default: `0.75`)
