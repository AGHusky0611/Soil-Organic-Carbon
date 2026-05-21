# Render Deploy Guide

## 1) Create a Render service

- Go to https://render.com and create a new Web Service.
- Choose "Docker" as the runtime.

## 2) Connect the repo

- Push this repo to GitHub.
- Select the repo and root directory in Render.

## 3) Build and start

Render will use the Dockerfile automatically.
- Build command: leave blank
- Start command: leave blank (uses Dockerfile CMD)

## 4) Environment variables (optional)

Only needed if model files are stored elsewhere:

- SOIL_MODEL_PATH
- SOIL_CLASSES_PATH
- SOC_MODEL_PATH
- SOC_META_PATH
- SOIL_CONF_THRESHOLD

If you keep the model files in the repo root, no env vars are needed.

## 5) Deploy

- Click Create Web Service.
- When it finishes, copy the public URL.

## 6) Update the mobile app

Use the Render URL as the API base, for example:

- https://your-service.onrender.com

## Notes

- The Docker image includes OpenCV runtime libs (libgl1, libglib2.0-0).
- Make sure the model files are in the repo root when you deploy.
