from fastapi import FastAPI, UploadFile, File
import torch
from io import BytesIO
from PIL import Image
import numpy as np
import os
import cv2
import tempfile
from pathlib import Path

# Torch Hub  offline

PROJECT_DIR = Path(__file__).parent.resolve()
TORCH_HUB_DIR = PROJECT_DIR / "yolov5_cache"
os.environ['TORCH_HUB_DIR'] = str(TORCH_HUB_DIR)



app = FastAPI(title="Fall Detector YOLOv5 Offline Backend")


MODEL_PATH = PROJECT_DIR / "model" / "best.pt"
model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path=str(MODEL_PATH),
    force_reload=False
)


@app.get("/")
async def root():
    return {"message": "Fall Detector YOLOv5 Offline Backend Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename.lower()

    # ===== if image =====
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img = Image.open(BytesIO(await file.read())).convert("RGB")

    # ===== if video =====
    elif filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        last_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
        cap.release()
        os.remove(tmp_path)

        if last_frame is None:
            return {"error": "Cannot read video frames."}

        # BGR TO RGB
        img = Image.fromarray(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))

    else:
        return {"error": "Unsupported file type."}

    #  inference
    results = model(np.array(img))
    preds = results.xyxy[0]

    if len(preds) > 0:
        pred_class_idx = int(preds[0, -1].item())
    else:
        pred_class_idx = 0

    prediction = model.names[pred_class_idx]
    return {"prediction": prediction}
