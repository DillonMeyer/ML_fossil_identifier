# app.py
# start command: uvicorn app:app --host 0.0.0.0 --port 10000

from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os

learn = load_learner("model.pkl")
app = FastAPI()

# 👇 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ml-fossil-identifier.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = PILImage.create(BytesIO(img_bytes))
    pred_class, pred_idx, probs = learn.predict(img)
    return {
        "prediction": str(pred_class),
        "confidence": float(probs[pred_idx])
    }