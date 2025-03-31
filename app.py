# app.py
# start command: uvicorn app:app --reload

from fastapi import FastAPI, File, UploadFile
from fastai.vision.all import *
from fastapi.middleware.cors import CORSMiddleware  # 👈 add this
from io import BytesIO
from PIL import Image
import urllib.request
import os

# 🔽 Download model if not present
MODEL_URL = "https://www.dropbox.com/scl/fi/q1151resw8zt4ko9wlcnk/model.pkl?rlkey=sx9dupg7kh1su43rmld4herkh&st=exfe3aam&dl=1"
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")

learn = load_learner("model.pkl")
app = FastAPI()

# 👇 Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify ["http://localhost:5500"] for stricter security
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