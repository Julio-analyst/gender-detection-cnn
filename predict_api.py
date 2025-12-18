from fastapi import FastAPI, File, UploadFile
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import uvicorn
import tempfile

app = FastAPI()

MODEL_PATH = "models/lstm_production.h5"
SR = 16000
N_MFCC = 13
N_FFT = 2048
HOP = 512

def extract_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=N_MFCC, 
        n_fft=N_FFT, 
        hop_length=HOP
    )
    return mfcc.T

@app.on_event("startup")
def load_dl_model():
    global model
    model = load_model(MODEL_PATH)

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    audio, sr = librosa.load(tmp_path, sr=SR, mono=True)
    mfcc = extract_mfcc(audio, sr)
    mfcc_padded = pad_sequences([mfcc], dtype='float32', padding='post')
    pred = model.predict(mfcc_padded, verbose=0)[0][0]
    label = "Perempuan" if pred > 0.5 else "Laki-laki"
    confidence = float(pred) if pred > 0.5 else float(1 - pred)
    return {"label": label, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
