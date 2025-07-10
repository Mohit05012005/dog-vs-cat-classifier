import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "model.h5"
FILE_ID = "1kIDT8hr8N62gGveHhA8ch_GuxVZT-Nww"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?id=1kIDT8hr8N62gGveHhA8ch_GuxVZT-Nww"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
