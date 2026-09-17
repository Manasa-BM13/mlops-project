from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "MLOps Iris API is Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        return {"error": str(e)}