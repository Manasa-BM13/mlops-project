from fastapi import FastAPI
import joblib
import numpy as np
import os

app = FastAPI()

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(model_path)

# 0,1,2 ge name map
class_names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.get("/")
def home():
    return {"message": "MLOps Iris API is Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        pred_id = int(prediction[0])
        pred_name = class_names[pred_id]
        return {"prediction_id": pred_id, "prediction_name": pred_name}
    except Exception as e:
        return {"error": str(e)}