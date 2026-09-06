# MLOps Iris Project - Production Ready

End-to-end MLOps pipeline for Iris classification.

## Tech Stack
- Python, Scikit-Learn, FastAPI, Docker, GitHub Actions

## Project Structure
- train.py: Trains RandomForest and saves model.pkl
- app.py: FastAPI with / and /predict endpoints
- Dockerfile: Containerization
- requirements.txt: Dependencies

## How to Run Locally
pip install -r requirements.txt
python train.py
uvicorn app:app --reload
# Open http://127.0.0.1:8000/docs

## How to Run with Docker
docker build -t mlops-iris.
docker run -p 8000:8000 mlops-iris

## API Example
POST /predict
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
-> {"prediction": 0}