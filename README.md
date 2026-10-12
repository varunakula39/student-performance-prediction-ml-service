# Student Performance Prediction - ML Microservice

FastAPI microservice providing machine learning predictions for student marks and academic performance recommendations.

## Tech Stack
- Python 3.10+
- FastAPI & Uvicorn
- Scikit-Learn
- Pandas & NumPy
- Joblib

## Endpoints
- `GET /health` - Health check & model status
- `POST /predict` - Accepts student metrics (attendance, study hours, internal marks, previous exam marks) and returns predicted score, category, and actionable recommendations.

## Local Run
```bash
pip install -r requirements.txt
python main.py
```
Default port: `8001`
