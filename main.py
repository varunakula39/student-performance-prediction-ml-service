"""
Student Performance Prediction - FastAPI Microservice
Runs on Port 8001
Provides REST APIs for:
- GET /health
- POST /predict
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Student Performance Prediction ML API",
    description="Microservice providing machine learning predictions for student exam marks.",
    version="1.0.0"
)

# CORS configuration to allow calls from Spring Boot and React local servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "student_model.joblib")
model_artifact = None

def load_or_train_model():
    """Load the joblib model, or train on-the-fly if not present."""
    global model_artifact
    if os.path.exists(MODEL_PATH):
        try:
            model_artifact = joblib.load(MODEL_PATH)
            print(f"[ML Service] Loaded pre-trained model from {MODEL_PATH}")
            return
        except Exception as e:
            print(f"[ML Service] Error loading model file: {e}")
            
    print("[ML Service] Pre-trained model not found or corrupted. Training fresh model...")
    try:
        from train import train_and_save_model
        train_and_save_model()
        if os.path.exists(MODEL_PATH):
            model_artifact = joblib.load(MODEL_PATH)
            print("[ML Service] Successfully trained and loaded model.")
    except Exception as e:
        print(f"[ML Service] Fallback notice: Model training error: {e}. Will use calibrated regression formula.")

# Load model on startup
@app.on_event("startup")
def startup_event():
    load_or_train_model()

class PredictionRequest(BaseModel):
    attendance: float = Field(..., ge=0.0, le=100.0, description="Attendance percentage (0-100)")
    study_hours: float = Field(..., ge=0.0, le=24.0, description="Daily study hours (0-24)")
    internal_marks: float = Field(..., ge=0.0, le=100.0, description="Internal assessment marks (0-100)")
    previous_exam_marks: float = Field(..., ge=0.0, le=100.0, description="Previous semester/exam marks (0-100)")
    student_name: Optional[str] = Field(None, description="Optional student name")
    student_id: Optional[int] = Field(None, description="Optional student ID")

class PredictionResponse(BaseModel):
    predicted_marks: float
    performance_category: str
    recommendations: List[str]
    input_summary: dict

def categorize_performance(marks: float) -> str:
    if marks >= 80.0:
        return "Excellent"
    elif marks >= 65.0:
        return "Good"
    elif marks >= 50.0:
        return "Average"
    else:
        return "Needs Improvement"

def generate_recommendations(attendance: float, study_hours: float, internal_marks: float, prev_marks: float, predicted: float) -> List[str]:
    recs = []
    
    # Study hours evaluation
    if study_hours < 2.5:
        recs.append("Increase daily dedicated study time to at least 3 to 4 hours.")
    elif study_hours < 4.5:
        recs.append("Maintain consistent daily revision sessions and practice problem solving.")
    else:
        recs.append("Great study schedule! Maintain your consistent study routine.")

    # Attendance evaluation
    if attendance < 75.0:
        recs.append(f"Attendance is currently {attendance:.1f}% (below the 75% threshold). Attend regular lectures and lab sessions to prevent exam debarment.")
    elif attendance < 85.0:
        recs.append("Good attendance. Try to target 90%+ for comprehensive syllabus coverage.")
    else:
        recs.append("Excellent attendance record. Continue participating actively in classroom discussions.")

    # Internal marks evaluation
    if internal_marks < 50.0:
        recs.append("Internal marks need urgent attention. Consult professors for clarification on weak topics and submit assignments on time.")
    elif internal_marks < 75.0:
        recs.append("Focus on scoring higher in upcoming mid-term quizzes and unit assessments.")
    else:
        recs.append("Solid performance in internal assessments. Keep up the high standard.")

    # Previous marks & predicted score trend
    if predicted >= 80.0:
        recs.append("Predicted for honors/distinction. Take on advanced projects or competitive exams.")
    elif predicted >= 65.0:
        recs.append("On track for a solid First Class grade. Review past question papers to reach distinction.")
    elif predicted >= 50.0:
        recs.append("Prioritize core subjects and form peer study groups to push into the 'Good' grade bracket.")
    else:
        recs.append("Immediate academic counseling and remedial tutorial classes are strongly recommended.")

    return recs

@app.get("/health")
def health_check():
    return {
        "status": "UP",
        "service": "Student Performance ML Service",
        "model_loaded": model_artifact is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_performance(req: PredictionRequest):
    attendance = float(req.attendance)
    study_hours = float(req.study_hours)
    internal_marks = float(req.internal_marks)
    prev_marks = float(req.previous_exam_marks)
    
    predicted_score = None
    
    # Use ML model if loaded
    if model_artifact and 'model' in model_artifact:
        try:
            model = model_artifact['model']
            features = pd.DataFrame([{
                'attendance': attendance,
                'study_hours': study_hours,
                'internal_marks': internal_marks,
                'previous_exam_marks': prev_marks
            }])
            raw_pred = float(model.predict(features)[0])
            predicted_score = raw_pred
        except Exception as e:
            print(f"[ML Service] Inference error: {e}. Using calibrated regression fallback.")
    
    # Calibrated fallback if model wasn't loaded or encountered an error
    if predicted_score is None:
        predicted_score = (
            0.15 * attendance +
            2.20 * min(study_hours, 10.0) +
            0.35 * internal_marks +
            0.40 * prev_marks
        )
    
    # Strictly constrain between 0.0 and 100.0
    predicted_score = max(0.0, min(100.0, round(predicted_score, 1)))
    category = categorize_performance(predicted_score)
    recommendations = generate_recommendations(attendance, study_hours, internal_marks, prev_marks, predicted_score)
    
    return PredictionResponse(
        predicted_marks=predicted_score,
        performance_category=category,
        recommendations=recommendations,
        input_summary={
            "student_name": req.student_name,
            "student_id": req.student_id,
            "attendance": attendance,
            "study_hours": study_hours,
            "internal_marks": internal_marks,
            "previous_exam_marks": prev_marks
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)

