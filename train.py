"""
Student Performance Prediction - Machine Learning Model Training
Trains a scikit-learn Regression model using:
- Attendance percentage
- Daily study hours
- Internal marks
- Previous exam marks
Saves the trained pipeline to 'student_model.joblib'
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def generate_student_dataset(n_samples=1500, random_state=42):
    """Generate realistic student performance data."""
    np.random.seed(random_state)
    
    # Feature 1: Attendance percentage (40% to 100%)
    attendance = np.random.normal(loc=78, scale=12, size=n_samples)
    attendance = np.clip(attendance, 40.0, 100.0)
    
    # Feature 2: Daily study hours (0.5 to 10 hours)
    study_hours = np.random.normal(loc=4.0, scale=1.8, size=n_samples)
    study_hours = np.clip(study_hours, 0.5, 10.0)
    
    # Feature 3: Internal marks (25 to 100)
    internal_marks = 0.25 * attendance + 2.0 * study_hours + np.random.normal(loc=45, scale=10, size=n_samples)
    internal_marks = np.clip(internal_marks, 25.0, 100.0)
    
    # Feature 4: Previous exam marks (25 to 100)
    previous_exam_marks = 0.20 * attendance + 2.5 * study_hours + 0.3 * internal_marks + np.random.normal(loc=30, scale=10, size=n_samples)
    previous_exam_marks = np.clip(previous_exam_marks, 25.0, 100.0)
    
    # Target: Final predicted exam marks (0 to 100)
    # Realistic academic weighted relationship
    noise = np.random.normal(loc=0, scale=2.5, size=n_samples)
    final_marks = (
        0.15 * attendance +
        2.20 * study_hours +
        0.35 * internal_marks +
        0.40 * previous_exam_marks +
        noise
    )
    final_marks = np.clip(np.round(final_marks, 1), 10.0, 100.0)
    
    df = pd.DataFrame({
        'attendance': np.round(attendance, 1),
        'study_hours': np.round(study_hours, 1),
        'internal_marks': np.round(internal_marks, 1),
        'previous_exam_marks': np.round(previous_exam_marks, 1),
        'final_marks': final_marks
    })
    
    return df

def train_and_save_model():
    print("==================================================")
    print("  Student Performance Model Training Starting...  ")
    print("==================================================")
    
    df = generate_student_dataset()
    data_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(data_dir, "student_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated dataset with {len(df)} records saved to {csv_path}")
    
    features = ['attendance', 'study_hours', 'internal_marks', 'previous_exam_marks']
    X = df[features]
    y = df['final_marks']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=10,
        min_samples_split=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("--------------------------------------------------")
    print(f"Model Evaluation Metrics:")
    print(f"  R-Squared (R2) Score : {r2:.4f}")
    print(f"  Mean Absolute Error  : {mae:.2f}")
    print(f"  Mean Squared Error   : {mse:.2f}")
    print("--------------------------------------------------")
    
    model_artifact = {
        'model': model,
        'features': features,
        'metrics': {
            'r2': float(r2),
            'mae': float(mae),
            'mse': float(mse)
        }
    }
    
    model_path = os.path.join(data_dir, "student_model.joblib")
    joblib.dump(model_artifact, model_path)
    print(f"Trained model saved successfully to {model_path}")
    print("==================================================")

if __name__ == "__main__":
    train_and_save_model()
