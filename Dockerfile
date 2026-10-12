FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files and pre-trained model
COPY main.py .
COPY train.py .
COPY student_dataset.csv .
COPY student_model.joblib .

EXPOSE 8001

CMD ["python", "main.py"]
