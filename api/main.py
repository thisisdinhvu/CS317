from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import pandas as pd
import joblib
import time
import logging
from sklearn.metrics import accuracy_score, f1_score
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import os
import json

# Đảm bảo thư mục logging tồn tại
os.makedirs("../logging", exist_ok=True)

# =======================
# Logging Configuration
# =======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../logging/api.log"),   # File log
        logging.StreamHandler()                   # stdout/stderr
    ]
)
# print(logging.getLogger().handlers)  # Xem các handler hiện tại

# =======================
# Prometheus Custom Metrics
# =======================
INFERENCE_TIME = Gauge("model_inference_time_seconds", "Time spent on model inference")
F1_SCORE = Gauge("model_f1_score", "F1 score of prediction")
ACCURACY = Gauge("model_accuracy", "Accuracy of prediction")
CONFIDENCE_SCORE = Gauge("model_mean_confidence_score", "Mean confidence score of prediction")


# =======================
# FastAPI App Init
# =======================
app = FastAPI(title="Diabetes Classification API with Monitoring")

# Prometheus Metrics Instrumentation
Instrumentator().instrument(app).expose(app)

# =======================
# Model and Features
# =======================
MODEL_PATH = "saved_models/KNeighborsClassifier_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
TARGET_COL = "Diabetes_binary"

@app.post("/alert")
async def receive_alert(request: Request):
    payload = await request.json()
    logging.info("Received alert:\n%s", json.dumps(payload, indent=2))
    return {"status": "received"}

# =======================
# Evaluation Endpoint
# =======================
@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    try:
        # Read uploaded CSV
        df = pd.read_csv(file.file)

        # Features and target
        X = df[FEATURE_COLS].values
        y_test = df[TARGET_COL]

        # Inference
        start_time = time.time()
        y_pred = model.predict(X)
        latency = time.time() - start_time

        # Evaluation
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Confidence score if available
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]  # Xác suất lớp 1
            mean_conf = proba.mean()
            CONFIDENCE_SCORE.set(mean_conf)
        else:
            mean_conf = None

        # Prometheus metrics
        INFERENCE_TIME.set(latency)
        ACCURACY.set(acc)
        F1_SCORE.set(f1)

        # Logging
        logging.info(f"{file.filename} evaluated | acc={acc:.4f}, f1={f1:.4f}, latency={latency:.3f}s, confidence={mean_conf}")

        # API response
        return {
            "test-file-name": file.filename,
            "accuracy": acc,
            "f1-score": f1,
            "latency": latency,
            "mean-confidence-score": mean_conf if mean_conf is not None else "N/A",
            "n-samples": len(df),
            "number-of-features": len(FEATURE_COLS)
        }

    except Exception as e:
        logging.error(f"Error evaluating {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error during evaluation")
