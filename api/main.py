# from fastapi import FastAPI, File, UploadFile, HTTPException, Request
# import pandas as pd
# import joblib
# import time
# import logging
# from sklearn.metrics import accuracy_score, f1_score
# from prometheus_fastapi_instrumentator import Instrumentator
# from prometheus_client import Gauge, Counter
# import os
# import json

# # Đảm bảo thư mục logging tồn tại
# # LOG_DIR = os.environ.get("LOG_DIR", "/logging")
# LOG_DIR = os.environ.get("LOG_DIR", "/app/logs")  # match Promtail config
# os.makedirs(LOG_DIR, exist_ok=True)

# # =======================
# # Logging Configuration
# # =======================
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler(f"{LOG_DIR}/api.log"),   # File log
#         logging.StreamHandler()                   # stdout/stderr
#     ]
# )

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setLevel(logging.INFO)

# stderr_handler = logging.StreamHandler(sys.stderr)
# stderr_handler.setLevel(logging.ERROR)

# logging.getLogger().addHandler(stdout_handler)
# logging.getLogger().addHandler(stderr_handler)
# # print(logging.getLogger().handlers)  # Xem các handler hiện tại

# # =======================
# # Prometheus Custom Metrics
# # =======================
# INFERENCE_TIME = Gauge("model_inference_time_seconds", "Time spent on model inference")
# F1_SCORE = Gauge("model_f1_score", "F1 score of prediction")
# ACCURACY = Gauge("model_accuracy", "Accuracy of prediction")
# CONFIDENCE_SCORE = Gauge("model_mean_confidence_score", "Mean confidence score of prediction")


# # =======================
# # FastAPI App Init
# # =======================
# app = FastAPI(title="Diabetes Classification API with Monitoring")

# # Prometheus Metrics Instrumentation
# Instrumentator().instrument(app).expose(app)

# # =======================
# # Model and Features
# # =======================
# MODEL_PATH = "saved_models/KNeighborsClassifier_model.pkl"
# model = joblib.load(MODEL_PATH)

# FEATURE_COLS = [
#     "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
#     "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
#     "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
#     "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
# ]
# TARGET_COL = "Diabetes_binary"

# ALERT_COUNT = Counter("received_alerts_total", "Number of alerts received")


# @app.post("/alert")
# async def receive_alert(request: Request):
#     try:
#         payload = await request.json()
#     except Exception as e:
#         logging.exception("Failed to parse JSON payload from /alert")
#         raise HTTPException(status_code=400, detail="Invalid JSON payload")

#     ALERT_COUNT.inc()

#     for alert in payload.get("alerts", []):
#         logging.warning(
#             f"ALERT: {alert.get('labels', {}).get('alertname')} | "
#             f"Severity: {alert.get('labels', {}).get('severity')} | "
#             f"Instance: {alert.get('labels', {}).get('instance')} | "
#             f"Description: {alert.get('annotations', {}).get('description')}"
#         )

#     return {"status": "received"}

# # =======================
# # Evaluation Endpoint
# # =======================
# @app.post("/evaluate")
# async def evaluate(file: UploadFile = File(...)):
#     try:
#         # Read uploaded CSV
#         df = pd.read_csv(file.file)

#         # Features and target
#         X = df[FEATURE_COLS].values
#         y_test = df[TARGET_COL]

#         # Inference
#         start_time = time.time()
#         y_pred = model.predict(X)
#         latency = time.time() - start_time

#         # Evaluation
#         acc = accuracy_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)

#         # Confidence score if available
#         if hasattr(model, "predict_proba"):
#             proba = model.predict_proba(X)[:, 1]  # Xác suất lớp 1
#             mean_conf = proba.mean()
#             CONFIDENCE_SCORE.set(mean_conf)
#         else:
#             mean_conf = None

#         # Prometheus metrics
#         INFERENCE_TIME.set(latency)
#         ACCURACY.set(acc)
#         F1_SCORE.set(f1)

#         # Logging
#         logging.info(f"{file.filename} evaluated | acc={acc:.4f}, f1={f1:.4f}, latency={latency:.3f}s, confidence={mean_conf}")

#         # API response
#         return {
#             "test-file-name": file.filename,
#             "accuracy": acc,
#             "f1-score": f1,
#             "latency": latency,
#             "mean-confidence-score": mean_conf if mean_conf is not None else "N/A",
#             "n-samples": len(df),
#             "number-of-features": len(FEATURE_COLS)
#         }

#     except Exception as e:
#         logging.error(f"Error evaluating {file.filename}: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail="Error during evaluation")

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import pandas as pd
import joblib
import time
import logging
from sklearn.metrics import accuracy_score, f1_score
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge, Counter
import os
import sys
from logging.handlers import SysLogHandler

# =======================
# Setup Log Directory
# =======================
LOG_DIR = os.environ.get("LOG_DIR", "/logging") 
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "api.log")

# =======================
# Logging Configuration
# =======================
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File handler
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stdout handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)

# Stderr handler
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.ERROR)
stderr_handler.setFormatter(formatter)
logger.addHandler(stderr_handler)

# Optional: Syslog handler
try:
    syslog_handler = SysLogHandler(address="/dev/log")
    syslog_handler.setLevel(logging.INFO)
    syslog_handler.setFormatter(formatter)
    logger.addHandler(syslog_handler)
except Exception as e:
    logger.warning(f"Could not attach syslog: {e}")

logger.info("=== DIABETES API STARTING ===")
logger.info(f"Log file: {LOG_FILE}")

# =======================
# Prometheus Custom Metrics
# =======================
INFERENCE_TIME = Gauge("model_inference_time_seconds", "Time spent on model inference")
F1_SCORE = Gauge("model_f1_score", "F1 score of prediction")
ACCURACY = Gauge("model_accuracy", "Accuracy of prediction")
CONFIDENCE_SCORE = Gauge("model_mean_confidence_score", "Mean confidence score of prediction")
ALERT_COUNT = Counter("received_alerts_total", "Number of alerts received")

# =======================
# FastAPI App Init
# =======================
app = FastAPI(title="Diabetes Classification API with Monitoring")
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

# =======================
# Alert Endpoint
# =======================
@app.post("/alert")
async def receive_alert(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        logger.exception("Failed to parse JSON payload from /alert")
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    ALERT_COUNT.inc()

    for alert in payload.get("alerts", []):
        logger.warning(
            f"ALERT: {alert.get('labels', {}).get('alertname')} | "
            f"Severity: {alert.get('labels', {}).get('severity')} | "
            f"Instance: {alert.get('labels', {}).get('instance')} | "
            f"Description: {alert.get('annotations', {}).get('description')}"
        )

    return {"status": "received"}

# =======================
# Evaluation Endpoint
# =======================
@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        X = df[FEATURE_COLS].values
        y_test = df[TARGET_COL]

        start_time = time.time()
        y_pred = model.predict(X)
        latency = time.time() - start_time

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            mean_conf = proba.mean()
            CONFIDENCE_SCORE.set(mean_conf)
        else:
            mean_conf = None

        INFERENCE_TIME.set(latency)
        ACCURACY.set(acc)
        F1_SCORE.set(f1)

        logger.info(f"{file.filename} evaluated | acc={acc:.4f}, f1={f1:.4f}, latency={latency:.3f}s, confidence={mean_conf}")

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
        logger.error(f"Error evaluating {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error during evaluation")
