from fastapi import FastAPI, HTTPException
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score


app = FastAPI(title="Diabetes Classification API")

MODEL_PATH = "saved_models/KNeighborsClassifier_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURE_COLS = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]
TARGET_COL = "Diabetes_binary"

@app.post("/evaluate")
async def evaluate(file: UploadFile = File(...)):
    # data = await file.read()

    df = pd.read_csv(file.file)

    X = df[FEATURE_COLS].values
    y_test = df[TARGET_COL]

    y_pred = model.predict(X)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "test-file-name" : file.filename,
        "accuracy" : acc,
        "f1-score" : f1,
        'n-samples': len(df),
        'number of features' : len(FEATURE_COLS)

    }