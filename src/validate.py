# validate.py
import mlflow.sklearn
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data

def validate_model(model_uri):
    _, X_val, _, _, y_val, _, _ = load_and_preprocess_data()
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_val)
    print("Validation Results:")
    print(classification_report(y_val, y_pred))
