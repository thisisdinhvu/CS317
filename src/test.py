# test.py
import mlflow.sklearn
from sklearn.metrics import classification_report
from data_preprocessing import load_and_preprocess_data

def test_model(model_uri):
    _, _, X_test, _, _, y_test, _ = load_and_preprocess_data()
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)
    print("Test Results:")
    print(classification_report(y_test, y_pred))