# run_pipeline.py
import mlflow
from train import train_model
from data_preprocessing import load_and_preprocess_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_pipeline():
    train_model()
    client = mlflow.tracking.MlflowClient()
    experiment_id = client.get_experiment_by_name("Default").experiment_id
    run_infos = client.list_run_infos(experiment_id, order_by=["start_time DESC"], max_results=1)
    run_id = run_infos[0].run_id

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    _, _, X_test, _, _, y_test, _ = load_and_preprocess_data()
    y_pred = model.predict(X_test)

    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    print("Test Precision:", precision_score(y_test, y_pred))
    print("Test Recall:", recall_score(y_test, y_pred))
    print("Test F1 Score:", f1_score(y_test, y_pred))

if __name__ == "__main__":
    run_pipeline()