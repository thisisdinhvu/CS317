import os
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from metaflow import FlowSpec, step, Parameter
from src.data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from mlflow.models.signature import infer_signature

class DiabetesClassificationFlow(FlowSpec):

    n_trials = Parameter("n_trials", default=30)

    @step
    def start(self):
        print("Starting the pipeline...")
        mlflow.set_tracking_uri("http://localhost:8080")
        self.next(self.load_and_preprocess)

    @step
    def load_and_preprocess(self):
        print("Loading and preprocessing data...")
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = load_and_preprocess_data()
        self.next(self.train_models)

    @step
    def train_models(self):
        print("Training models with Optuna...")

        def objective(trial):
            model_name = trial.suggest_categorical("model_name", ["LogisticRegression", "KNeighborsClassifier"])

            if model_name == "LogisticRegression":
                c = trial.suggest_float("C", 0.01, 10.0, log=True)
                model = LogisticRegression(C=c, solver="liblinear", random_state=42)
            else:
                k = trial.suggest_int("n_neighbors", 3, 10)
                model = KNeighborsClassifier(n_neighbors=k)

            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_val)
            return f1_score(self.y_val, y_pred)

        self.best_models = {}
        model_names = ["LogisticRegression", "KNeighborsClassifier"]

        for model_name in model_names:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: objective(trial), n_trials=int(self.n_trials))

            best_params = study.best_params
            best_model_name = best_params.pop("model_name", model_name)

            if best_model_name == "LogisticRegression":
                best_model = LogisticRegression(**best_params, solver="liblinear", random_state=42)
            else:
                best_model = KNeighborsClassifier(**best_params)

            best_model.fit(self.X_train, self.y_train)

            mlflow.set_experiment("diabetes_classification")
            with mlflow.start_run(run_name=f"{best_model_name}_Optuna_Training"):
                mlflow.log_params({"model_name": best_model_name, **best_params})
                input_example = self.X_train.iloc[[0]] if hasattr(self.X_train, "iloc") else self.X_train[:1]
                signature = infer_signature(self.X_train, best_model.predict(self.X_train))

                mlflow.sklearn.log_model(
                    best_model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )

            self.best_models[best_model_name] = {
                "model": best_model,
                "best_params": best_params
            }

        self.next(self.test_models)

    @step
    def test_models(self):
        print("Testing model on the test set...")

        self.evaluation_results = {}
        os.makedirs("results", exist_ok=True)

        for name, info in self.best_models.items():
            model = info["model"]
            y_pred = model.predict(self.X_test)

            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred)
            rec = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)

            report_path = f"results/{name}_classification_report.json"
            with open(report_path, "w") as f:
                json.dump(report, f)

            cm_path = f"results/{name}_confusion_matrix.png"
            plt.figure()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close()

            with mlflow.start_run(run_name=f"{name}_Evaluation"):
                input_example = self.X_test.iloc[[0]] if hasattr(self.X_test, "iloc") else self.X_test[:1]
                signature = infer_signature(self.X_test, model.predict(self.X_test))

                mlflow.sklearn.log_model(
                    model,
                    artifact_path="model",
                    input_example=input_example,
                    signature=signature
                )

                mlflow.log_params(info["best_params"])
                mlflow.log_metrics({
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1
                })
                mlflow.log_artifact(report_path)
                mlflow.log_artifact(cm_path)

            self.evaluation_results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            }

        self.next(self.save_models)

    @step
    def save_models(self):
        print("Saving trained models...")
        os.makedirs("saved_models", exist_ok=True)

        for name, info in self.best_models.items():
            path = f"saved_models/{name}_model.pkl"
            joblib.dump(info["model"], path)
            print(f"Saved model: {name} -> {path}")

        self.next(self.end)

    @step
    def end(self):
        print("Pipeline completed.")
        print("Evaluation results:")
        for name, metrics in self.evaluation_results.items():
            print(f"\n{name}:")
            for metric, value in metrics.items():
                print(f"  - {metric}: {value:.4f}")

