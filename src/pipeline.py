import os
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from metaflow import FlowSpec, step, Parameter
from src.data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

class DiabetesClassificationFlow(FlowSpec):

    # rf_n_estimators = Parameter("rf_n_estimators", default="100,150,200")
    # rf_max_depth = Parameter("rf_max_depth", default="None,10")
    # rf_class_weight = Parameter("rf_class_weight", default="balanced,None")

    logreg_c = Parameter("logreg_c", default="0.1,1.0")

    knn_neighbors = Parameter("knn_neighbors", default="3,5,7,10")

    # dt_max_depth = Parameter("dt_max_depth", default="None,5,10")
    # dt_min_samples_split = Parameter("dt_min_samples_split", default="2, 5")

    @step
    def start(self):
        print("Starting the pipeline...")
        self.next(self.load_and_preprocess)

    @step
    def load_and_preprocess(self):
        print("Loading and preprocessing data...")
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test) = load_and_preprocess_data()
        self.next(self.train_models)

    @step
    def train_models(self):
        print("Training models with GridSearchCV...")

        self.best_models = {}

        def parse_param(val):
            def convert(v):
                v = v.strip()
                if v == "None":
                    return None
                elif v in ("balanced", "balanced_subsample"):
                    return v  # giữ nguyên chuỗi
                try:
                    return int(v)
                except ValueError:
                    try:
                        return float(v)
                    except ValueError:
                        return v  # fallback cuối cùng là chuỗi
            return [convert(v) for v in val.split(",")]

        self.model_configs = {
            # "RandomForest": {
            #     "model": RandomForestClassifier(random_state=42),
            #     "params": {
            #         "n_estimators": parse_param(self.rf_n_estimators),
            #         "max_depth": parse_param(self.rf_max_depth),
            #         "class_weight": parse_param(self.rf_class_weight)
            #     }
            # },
            "LogisticRegression": {
                "model": LogisticRegression(solver="liblinear", random_state=42),
                "params": {
                    "C": parse_param(self.logreg_c)
                }
            },
            "KNeighborsClassifier": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": parse_param(self.knn_neighbors)
                }
            }
            # },
            # "DecisionTree": {
            #     "model": DecisionTreeClassifier(random_state=42),
            #     "params": {
            #         "max_depth": parse_param(self.dt_max_depth),
            #         "min_samples_split": parse_param(self.dt_min_samples_split)
            #     }
            # }
        }

        mlflow.set_experiment("diabetes_classification")

        for name, config in self.model_configs.items():
            print(f"Tuning model: {name}")
            grid_search = GridSearchCV(
                config["model"], config["params"], cv=3, scoring="f1", n_jobs=2
            )
            grid_search.fit(self.X_train, self.y_train)

            best_model = grid_search.best_estimator_

            with mlflow.start_run(run_name=f"{name}_Training"):
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_param("model_name", name)
                mlflow.sklearn.log_model(best_model, artifact_path="model")

            self.best_models[name] = {
                "model": best_model,
                "best_params": grid_search.best_params_
            }

        self.next(self.validate_models)

    @step
    def validate_models(self):
        print("Validating models on the validation set...")

        self.validation_scores = {}

        for name, info in self.best_models.items():
            model = info["model"]
            y_pred_val = model.predict(self.X_val)
            f1_val = f1_score(self.y_val, y_pred_val)
            self.validation_scores[name] = f1_val
            print(f"{name} - F1 Validation Score: {f1_val:.4f}")

        self.next(self.test_models)

    @step
    def test_models(self):
        print("Testing models on the test set...")

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
                mlflow.sklearn.log_model(model, artifact_path="model")
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
