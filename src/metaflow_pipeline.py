# metaflow_pipeline.py
from metaflow import FlowSpec, step
import mlflow
import mlflow.sklearn
from src.data_preprocessing import load_and_preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

class DiabetesPipelineFlow(FlowSpec):

    @step
    def start(self):
        print("🔹 Bắt đầu pipeline: load và xử lý dữ liệu")
        (self.X_train, self.X_val, self.X_test,
         self.y_train, self.y_val, self.y_test,
         self.selected_features) = load_and_preprocess_data()
        self.next(self.train_models)

    @step
    def train_models(self):
        print("🔹 Huấn luyện và tối ưu mô hình")

        self.results = {}
        # Hyperparameter tuning cho các mô hình
        models = {
            "RandomForest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10],
                    "min_samples_split": [2]
                }
            },
            "LogisticRegression": {
                "model": LogisticRegression(solver='liblinear', random_state=42),
                "params": {
                    "C": [0.1, 1.0],
                    "penalty": ["l2"]
                }
            },
            "KNeighborsClassifier": {
                "model": KNeighborsClassifier(),
                "params": {
                    "n_neighbors": [3, 5],
                    "weights": ["uniform"]
                }
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(random_state=42),
                "params": {
                    "max_depth": [None, 10],
                    "min_samples_split": [2]
                }
            }
        }

        for model_name, config in models.items():
            print(f"Đang huấn luyện mô hình {model_name}...")
            with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
                grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring='f1', n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)

                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(self.X_val)

                acc = accuracy_score(self.y_val, y_pred)
                prec = precision_score(self.y_val, y_pred)
                rec = recall_score(self.y_val, y_pred)
                f1 = f1_score(self.y_val, y_pred)

                mlflow.sklearn.log_model(best_model, "model")
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_param("selected_features", self.selected_features.tolist())

                self.results[model_name] = {
                    "model": best_model,
                    "f1_score": f1
                }

        self.next(self.select_best_model)

    @step
    def select_best_model(self):
        print("🔹 Chọn mô hình tốt nhất theo f1_score")
        best_model_name = max(self.results, key=lambda k: self.results[k]["f1_score"])
        self.best_model = self.results[best_model_name]["model"]
        self.best_model_name = best_model_name
        print(f"✅ Mô hình tốt nhất: {best_model_name}")
        self.next(self.test_model)

    @step
    def test_model(self):
        print("🔹 Đánh giá mô hình trên tập test")
        y_pred = self.best_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        prec = precision_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)

        from metaflow.cards import Markdown
        self.card = Markdown(f"""
        ## Kết quả kiểm thử
        **Mô hình:** {self.best_model_name}  
        **Accuracy:** {acc:.4f}  
        **Precision:** {prec:.4f}  
        **Recall:** {rec:.4f}  
        **F1 Score:** {f1:.4f}  
        """)

        self.next(self.save_model)

    def save_model(self):
        print("💾 Lưu mô hình tốt nhất")
        model_path = f"saved_models/{self.best_model_name}_best_model.pkl"
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(self.best_model, model_path)
        print(f"✅ Mô hình đã lưu tại: {model_path}")

        # Log file vào MLflow artifact
        with mlflow.start_run(run_name=f"{self.best_model_name}_FinalSave"):
            mlflow.log_artifact(model_path)
            mlflow.log_param("selected_features", self.selected_features.tolist())
            mlflow.log_param("model_name", self.best_model_name)

        self.next(self.end)

    @step
    def end(self):
        print("🎉 Pipeline Metaflow hoàn tất!")

