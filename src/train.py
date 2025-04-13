# train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from data_preprocessing import load_and_preprocess_data

def train_model():
    X_train, X_val, X_test, y_train, y_val, y_test, selected_features = load_and_preprocess_data()

    models = {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(solver='liblinear', random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "penalty": ["l1", "l2"]
            }
        },
        "KNeighborsClassifier": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        }
    }

    for model_name, config in models.items():
        with mlflow.start_run(run_name=f"{model_name}_GridSearch"):
            grid_search = GridSearchCV(config["model"], config["params"], cv=3, scoring='f1', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_val)

            mlflow.sklearn.log_model(best_model, "model")
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("accuracy", accuracy_score(y_val, y_pred))
            mlflow.log_metric("precision", precision_score(y_val, y_pred))
            mlflow.log_metric("recall", recall_score(y_val, y_pred))
            mlflow.log_metric("f1_score", f1_score(y_val, y_pred))
            mlflow.log_param("selected_features", selected_features.tolist())