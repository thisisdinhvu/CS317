# CS317: Lab 1

This project is part of the CS317 course and demonstrates a complete machine learning workflow—from data preprocessing to model training, evaluation, and deployment—using **Metaflow** and **MLflow**. The pipeline is designed to train multiple models and track their performance in a reproducible and scalable way.

---

## 🔍 Project Overview

This project tackles the problem of diabetes classification using multiple machine learning models. The key highlights include:

- **Multi-model training** with hyperparameter tuning using `GridSearchCV`.
- **Model tracking** and **versioning** with MLflow.
- **Pipeline orchestration** using Metaflow, enabling step-wise execution, caching, and visualization.
- **Evaluation artifacts** such as confusion matrices and classification reports are saved automatically.
- **Modular and extensible design**, allowing easy addition of new models or metrics.

---

## 🧠 Pipeline Structure

The machine learning pipeline is built using [Metaflow](https://docs.metaflow.org/) and includes the following steps:

1. ### `start`
   - Initializes the flow and starts the pipeline.

2. ### `load_and_preprocess`
   - Loads the dataset and applies preprocessing (e.g., cleaning, scaling, splitting).
   - The data is split into train, validation, and test sets.

3. ### `train_models`
   - Trains four different classifiers:  
     - Random Forest  
     - Logistic Regression  
     - K-Nearest Neighbors  
     - Decision Tree
   - Hyperparameters are tuned using `GridSearchCV`.
   - Best models are logged to MLflow with their optimal parameters.

4. ### `validate_models`
   - Each trained model is evaluated on the validation set.
   - F1 scores are computed to help compare model performance before testing.

5. ### `test_models`
   - The selected models are evaluated on the test set.
   - Metrics like accuracy, precision, recall, F1-score, and confusion matrix are logged.
   - Classification reports and confusion matrices are saved locally and in MLflow.

6. ### `save_models`
   - Best trained models are serialized using `joblib` and saved to disk for future use.

7. ### `end`
   - Summarizes and prints evaluation results of all models.

---

## 🛠 Technologies Used

| Framework       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| **Metaflow**    | Used to define and manage the ML pipeline with ease, enabling reproducibility and UI visualization. |
| **MLflow**      | Provides model tracking, logging, and a UI for monitoring experiments and metrics. |
| **scikit-learn**| Used for building and training machine learning models, including hyperparameter tuning. |
| **matplotlib / seaborn** | Used for visualizing confusion matrices. |
| **joblib**      | Saves models efficiently to disk for later use. |

---

## 🚀 Installation & Setup

> **Note**: This project is intended to run on **Ubuntu**. If you're on Windows, consider using WSL or a VM.

### 1. Clone the Repository

```bash
git clone https://github.com/thisisdinhvu/CS317
cd CS317
```

### 2. Set Up Python Environment

```bash
python3 -m venv metaflow_env
source metaflow_env/bin/activate
pip install metaflow
metaflow-dev up
(remember to install docker desktop , tutorial: https://www.youtube.com/watch?v=ZyBBv1JmnWQ&ab_channel=CodeBear)
```

Press `Enter` when prompted by `metaflow-dev`.

Visit: [http://localhost:10350/](http://localhost:10350/)  
Wait until all services are fully active.

Your terminal should look like this after setup:

![Metaflow Terminal](https://github.com/truong04/MLOPS/blob/main/image/metaflow-dev-screen.png?raw=true)

### 3. (Optional) Set Up Conda and Mamba

If you don’t already have them:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda install mamba -n base -c conda-forge
```

Activate your environments:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate
```

### 4. Install Project Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run the Pipeline

From the project root, activate your environment and run:

```bash
python app.py run
```

You’ll see a URL in the terminal where you can view your pipeline in the Metaflow UI.

To view model performance and experiment details in MLflow:

```bash
mlflow ui
```

Then go to: [http://localhost:5000](http://localhost:5000)

After running your pipeline, your terminal may look like this:

![Metaflow Results](https://github.com/truong04/MLOPS/blob/main/image/RESULT.png?raw=true)

---

## 📂 Output

- Trained models saved to `saved_models/`
- Evaluation metrics and visualizations saved to `results/`
- All runs and artifacts are logged in MLflow

---

## Evaluation

Each model is evaluated with:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

You can find all metrics in the terminal logs and MLflow dashboard.

---

## 📬 Contact

For any questions or suggestions, please open an issue or reach out via GitHub.

---

