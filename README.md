## 👥 Members

- Nguyễn Đình Vũ - 22521692  
- Ngô Thành Trung - 22521560  
- Đinh Nhật Trường - 22521575

---
  
# CS317: Lab 1 – Diabetes Classification Pipeline

This project is part of the **CS317** course and demonstrates a full **machine learning workflow** using **Metaflow** and **MLflow**—from data preprocessing to model training, evaluation, and deployment. The pipeline is designed to be **modular**, **reproducible**, and **scalable**, supporting multi-model training and experiment tracking.

---

## 🔍 Overview

We tackle the **diabetes classification** problem using multiple machine learning models. This project showcases:

- 🔁 **Multi-model training** with hyperparameter tuning via `GridSearchCV`.
- 📈 **Experiment tracking and versioning** with MLflow.
- 🧹 **Step-wise pipeline orchestration** using Metaflow.
- 📊 Automatic saving of evaluation artifacts like **confusion matrices** and **classification reports**.
- 🧱 **Modular design**—easily add new models or metrics.

---

## 🧠 Pipeline Structure

Built using [Metaflow](https://docs.metaflow.org/), the pipeline consists of these stages:

1. ### `start`
   - Initializes the pipeline.

2. ### `load_and_preprocess`
   - Loads and cleans the dataset.
   - Applies feature scaling and splits data into train, validation, and test sets.

3. ### `train_models`
   - Trains the following models with `GridSearchCV` for hyperparameter tuning:
     - Random Forest
     - Logistic Regression
     - K-Nearest Neighbors
     - Decision Tree
   - Logs all models and parameters to MLflow.

4. ### `validate_models`
   - Evaluates models on the validation set using F1 Score for comparison.

5. ### `test_models`
   - Evaluates final models on the test set.
   - Logs accuracy, precision, recall, F1 score, and confusion matrix to MLflow.
   - Saves visual reports locally and to MLflow.

6. ### `save_models`
   - Saves trained models to disk using `joblib`.

7. ### `end`
   - Prints a summary of results for all models.

---

## 🚰 Technologies

| Tool/Library         | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Metaflow**         | Orchestrates the ML pipeline with step-wise execution and visualization |
| **MLflow**           | Logs model performance, parameters, and artifacts                        |
| **scikit-learn**     | Core ML library (models + `GridSearchCV`)                                |
| **matplotlib / seaborn** | Visualizes evaluation metrics like confusion matrices             |
| **joblib**           | Efficiently saves trained models                                         |
| **Docker**           | Required for running `metaflow-dev` environment                          |

---

## ⚙️ Installation & Setup

> **Note**: This project is optimized for **Ubuntu/Linux** systems. If you're using **Windows**, consider using **WSL** or a **virtual machine**.

### 1. Clone the Repository

```bash
git clone https://github.com/thisisdinhvu/CS317
cd CS317
```

### 2. Set Up Python Environment

```bash
python3 -m venv metaflow_env
source metaflow_env/bin/activate
```

### 3. Install Docker (if you haven't installed it before)

Follow this guide to install Docker on Ubuntu:
[YouTube: Docker Desktop Setup](https://www.youtube.com/watch?v=ZyBBv1JmnWQ&ab_channel=CodeBear)

### 4. Set Up Metaflow Environment

Install Metaflow and launch the local development environment:

```bash
pip install metaflow
metaflow-dev up
```

> When prompted, **press `Enter`** to continue setup.

Once ready, go to: [http://localhost:10350/](http://localhost:10350/)  
Wait until all services are marked as **healthy**.

You should see a screen similar to this:

![Metaflow Terminal](https://github.com/truong04/MLOPS/blob/main/image/metaflow-dev-screen.png?raw=true)

### 5. (Optional) Install Conda and Mamba

Recommended for managing dependencies:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda install mamba -n base -c conda-forge
```

Then activate:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate
```

### 6. Prepare Environment & Install Dependencies

Ensure `metaflow-dev` shell is active:

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate

metaflow-dev shell

# Re-activate environment after entering metaflow shell

eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate

pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

Once your environment is ready, run the pipeline:

```bash
python app.py run
```

To view the pipeline in Metaflow UI:  
Visit [http://localhost:10350/](http://localhost:10350/)

To view experiment logs and metrics in MLflow:

```bash
mlflow ui
```

Then go to: [http://localhost:5000](http://localhost:5000)

Sample output after running:

![Metaflow Results](https://github.com/truong04/MLOPS/blob/main/image/RESULT.png?raw=true)

---

## 📂 Output Files

- Trained models: `saved_models/`
- Evaluation reports and plots: `results/`
- Logged runs and artifacts: MLflow UI

---

## 📊 Evaluation Metrics

Each model is assessed with:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Full Classification Report

All results are printed in the terminal and stored in MLflow.

---
## Tutorial: [For MAC](https://www.youtube.com/watch?v=mCJgK6Eq-nE)
---

## 📬 Contact

For questions, feedback, or contributions, feel free to open an issue or contact via GitHub.

