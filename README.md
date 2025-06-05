<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS317.P21 - PHÁT TRIỂN VÀ VẬN HÀNH HỆ THỐNG MÁY HỌC</b></h1>

## COURSE INTRODUCTION
<a name="gioithieumonhoc"></a>
* *Course Title*: Phát triển và vận hành hệ thống máy học
* *Course Code*: CS317.P21
* *Year*: 2024-2025

## ACADEMIC ADVISOR
<a name="giangvien"></a>
* *Đỗ Văn Tiến* - tiendv@uit.edu.vn
* *Lê Trần Trọng Khiêm* - khiemltt@uit.edu.vn

## MEMBERS
<a name="thanhvien"></a>
* Nguyễn Đình Vũ - 22521692
* Ngô Thành Trung - 22521560
* Đinh Nhật Trường - 22521575

---
# CS317: Lab 1 – Diabetes Classification Pipeline

This project is part of the **CS317** course and demonstrates a full **machine learning workflow** using **Metaflow** and **MLflow**—from data preprocessing to model training, evaluation, and deployment. The pipeline is designed to be **modular**, **reproducible**, and **scalable**, supporting multi-model training and experiment tracking.

---

##  Overview

We tackle the **diabetes classification** problem by **predicting the risk of diabetes** from medical records using multiple machine learning models. This project showcases:

-  **Multi-model training** with hyperparameter tuning via `Optuna`.
-  **Experiment tracking and versioning** with MLflow.
-  **Step-wise pipeline orchestration** using Metaflow.
-  Automatic saving of evaluation artifacts like **confusion matrices** and **classification reports**.
-  **Modular design**—easily add new models or metrics.

---

##  Pipeline Structure

Built using [Metaflow](https://docs.metaflow.org/), the pipeline consists of these stages:

1. ### `start`
   - Initializes the pipeline.

2. ### `load_and_preprocess`
   - Loads and cleans the dataset.
   - Applies feature scaling and splits data into train, validation, and test sets.

3. ### `train_models`
   - Trains the following models with `Optuna` for hyperparameter tuning:
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

##  Technologies

| Tool/Library         | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Metaflow**         | Orchestrates the ML pipeline with step-wise execution and visualization |
| **MLflow**           | Logs model performance, parameters, and artifacts                        |
| **scikit-learn**     | Core ML library                                                          |
| **Optuna**           | Efficient hyperparameter optimization                                    |
| **matplotlib / seaborn** | Visualizes evaluation metrics like confusion matrices             |
| **joblib**           | Efficiently saves trained models                                         |
| **Docker**           | Required for running `metaflow-dev` environment                          |

---

##  Installation & Setup

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

##  Running the Pipeline

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

##  Output Files

- Trained models: `saved_models/`
- Evaluation reports and plots: `results/`
- Logged runs and artifacts: MLflow UI

---

##  Evaluation Metrics

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

##  Contact

For questions, feedback, or contributions, feel free to open an issue or contact via GitHub.

---

# CS317: Lab 2 – API, Docker and Deploy API

This project is part of the CS317 course and demonstrates a full machine learning workflow using FastAPI, Docker and Remote Server to create and deploy usable API.

##  Pipeline Structure

Built using FastAPI, the pipeline consists of these stages:

Note: Clone the GitHub repository to your personal computer.

###  Create API with FastAPI
```bash
app = FastAPI()
uvicorn main:app --host 0.0.0.0 --port 8080
```

###  Pull Docker Image from Docker Hub 

Instead of building the image manually, you can directly pull my pre-built image from Docker Hub:

```bash
docker pull imisschunyuan/diabetes-api
docker run -d -p 8080:8080 imisschunyuan/diabetes-api
```

###  Build docker image locally
```bash
docker build -t <image_name> .
docker run -d -p 8080:8080 --name <container_name> <image_name>
```

###  Docker compose locally
> Requires `docker-compose.yaml` file
```bash
docker compose up --build
```

###  Testing built APIS
> Access this URL to use FastAPI Swagger to test the API:
[http://0.0.0.0:8080/docs](http://0.0.0.0:8080/docs)

###  Deploy API via remote server
- Access the remote server via SSH
- Install Docker as guided (for Ubuntu)
- Clone the GitHub repo to the server
- Run the service with:
```bash
docker compose up --build
```

###  Access API externally
> Docker runs locally on the server and can be accessed via:
```
http://<SERVER_IP>:8080/docs
```
> You must also maintain OpenVPN connection.

---

##  Technologies

| Tool/Library | Purpose |
|--------------|---------|
| **FastAPI** | Modern, high-performance web framework for APIs |
| **Docker** | Environment-agnostic app packaging and deployment |

---

##  Results

-  Serving API and Docker locally
-  Serving API and Docker on remote server
-  Push my prebuilt image on Docker Hub (https://hub.docker.com/r/imisschunyuan/diabetes-api)
---

## Demo:

[Serving API and Docker locally](https://www.youtube.com/watch?v=ldGWFFqCT4s&ab_channel=TrungNg%C3%B4Th%C3%A0nh)

[Serving API and Docker on remote server](https://www.youtube.com/watch?v=pX-mLY8qgQs&ab_channel=TrungNg%C3%B4Th%C3%A0nh)

-  Note: The API serving on the server uses **ngrok** for testing purposes, allowing the creation of a public URL that provides internet access to the local network on the server. It is **not recommended** to expose critical APIs through ngrok.


##  Contact

For questions, feedback, or contributions, feel free to open an issue or contact via GitHub.

---

Dưới đây là nội dung README tiếp tục cho **Lab 3** trong dự án CS317, theo đúng yêu cầu bạn nêu:

---

# CS317: Lab 3 – Monitoring & Logging for Machine Learning API

This lab extends the API developed in **Lab 2** by integrating **monitoring** and **logging services** to observe system performance, track model metrics, and capture logs from multiple sources for debugging and auditing.

---

##  Objectives

* Monitor **server-level metrics**: CPU, RAM, Disk, Network I/O (and optional GPU).
* Monitor **API performance**: requests/sec, latency, error rate.
* Monitor **model inference**: response time (CPU/GPU), confidence scores.
* Capture logs from:

  * System logs (`syslog`)
  * Application logs (`stdout`, `stderr`)
  * Custom log files (`logfile.log`)
* Setup **alerts** (optional) and centralized logging via Fluentd, Prometheus, Grafana.

---

##  Technologies

| Tool/Service         | Purpose                                 |
| -------------------- | --------------------------------------- |
| **FastAPI**          | API framework                           |
| **Docker**           | Containerization                        |
| **Prometheus**       | Metrics collection                      |
| **Grafana**          | Monitoring dashboards                   |
| **Fluentd**          | Centralized log collector               |
| **node\_exporter**   | Server resource exporter for Prometheus |
| **uvicorn/gunicorn** | ASGI server for running FastAPI         |
| **psutil**           | Python system metrics (used in API)     |
| **loguru**           | Structured Python logging               |

---

##  Setup and Run

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/CS317
cd CS317
```

### 2. Launch Full Monitoring Stack

```bash
docker compose up --build
```

### 3. Access Services

| Service              | URL                                                      |
| -------------------- | -------------------------------------------------------- |
| FastAPI              | [http://localhost:8080/docs](http://localhost:8080/docs) |
| Prometheus           | [http://localhost:9090](http://localhost:9090)           |
| Grafana              | [http://localhost:3000](http://localhost:3000)           |
| Fluentd Logs (local) | `docker/fluentd/log/` folder                             |

> **Default Grafana login**:
> Username: `admin`
> Password: `admin`

---

## Metrics Monitored

### Server Resources (via `node_exporter`)

* CPU Usage
* Memory (RAM) Usage
* Disk Usage & I/O
* Network I/O (tx/rx)
* 
### API Performance (via Prometheus + FastAPI Middleware)

* Requests per second
* Average Latency
* HTTP Error Rate

### Model Metrics (via custom Prometheus exporter)

* Inference latency (CPU/GPU)
* Confidence score (mean, min, max)
* Input batch size

---

## 📈 Grafana Dashboards

> Dashboards available in `docker/grafana/dashboards/` or imported via UI

**Sample Dashboards:**

* **System Overview**: CPU, RAM, Disk, Network
* **API Performance**: Latency, RPS, Error Rate
* **Model Monitoring**: Inference time, confidence trends

---

## 🚨 Optional: Alerting with Alertmanager

> For bonus/advanced setup:

* Configure `Alertmanager` to trigger alerts on:

  * High CPU usage
  * Low confidence scores
  * > 50% error rate
  * High latency or inference time

Alerts can be sent to:

* **Email**
* **Slack**
* **Telegram**
* **Webhook** (for auto model retrain)

---

##  Testing & Validation

After deployment, use the following tools:

```bash
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" \
     -d '{"age": 35, "glucose": 180, "bmi": 33, "blood_pressure": 85}'
```

* Check Prometheus `/metrics`
* Check FastAPI Swagger UI: [http://localhost:8080/docs](http://localhost:8080/docs)
* Check logs in `docker/fluentd/log/`
* View dashboards in Grafana

---

## Demo

* [Monitoring System and API with Grafana + Prometheus](https://youtu.be/YOUR_VIDEO_LINK)
* [Centralized Logging with Fluentd](https://youtu.be/YOUR_VIDEO_LINK)

---

---

## Contact

Feel free to open an issue or email for bugs or feature requests.

---

