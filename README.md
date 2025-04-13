# CS317 Project Guide

This project contains source code for training models as well as tracking them via **MLflow** and **Metaflow**.

**Note:** The project is designed to run on **Ubuntu OS**. If you are using **Windows**, follow the corresponding steps for your environment.

---

## 1. Setting Up Your Environment

Navigate to your working directory (e.g., `MLOPS`) and create a new Python virtual environment for Metaflow.

### Create a Python Virtual Environment

```bash
python3 -m venv metaflow_env
source metaflow_env/bin/activate
pip install metaflow
metaflow-dev up
```

When you run `metaflow-dev up`, press `Enter` when prompted to select the service.

### After Services are Activated

Once the services are fully activated, your Ubuntu terminal should look like this:

![Metaflow Terminal](https://github.com/truong04/MLOPS/blob/main/image/metaflow-dev-screen.png?raw=true)

Now, open your browser and visit [http://localhost:10350/](http://localhost:10350/) to ensure all services are fully activated.

---

## 2. Setting Up Conda and Mamba (Optional)

If you **do not** have Conda or Mamba installed, skip this section. If you do, follow the steps below:

### Install Conda and Mamba

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda install mamba -n base -c conda-forge
```

### Activate Mamba and Your Environment

```bash
eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate
```

---

## 3. Installing Dependencies

Now, install all the required dependencies by running:

```bash
pip install -r requirements.txt
```

---

## 4. Running the Application

You can now run your app and monitor the pipeline in the **Metaflow UI** by executing the following command:

```bash
python your_app_name.py run
```

Visit the Metaflow UI to track the progress of the pipeline.

To monitor model performance and versions in **MLflow**, run:

```bash
mlflow ui
```

Your terminal screen should look like this:

![Metaflow Results](https://github.com/truong04/MLOPS/blob/main/image/RESULT.png?raw=true)

---
