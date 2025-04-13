# CS317
This project consist of source code for trainning model as well as tracking via mlflow and metaflow

(The project should run on ubuntu os if you are using window)

First you have to create an enviroment in your workdir(MLOPS) and active metaflow-dev by 

```
python3 -m venv metaflow_env
source metaflow_env/bin/activate
pip install metaflow
metaflow-dev up
```

After running meta flow-dev up then press enter when it ask which service you are going to use 

your ubuntu should look like this

![alt text](https://github.com/truong04/MLOPS/blob/main/image/metaflow-dev-screen.png?raw=true)

Then click on http://localhost:10350/ and wait for all service full activate


After that open another ubuntu terminal and go to your workdir where you activate metaflow-dev and type the following code one by one

IF YOU DO NOT HAVE CONDA AND MAMBA ELSE SKIP THIS
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda install mamba -n base -c conda-forge
```

```
eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate

metaflow-dev shell

eval "$(mamba shell hook --shell bash)"
mamba activate metaflow-dev
source metaflow_env/bin/activate

pip install -r requirement.txt
```

Now you can run your app and watch the pipeline every time you type 
```
python ur_ap[_name.py run # click on the url to go to metaflow ui
mlflow ui #to see the model performance and version in mlflow ui
```

Your teminal screen should look like this

![alt text](https://github.com/truong04/MLOPS/blob/main/image/RESULT.png?raw=true)

