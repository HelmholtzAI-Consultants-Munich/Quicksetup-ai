# Quickstart
## TL;DR
```bash
# given that project_name is `ML-Pipeline-Template` via cookiecutter installation step
cd ML-Pipeline-Template/
conda create -n venv python=3.9
conda activate venv
pip install -r requirements.txt
pip install -e .
python scripts/train.py
```

## Create the pipeline environment
The libraries used by the pipeline are all listed in `requirements.txt`.
* First, create a virtual environment (for the sake of the example, we'll call it `ml_template_env`).
> You can either do it with conda (preferred) or venv.
Using conda:
```bash
conda create -n venv python=3.9
```

* Then, activate the environment
```bash
conda activate venv
```

* Finally, install all dependencies using `pip`. Run:
```bash
pip install -r requirements.txt
 ```

## Install the ml-pipeline-template package
Before using the template, one needs to install the project as a package. Run:
```bash
pip install -e .
```
## Run the MNIST example
This pipeline comes with a toy example (MNIST dataset with a simple feedforward neural network). To run the training (resp. testing) pipeline, simply run:
```
python scripts/train.py
# or python scripts/test.py
```
Or, if you want to submit the training job to a submit (resp. interactive) cluster node via slurm, run:
```
sbatch job_submission.sbatch
# or sbatch job_submission_interactive.sbatch
```
> * The experiments, evaluations, etc., are stored under the `logs` directory.
> * The default experiments tracking system is mlflow. The `mlruns` directory is contained in `logs`. To view a user friendly view of the experiments, run:
> ```
> # make sure you are inside logs (where mlruns is located)
> mlflow ui --host 0000
> ```
> * When evaluating (running `test.py`), make sure you give the correct checkpoint path in `configs/test.yaml`
