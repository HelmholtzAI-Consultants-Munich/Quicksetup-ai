# Quickstart
## TL;DR
```bash
# given that project_name is `Quicksetup-ai` via cookiecutter installation step
cd Quicksetup-ai/
conda create -n quicksetup_ai_env python=3.9
conda activate quicksetup_ai_env
pip install -e .
python scripts/train.py
```

## Create the pipeline environment
* First, create a virtual environment (for the sake of the example, we'll call it `quicksetup_ai_env`).
> You can either do it with conda (preferred) or venv.
Using conda:
```bash
conda create -n quicksetup_ai_env python=3.9
```

* Then, activate the environment
```bash
conda activate quicksetup_ai_env
```

## Install the quicksetup-ai package
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
