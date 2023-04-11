

<div align="center">

# Quicksetup-ai: A flexible template as a quick setup for deep learning projects in research
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a>
<a href="https://github.com/pyscaffold/pyscaffoldext-dsproject"><img alt="Template" src="https://img.shields.io/badge/-Pyscaffold--Datascience-017F2F?style=flat&logo=github&labelColor=gray"></a>

[Docs] | [Quickstart] | [Tutorials] |

[Docs]: https://quicksetup-ai.readthedocs.io/
[Quickstart]: https://quicksetup-ai.readthedocs.io/en/latest/notes/getting_started/quickstart.html
[Tutorials]: https://quicksetup-ai.readthedocs.io/en/latest/index.html#:~:text=TUTORIALS-,How%20to%20set%20up%20a%20different%20model,-Define%20the%20new

</div>

# Description

This template is a combination of [pyscaffold datascience](https://github.com/pyscaffold/pyscaffoldext-dsproject) and [lightning-hydra](https://github.com/ashleve/lightning-hydra-template). It provides a general baseline for Deep Learning projects including: 
* A predefined structure which simplifies the development of the project.
* A set of tools for experiment tracking, hyper parameter search and rapid experimentation using configuration files. More details in [lightning-hydra](https://github.com/ashleve/lightning-hydra-template).
* Pre-commit hooks and automatic documentation generation.

> :warning: **Package compatibility**: This template relies on Pytorch Lightning (whose API might change) we use a fixed version of the package to ensure the template doesn't break

# Installation
## Using Cookiecutter
1. Create and activate your environment:
    ```bash
    conda create -y -n venv_cookie python=3.9 && conda activate venv_cookie
    ```

2. Install cookiecutter in your environment:
    ```bash
    pip install cookiecutter dvc
    ```
3. Create your own project using this template via cookiecutter:
    ```bash
    cookiecutter https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai.git
    ```
   
# Quickstart
## Create the pipeline environment and install the ml-pipeline-template package
Before using the template, one needs to install the project as a package.
* First, create a virtual environment. 
> You can either do it with conda (preferred) or venv.
* Then, activate the environment
* Finally, install the project as a package. Run:
```
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


# Project Organization
```
├── configs                              <- Hydra configuration files
│   ├── callbacks                               <- Callbacks configs
│   ├── datamodule                              <- Datamodule configs
│   ├── debug                                   <- Debugging configs
│   ├── experiment                              <- Experiment configs
│   ├── hparams_search                          <- Hyperparameter search configs
│   ├── local                                   <- Local configs
│   ├── log_dir                                 <- Logging directory configs
│   ├── logger                                  <- Logger configs
│   ├── model                                   <- Model configs
│   ├── trainer                                 <- Trainer configs
│   │
│   ├── test.yaml                               <- Main config for testing
│   └── train.yaml                              <- Main config for training
│
├── data                                 <- Project data
│   ├── processed                               <- Processed data
│   └── raw                                     <- Raw data
│
├── docs                                 <- Directory for Sphinx documentation in rst or md.
├── models                               <- Trained and serialized models, model predictions
├── notebooks                            <- Jupyter notebooks.
├── reports                              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                                 <- Generated plots and figures for reports.
├── scripts                              <- Scripts used in project
│   ├── job_submission.sbatch               <- Submit training job to slurm
│   ├── job_submission_interactive.sbatch   <- Submit training job to slurm (interactive node)
│   ├── test.py                             <- Run testing
│   └── train.py                            <- Run training
│
├── src/<your_project_name>              <- Source code
│   ├── datamodules                             <- Lightning datamodules
│   ├── models                                  <- Lightning models
│   ├── utils                                   <- Utility scripts
│   │
│   ├── testing_pipeline.py
│   └── training_pipeline.py
│
├── tests                                <- Tests of any kind
│   ├── helpers                                 <- A couple of testing utilities
│   ├── shell                                   <- Shell/command based tests
│   └── unit                                    <- Unit tests
│
├── .coveragerc                          <- Configuration for coverage reports of unit tests.
├── .gitignore                           <- List of files/folders ignored by git
├── .pre-commit-config.yaml              <- Configuration of pre-commit hooks for code formatting
├── setup.cfg                            <- Configuration of linters and pytest
├── LICENSE.txt                          <- License as chosen on the command-line.
├── pyproject.toml                       <- Build configuration. Don't change! Use `pip install -e .`
│                                           to install for development or to build `tox -e build`.
├── setup.cfg                            <- Declarative configuration of your project.
├── setup.py                             <- [DEPRECATED] Use `python setup.py develop` to install for
│                                           development or `python setup.py bdist_wheel` to build.
└── README.md
```
