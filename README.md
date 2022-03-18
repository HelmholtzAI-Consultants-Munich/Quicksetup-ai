

<div align="center">

# Your Project Name
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)


Template based from:
- https://github.com/pyscaffold/pyscaffoldext-dsproject
- https://github.com/ashleve/lightning-hydra-template/blob/main/README.md


</div>

## Description

Project description goes here.

## Project Organization
```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── local                    <- Local configs
│   ├── log_dir                  <- Logging directory configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── test.yaml             <- Main config for testing
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│   ├── processed                <- Main config for testing
│   └── raw                      <- Main config for training
│
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── models                 <- Trained and serialized models, model predictions
├── notebooks              <- Jupyter notebooks.
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures                  <- Generated plots and figures for reports.
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── datamodules              <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   ├── vendor                   <- Third party code that cannot be installed using PIP/Conda
│   │
│   ├── testing_pipeline.py
│   └── training_pipeline.py
│
├── tests                  <- Tests of any kind
│   ├── helpers                  <- A couple of testing utilities
│   ├── shell                    <- Shell/command based tests
│   └── unit                     <- Unit tests
│
├── test.py               <- Run testing
├── train.py              <- Run training
│
├── .coveragerc               <- Configuration for coverage reports of unit tests.
├── .gitignore                <- List of files/folders ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── requirements.txt          <- File for installing python dependencies
├── setup.cfg                 <- Configuration of linters and pytest
├── LICENSE.txt               <- License as chosen on the command-line.
├── pyproject.toml            <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
└── README.md
```

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
