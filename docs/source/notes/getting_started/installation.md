# Installation

### TL;DR:
```bash
conda create -y -n venv_cookie python=3.9 && conda activate venv_cookie
pip install cookiecutter
cookiecutter https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai.git
```



## Using Cookiecutter
1. Create and activate your environment:
    ```bash
    conda create -y -n venv_cookie python=3.9 && conda activate venv_cookie
    ```

2. Install [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/) and [DVC](https://dvc.org/) (optional) in 
   your 
   environment:
    ```bash
    pip install cookiecutter dvc
    ```
3. Create your own project using this template via cookiecutter:
    ```bash
    cookiecutter https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai.git
    ```