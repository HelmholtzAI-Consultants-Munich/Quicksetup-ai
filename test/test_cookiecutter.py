import json
import subprocess
import os
import pytest
import shutil

with open("../cookiecutter.json") as f:
    cookie_config = json.load(f)


@pytest.mark.slow
def test_cookiecutter_project_creation():
    os.makedirs("./tmp_test_dir/", exist_ok=True)
    os.chdir("./tmp_test_dir/")
    cmd_ = ["cookiecutter", "--no-input", "https://github.com/HelmholtzAI-Consultants-Munich/ML-Pipeline-Template.git"]
    subprocess.run(cmd_)

    assert os.path.exists(cookie_config["project_name"])
    os.chdir("../")

