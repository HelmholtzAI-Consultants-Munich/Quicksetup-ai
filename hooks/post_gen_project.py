import subprocess
import os


def git_init():
    print("Initializing git")
    subprocess.run(["git", "init"])


def dvc_init():
    print("Initializing dvc")
    subprocess.run(["dvc", "init"])
    subprocess.run(["git", "commit", "-qm", "build: initialize DVC"])


def git_commit():
    print("Staging files")
    subprocess.run(["git", "add", "-A"])
    print("Committing files")
    subprocess.run(["git", "commit", "-aqm", "build: add initial files"])
    subprocess.run(["git", "branch", "-m", "main"])


def delete_license_dir():
    if os.name == 'nt':
        subprocess.run(["rd", "/s", "/q", "licenses/"])
    else:
        subprocess.run(["rm", "-r", "licenses/"])


if __name__ == "__main__":
    delete_license_dir()
    git_init()
    git_commit()
    if "{{cookiecutter.use_dvc}}" == "yes":
        dvc_init()
