import subprocess


def git_init():
    print("Initializing git")
    subprocess.run(["git", "init", "-b", "main"])


def dvc_init():
    print("Initializing dvc")
    subprocess.run(["dvc", "init"])
    subprocess.run(["git", "commit", "-qm", "build: initialize DVC"])


def git_commit():
    print("Staging files")
    subprocess.run(["git", "add", "-A"])
    print("Committing files")
    subprocess.run(["git", "commit", "-aqm", "buid: add initial files"])


def delete_license_dir():
    subprocess.run(["rm", "-r", "licenses/"])


if __name__ == "__main__":
    delete_license_dir()
    git_init()
    git_commit()
    if "{{cookiecutter.use_dvc}}" == "yes":
        dvc_init()

