import subprocess


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
    subprocess.run(["git", "commit", "-aqm", "buid: add initial files"])
    subprocess.run(["git", "branch", "-m", "main"])


def delete_license_dir():
    subprocess.run(["rm", "-r", "licenses/"])


# def delete_detection_and_segmentation_modules():
#     # TODO detection and segmentation modules


if __name__ == "__main__":
    delete_license_dir()

    # if "{{cookiecutter.include_detection_segmentation_modules}}".lower() == "no":
    #     delete_detection_and_segmentation_modules()

    git_init()
    git_commit()
    
    if "{{cookiecutter.use_dvc}}" == "yes":
        dvc_init()
