import subprocess
import yaml

def save_initial_data():
    version_name = 'v1'
    data_path = input('Data path:')
    print("Saving initial data")
    subprocess.run(["dvc", "add", "data_path"])
    subprocess.run(["git", "add", ".gitignore", data_path + ".dvc"])
    subprocess.run(["git", "commit", "-m", "data: Save initial data"])
    subprocess.run(["git", "tag", '-a', "'" + version_name + "'", '-m', 'data: Save initial data' ])
    subprocess.run(["dvc", "push"])


def save_data_version():
    version_name = input('Version name:')
    data_path = input('Data path:')
    print("Saving data version: ", version_name)
    subprocess.run(["dvc", "commit", "data_path"])
    subprocess.run(["git", "add", ".gitignore", data_path + ".dvc"])
    subprocess.run(["git", "commit", "-m", "data: Create data version: " + version_name])
    subprocess.run(["git", "tag", '-a', "'" + version_name + "'", '-m', "data: Create data version: " + version_name])
    subprocess.run(["dvc", "push"])
    change_yaml_entry("data_version", version_name, "configs/train.yaml")


def switch_to_data_version():
    version_name = input('Version name:')
    print("Switching to data version: ", version_name)
    subprocess.run(["git", "checkout", version_name])
    subprocess.run(["dvc", "checkout"])
    change_yaml_entry("data_version", version_name, "configs/train.yaml")


def change_yaml_entry(entry, value, config_file):
    with open(config_file) as f:
        list_configs = yaml.safe_load(f)

    for line in list_configs:
        if line["name"] == entry:
            line["value"] = value

    with open(config_file, "w") as f:
        yaml.dump(list_configs, f)