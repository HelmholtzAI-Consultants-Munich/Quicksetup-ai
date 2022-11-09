## How to use DVC
Data Version Control or DVC is a tool built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code. The full DVC documentation can be found [here](https://dvc.org/doc). In this tutorial, we will show how one can use this tool to keep track of data versions.

### Set up Data Versioning with DVC
This section summarizes [this article from the DVC documentation](https://dvc.org/doc/start/data-management).
>If you chose to create a project using DVC with Quicksetup-ai, then DVC has already been initialized. Otherwise, you need to do it by running
> ```dvc init```
> ```git commit -m "Initialize DVC"```

To start tracking a file or directory, use `dvc add`. For example in this tutorial, we want to keep track of the whole data directory:
```
dvc add data/
```
Then, to track changes with git, run:
```
git add data.dvc .gitignore
git commit -m "Add raw data"
```
### Store and retrieve data stored remotely
In this example, the data is stored using Amazon S3. To set up the remote storage location, run:
```
dvc remote add -d storage s3://mybucket/dvcstore
git add .dvc/config
git commit -m "Configure remote storage"
dvc push
```
To retrieve the data, run:
```
dvc pull
```
### Track changes in your dataset
Suppose, you have changed your dataset inside `data/` and want to run another experiment, to keep track of this, run:
```
dvc add data/
git add data.dvc
git commit -m "updated dataset"

# Using a tag is a useful way to keep track of the dataset version
git tag -a "version_name" -m "updated dataset by ...(describe what you did)"
dvc push
```
Don't forget to update the data version in the training configuration file `configs/train.yaml` (by defaults 'v1'). Set it up to your new version name. That way, you will also keep track of the dataset version in your experiments.

To switch between versions, run:
```
git checkout <version_name>
dvc checkout
```
Here again, update the data version in the training configuration file `configs/train.yaml` and put the one you just switched to.

Congratulation! You can now use the basics of DVC to version your data. Please refer to the [full documentation](https://dvc.org/doc) for more !
