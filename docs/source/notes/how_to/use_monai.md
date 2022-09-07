# MedNIST classification
MedNIST classification using MONAI and MLPT

## MONAI and MLPT
1. Install [MONAI](https://docs.monai.io/en/stable/installation.html) in your environment:
   ```bash
   pip install monai
   ```
2. Create your own project using this template via cookiecutter
(skip if project has already been created via cookiecutter installation step):
   ```bash
   pip install cookiecutter
   cookiecutter https://github.com/HelmholtzAI-Consultants-Munich/Quicksetup-ai.git
   ```
Check the [installation page](https://ml-pipeline-template.readthedocs.io/en/latest/notes/getting_started/installation.html) if you are having issues creating a project using this template via
cookiecutter.




## Setup the data
1. To setup the data, we define two classes:
    - `ClassificationTaskDataModule` (inherit from `pytorch_lightning.LightningDataModule`)
    - `MedNISTDataset` (inherits from `torch.utils.data.Dataset`)
    - We saved it as `Quicksetup-ai/src/quicksetup_ai/datamodules/classification.py`
2. Below are the contents of this file containing the methods we need to define.
    - `prepare_data`
        - download MedNIST tar file and saved it under `data/MedNIST/raw`
        - extract contents of downloaded file and save it under `data/MedNIST/processed`
    - `setup`
        - define transforms
        - split dataset into train|test|validation set


```python
# Quicksetup-ai/src/quicksetup_ai/datamodules/classification.py file
import numpy as np
import pytorch_lightning as pl
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset

from monai.transforms import (
    AddChannel,
    Compose,
    LoadImage,
    RandFlip,
    EnsureType,
    ScaleIntensity
)

from monai.apps import download_url, extractall
import os
import glob

from quicksetup_ai import utils

log = utils.get_logger(__name__)


class ClassificationTaskDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, num_classes=6, batch_size: int = 64,
                 num_workers: int = 0, pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.img_all = []
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.save_hyperparameters(logger=False)
        self.label_all = None

    def prepare_data(self):
        # Download data and save under `data/raw/` directory.
        data_dir_raw = os.path.join(self.data_dir, "MedNIST/raw/")
        data_dir_processed = os.path.join(self.data_dir, "MedNIST/processed/")
        resource = "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz"
        md5 = "0bc7306e7427e00ad1c5526a6677552d"
        compressed_file = os.path.join(data_dir_raw, "MedNIST.tar.gz")
        if not os.path.exists(compressed_file):
            download_url(url=resource, filepath=compressed_file, hash_val=md5)
        if not os.path.exists(data_dir_processed):
            extractall(filepath=compressed_file, output_dir=data_dir_processed)
        new_data_dir = os.path.join(data_dir_processed, "**", "*.jpeg")
        self.img_all = np.array(glob.glob(new_data_dir, recursive=True))
        log.info(f"Total images: {len(self.img_all)}")
        self.label_all = np.array([i.split("/")[-2] for i in self.img_all])
        le = preprocessing.LabelEncoder()
        self.label_all = le.fit_transform(self.label_all)
        log.info(f"Total labels: {len(self.label_all)}")

    def setup(self, stage=None):
        # Define transforms
        train_transforms = Compose(
            [LoadImage(image_only=True), AddChannel(), ScaleIntensity(),
             RandFlip(spatial_axis=0, prob=0.5),
             EnsureType()]
        )
        val_transforms = Compose(
            [LoadImage(image_only=True), AddChannel(), ScaleIntensity(), EnsureType()]
        )
        # Split dataset in to train, val, test sets.
        num_samples = len(self.img_all)
        tr_split = int(num_samples * .8)
        val_split = int(num_samples * .9)
        all_indices = np.arange(num_samples)
        np.random.shuffle(all_indices)
        train_indices = all_indices[:tr_split]
        val_indices = all_indices[tr_split:val_split]
        test_indices = all_indices[val_split:]
        train_subjects, train_labels = self.img_all[train_indices], self.label_all[
            train_indices]
        val_subjects, val_labels = self.img_all[val_indices], self.label_all[
            val_indices]
        test_subjects, test_labels = self.img_all[test_indices], self.label_all[
            test_indices]

        self.train_set = MedNISTDataset(train_subjects, train_labels, train_transforms)
        self.val_set = MedNISTDataset(val_subjects, val_labels, val_transforms)
        self.test_set = MedNISTDataset(test_subjects, test_labels, val_transforms)

        log.info(f"Number of trainset: {len(self.train_set)}")
        log.info(f"Number of testset: {len(self.test_set)}")
        log.info(f"Number of valset: {len(self.val_set)}")

    def train_dataloader(self):
        return DataLoader(self.train_set, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          drop_last=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, drop_last=True)


class MedNISTDataset(Dataset):
    def __init__(self, img_paths, label_paths, transforms):
        self.img_paths = img_paths
        self.label_paths = label_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return self.transforms(self.img_paths[index]), self.label_paths[index]

```

## Setup the config for data
1. This `yaml` file will be used to initialize `ClassificationTaskDataModule`.
2. We create a `yaml` file under `.Quicksetup-ai/configs/datamodule/` and named it `mednist.yaml`
3. Below are the contents of the corresponding `yaml` file for the custom `ClassificationTaskDataModule`.
   - File saved in `.Quicksetup-ai/configs/datamodule/mednist.yaml`

```yaml
# Quicksetup-ai/configs/datamodule/mednist.yaml file
_target_: quicksetup_ai.datamodules.classification.ClassificationTaskDataModule

# Any argument in the `ClassificationTaskDataModule` can be modified using this file
data_dir: ${data_dir}
num_classes: 6
batch_size: 256
num_workers: 0
pin_memory: False
```

## Setup the model
1. We will use the `DenseNet169` model from MONAI.
    - First, write your own class which inherits from `pytorch_lightning.LightningModule`
    - Save it under `Quicksetup-ai/src/quicksetup_ai/models/`.
    - Below are the contents of the example file (`Quicksetup-ai/src/quicksetup_ai/models/classification.py`) we will use in this tutorial.

```python
# Quicksetup-ai/src/quicksetup_ai/models/classification.py file
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from quicksetup_ai import utils

log = utils.get_logger(__name__)


class ClassificationTaskModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.val_acc(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
```

## Setup the config for model
1. We create a `yaml` file named `mednist.yaml` inside `.Quicksetup-ai/configs/datamodule/`
2. Below are the contents of the corresponding `yaml` file for `ClassificationTaskModule`.
    - Saved in `Quicksetup-ai/configs/datamodule/densenet.yaml`.

```yaml
# Quicksetup-ai/configs/datamodule/densenet.yaml file
_target_: quicksetup_ai.models.custom_model.MedNISTModel

lr: 0.001
weight_decay: 0.0005
net:
  _target_: monai.networks.nets.DenseNet169
  spatial_dims: 2
  in_channels: 1
  out_channels: 6
```

## Model training
CPU-based training:
```bash
python scripts/train.py datamodule=mednist model=densenet logger=tensorboard
```

GPU-based training:
```bash
python scripts/train.py datamodule=mednist model=densenet trainer.gpus=1 logger=tensorboard
```


## Model testing
1. The best model is saved as `logs/experiments/runs/default/{date}/checkpoints/epoch_{num}.ckpt`.
   - for example: `logs/experiments/runs/default/2022-05-03_18-52-31/checkpoints/epoch_004.ckpt`

CPU-based testing:
```bash
CKPT_PATH=logs/experiments/runs/default/2022-05-03_18-52-31/checkpoints/epoch_004.ckpt
python scripts/test.py datamodule=mednist model=densenet ckpt_path=$CKPT_PATH
```

GPU-based testing:
```bash
CKPT_PATH=logs/experiments/runs/default/2022-05-03_18-52-31/checkpoints/epoch_004.ckpt
python scripts/test.py datamodule=mednist model=densenet trainer.gpus=1 ckpt_path=$CKPT_PATH
```
