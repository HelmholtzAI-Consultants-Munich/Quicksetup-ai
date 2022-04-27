# How to set up a different dataset
In this Tutorial, we show how one can quickly edit the template to use another dataset. In this example, the new dataset is CIFAR10.

## Create a new DataModule
Under `src/ml_pipeline_template/datamodules/`, we create a new file called `cifar_10_datamodule.py` with the content of `mnist_datamodule.py`. Then, we edit the necessary parts, namely in `prepare_data`, we download the CIFAR10 dataset instead of MNIST. We also modify the default splits and transforms.
```
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

class CIFAR10DataModule(LightningDataModule):
    """
    Example of LightningDataModule for CIFAR10 dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] =  (40_000, 5_000, 15_000),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        
        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        @property
        def num_classes(self) -> int:
            return 10

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        CIFAR10(self.hparams.data_dir, train=True, download=True)
        CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = CIFAR10(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = CIFAR10(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

```
## Create the configuration file
In `configs/datamodule/`, we create a new file called `cifar_10.yaml`. We provide the correct DataModule class and the parameters we want for the experiment. Here's the content of `cifar_10.yaml`.

```
_target_: ml_pipeline_template.datamodules.cifar_10_datamodule.CIFAR10DataModule

data_dir: ${data_dir} # data_dir is specified in config.yaml
batch_size: 64
train_val_test_split: [40_000, 5_000, 15_000]
num_workers: 0
pin_memory: False
```
## Edit the main train/test configurations 
The final step is to configure `configs/train.yaml` and `configs/test.yaml`.

First, we edit the training configuration. Under `defaults`, we set `datamodule` to `cifar_10.yaml` to make it use the new datamodule.
> Note that we also need to create a new model configuration file: `configs/model/cifar_10.yaml` where we adjust the parameters to suit the new dataset. We set `model` in `train.yaml` under `defaults` to `cifar_10.yaml`.

Finally, we edit the test configuration: We also set `datamodule` and `model` to `cifar_10.yaml` under `defaults`. Then, we provide the `ckpt_path` to the model we want to test.

Congratulations! You can now use other datasets in the template.
