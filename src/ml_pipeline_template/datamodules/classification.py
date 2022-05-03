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

from ml_pipeline_template import utils

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