import logging
from glob import glob
from typing import Optional

import cv2
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.ops import masks_to_boxes
from torch.nn import functional as F

log = logging.getLogger(__name__)


class DetectionAndSegmentationDataModule(LightningDataModule):
    """ Datamodule for detection and segmentation tasks.
    # TODO
    """
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_class=2
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # List of transformations for data augmentation
        transforms_list = [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(
            p=0.5), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), shear=(5, 5), scale=(1.1, 1.1))]
        self.transforms = transforms.Compose(transforms_list)

        self.train_image_paths = sorted(glob(f"{data_dir}/train/images/*", recursive=True))
        self.val_image_paths = sorted(glob(f"{data_dir}/val/images/*", recursive=True))
        self.test_image_paths = sorted(glob(f"{data_dir}/test/images/*", recursive=True))

        self.train_segmentation_paths = sorted(
            glob(f"{data_dir}/train/segmentation/*", recursive=True))
        self.test_segmentation_paths = sorted(
            glob(f"{data_dir}/val/segmentation/*", recursive=True))
        self.test_segmentation_paths = sorted(
            glob(f"{data_dir}/test/segmentation/*", recursive=True))
        self.num_class = num_class

    @property
    def num_classes(self) -> int:
        return self.num_class

    def setup(self, stage: Optional[str] = None):
        self.data_train = DetectionAndSegmentationDataset2D(self.train_image_paths, self.train_segmentation_paths)
        self.data_val = DetectionAndSegmentationDataset2D(self.test_image_paths, self.test_segmentation_paths)
        self.data_test = DetectionAndSegmentationDataset2D(self.test_image_paths, self.test_segmentation_paths)

        log.info(f'Number of training images: {len(self.data_train)} | '
                 f'validation images: {len(self.data_val)} | '
                 f'test images: {len(self.data_test)}')
        log.info(f'Dimension of training image: {self.data_train[0][0].shape} | '
                 f'Dimension of validation image: {self.data_val[0][0].shape} | '
                 f'Dimension of test image: {self.data_test[0][0].shape}')

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=lambda x: list(zip(*x)))

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x: list(zip(*x)))

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x: list(zip(*x)))


class DetectionAndSegmentationDataset2D(Dataset):
    """# TODO"""
    def __init__(self, img_path_list, mask_list, transforms=None):
        self.img_path_list = img_path_list
        self.mask_list = mask_list
        self.transforms = transforms
        self.num_img_channels = 6

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path_list[idx])
        mask = cv2.imread(self.img_path_list[idx])

        if self.transforms is not None:
            input = torch.vstack([img, mask])
            res = self.transforms(input)
            img, mask = res[:self.num_img_channels, :, :], res[self.num_img_channels:, :, :]

        target = self._segmentation_to_bbox(mask)
        return img, target

    def __len__(self):
        return len(self.img_path_list)

    @staticmethod
    def _segmentation_to_bbox(mask):
        """Convert segmentation maps to bounding-boxes
        #TODO
        """
        mask = F.convert_image_dtype(mask, dtype=torch.float)

        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set of boolean masks.
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)

        # there is only one class
        labels = torch.ones((masks.shape[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        return target
