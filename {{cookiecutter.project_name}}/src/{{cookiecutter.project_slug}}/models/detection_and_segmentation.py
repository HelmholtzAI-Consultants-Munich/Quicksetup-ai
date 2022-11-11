from typing import Any, List

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pytorch_lightning import LightningModule


class DetectionAndSegmentation(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.0005,
        momentum=0.9,
        weight_decay: float = 0.0005,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.net(x=(images, targets))
        loss_mask = loss_dict['loss_mask']
        loss_box_reg = loss_dict['loss_box_reg']
        loss_classifier = loss_dict['loss_classifier']
        loss = sum(loss for loss in loss_dict.values())
        return loss, loss_mask, loss_box_reg, loss_classifier

    def training_step(self, batch: Any, batch_idx: int):
        self.net = self.net.train()
        loss, loss_mask, loss_box_reg, loss_classifier = self.step(batch)
        return {"loss": loss, "loss_mask": loss_mask, "loss_box_reg": loss_box_reg, "loss_classifier": loss_classifier}

    def training_epoch_end(self, outputs: List[Any]):
        print('Epoch', self.current_epoch)
        loss = outputs[-1]["loss"]
        loss_mask = outputs[-1]["loss_mask"]
        loss_box = outputs[-1]["loss_box_reg"]
        loss_cls = outputs[-1]["loss_classifier"]

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_mask", loss_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_box", loss_box, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            self.net = self.net.train()
            loss, loss_mask, loss_box_reg, loss_classifier = self.step(batch)
        return {"loss": loss, "loss_mask": loss_mask, "loss_box_reg": loss_box_reg, "loss_classifier": loss_classifier}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = outputs[-1]["loss"]
        loss_mask = outputs[-1]["loss_mask"]
        loss_box = outputs[-1]["loss_box_reg"]
        loss_cls = outputs[-1]["loss_classifier"]

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_mask", loss_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_box", loss_box, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_cls", loss_cls, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            self.net = self.net.train()
            loss, loss_mask, loss_box_reg, loss_classifier = self.step(batch)
        return {"loss": loss, "loss_mask": loss_mask, "loss_box_reg": loss_box_reg, "loss_classifier": loss_classifier}

    def test_epoch_end(self, outputs: List[Any]):
        loss = outputs[-1]["loss"]
        loss_mask = outputs[-1]["loss_mask"]
        loss_box = outputs[-1]["loss_box_reg"]
        loss_cls = outputs[-1]["loss_classifier"]

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_mask", loss_mask, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_box", loss_box, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/loss_cls", loss_cls, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )


class MRCNN(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.model = self._get_model_instance_segmentation()

    def _get_model_instance_segmentation(self):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           self.num_classes)
        return model

    def forward(self, x):
        if self.model.training:
            x = self.model(x[0], x[1])
        else:
            x = self.model(x)
        return x
