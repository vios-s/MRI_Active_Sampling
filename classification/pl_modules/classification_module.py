from argparse import ArgumentParser
import torch
import numpy as np
import torch.nn as nn
from .mri_module import MriModule
import torch.nn.functional as F
import sys
from pytorch_grad_cam import GradCAM

sys.path.append('../')
from torchvision import models  # Assuming you have a ResNet50 model implementation



class ResNet50Module(MriModule):
    def __init__(
            self,
            num_classes: int = 2,  # Change to the appropriate number of classes
            lr: float = 1e-3,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            feature_map_layer: str = 'layer4',
            dropout_prob: float = 0.5,
            **kwargs
    ):
        """_summary_

        Args:
            in_chans (int, optional): _description_. Defaults to 1.
            out_chans (int, optional): _description_. Defaults to 1.
            chans (int, optional): _description_. Defaults to 32.
            num_classes (int, optional): _description_. Defaults to 4.
            lr (float, optional): _description_. Defaults to 1e-3.
            lr_step_size (int, optional): _description_. Defaults to 40.
            lr_gamma (float, optional): _description_. Defaults to 0.1.
            weight_decay (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__(**kwargs)

        self.save_hyperparameters()
        self.num_classes = num_classes
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.loss = nn.BCELoss()
        self.feature_map = torch.zeros([2,2])

        # Load pre-trained ResNet50 model
        self.resnet50 = models.resnet50(weights='DEFAULT')
        self.append_dropout(self.resnet50, rate=dropout_prob)

        # Modify the output layer to match the number of output channels/classes
        in_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),  # Add dropout
            nn.Linear(in_features // 2, num_classes)
        )

        # desired_layer = getattr(self.resnet50, feature_map_layer)
        print(self.resnet50)
        # self.gradcam = GradCAM(model=self.resnet50, target_layers=desired_layer)

    def append_dropout(self, module, rate):
        for name, child_module in module.named_children():
            if len(list(child_module.children())) > 0:
                self.append_dropout(child_module, rate)
            if isinstance(child_module, nn.ReLU) and not isinstance(child_module, nn.Dropout2d):
                # Create a new ReLU layer and add dropout to it
                new_module = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout2d(p=rate)  # Set inplace to False
                )
                setattr(module, name, new_module)

    def forward(self, image):
        # Assuming binary classification, use sigmoid activation for probabilities
        return self.resnet50(image)

    def training_step(self, batch, batch_idx):

        output = self(batch.image)
        one_hot_label = F.one_hot(batch.label, num_classes=self.num_classes).float()
        loss = self.loss(torch.sigmoid(output), one_hot_label)
        acc = (torch.sigmoid(output).argmax(dim=-1) == batch.label).float().mean()
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc)
        self.log("train_loss", loss.detach())

        return loss

    def log_image(self, name, image):
        self.logger.experiment.add_image(name, image, self.global_step)

    def validation_step(self, batch, batch_idx, one_hot_label=None):
        output = self(batch.image)
        one_hot_label = F.one_hot(batch.label, num_classes=self.num_classes).float()
        # Access feature maps
        feature_map = self.feature_map
        if batch.image.ndim == 4:
            mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            val_logs = {
                "image": batch.image.squeeze() * std + mean,
                "mask": batch.mask,
                "batch_idx": batch_idx,
                "fname": batch.fname,
                "slice_num": batch.slice_num,
                "max_value": batch.max_value,
                "meta_data": batch.metadata,
                "feature_map": feature_map,
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output),one_hot_label).item()
            }
        else:
            mean = batch.mean.unsqueeze(1).unsqueeze(2)
            std = batch.std.unsqueeze(1).unsqueeze(2)

            val_logs = {
                "image": batch.image.squeeze() * std + mean,
                "mask": batch.mask,
                "batch_idx": batch_idx,
                "fname": batch.fname,
                "slice_num": batch.slice_num,
                "max_value": batch.max_value,
                "meta_data": batch.metadata,
                "feature_map": feature_map,
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output),one_hot_label).item()
            }

        for k in ("mask", "batch_idx", "fname", "slice_num", "max_value", "meta_data", "feature_map", "predictions", "labels", "loss"):
            assert k in val_logs, f"Missing {k} in val_logs"

        # * pick an image to log
        if self.val_log_indices is None:
            self.val_log_indices = list(np.random.permutation(len(self.trainer.val_dataloaders))[: self.num_log_images])

            # * log the image to tensorboard
        if isinstance(val_logs["batch_idx"], int):
            batch_indices = [val_logs["batch_idx"]]
        else:
            batch_indices = val_logs["batch_idx"]

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.val_log_indices:
                label = val_logs["labels"][i]
                pred = val_logs["predictions"][i]
                key = f"val_image_{batch_idx}_label_{label}_pred_{pred}"
                image = val_logs["image"][i][0].unsqueeze(0)
                # feature_map = val_logs["feature_map"][i].unsqueeze(0)
                image = image / image.max()
                # feature_map = feature_map / feature_map.max()

                self.log_image(f"{key}/image", image)
                # self.log_image(f"{key}/feature_map", feature_map)

        output_log = {
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output), one_hot_label)
                 }

        self.validation_step_outputs.append(output_log)

        return output_log

    def test_step(self, batch, batch_idx):
        output = self(batch.image)
        feature_map = self.feature_map
        one_hot_label = F.one_hot(batch.label, num_classes=self.num_classes).float()

        if batch.image.ndim == 4:
            mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            test_logs = {
                "image": batch.image.squeeze() * std + mean,
                "mask": batch.mask,
                "batch_idx": batch_idx,
                "fname": batch.fname,
                "slice_num": batch.slice_num,
                "max_value": batch.max_value,
                "meta_data": batch.metadata,
                "feature_map": feature_map,
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output),one_hot_label)
            }
        else:
            mean = batch.mean.unsqueeze(1).unsqueeze(2)
            std = batch.std.unsqueeze(1).unsqueeze(2)

            test_logs = {
                "image": batch.image.squeeze() * std + mean,
                "mask": batch.mask,
                "batch_idx": batch_idx,
                "fname": batch.fname,
                "slice_num": batch.slice_num,
                "max_value": batch.max_value,
                "meta_data": batch.metadata,
                "feature_map": feature_map,
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output),one_hot_label)
            }

        for k in (
        "mask", "batch_idx", "fname", "slice_num", "max_value", "meta_data", "feature_map", "predictions", "labels",
        "loss"):
            assert k in test_logs, f"Missing {k} in test_logs"

        # * pick an image to log
        if self.test_log_indices is None:
            self.test_log_indices = list(
                np.random.permutation(len(self.trainer.test_dataloaders))[: self.num_log_images])

        if isinstance(test_logs["batch_idx"], int):
            batch_indices = [test_logs["batch_idx"]]
        else:
            batch_indices = test_logs["batch_idx"]

        for i, batch_idx in enumerate(batch_indices):
            if batch_idx in self.test_log_indices:
                label = test_logs["labels"][i]
                pred = test_logs["predictions"][i]
                key = f"test_image_{batch_idx}_label_{label}_pred_{pred}"
                # torch.set_grad_enabled(True)  # required for grad cam
                # cam = self._initialize_grad_cam(False)
                # grayscale_cam = cam(input_tensor=batch.image[i].unsqueeze(0).clone(), targets=label)
                image = test_logs["image"][i][0].unsqueeze(0)
                # feature_map = grayscale_cam.unsqueeze(0)
                image = image / image.max()
                # feature_map = feature_map / feature_map.max()

                self.log_image(f"{key}/image", image)
                # self.log_image(f"{key}/feature_map", feature_map)

        output_log = {
                "predictions": torch.sigmoid(output),
                "labels": batch.label,
                "loss": self.loss(torch.sigmoid(output), one_hot_label)
                 }

        self.test_step_outputs.append(output_log)
        return output_log


    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--lr", type=float, default=5e-5)
        parser.add_argument("--lr_step_size", type=int, default=40)
        parser.add_argument("--lr_gamma", type=float, default=0.1)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--feature_map_layer", type=str, default='layer4')
        parser.add_argument("--dropout_prob", type=float, default=0.2)
        return parser



