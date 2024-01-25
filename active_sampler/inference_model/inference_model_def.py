"""
Part of this code is based on or a copy of the Facebook fastMRI code.

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
from torchvision import models  # Assuming you have a ResNet50 model implementation


class GradCAMPPModel(nn.Module):
    def __init__(self, model, target_layers):
        super(GradCAMPPModel, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)

    def forward(self, input_tensor, targets=None):
        input_targets = [ClassifierOutputTarget(category) for category in targets.tolist()] if targets is not None else None
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=input_targets)
        model_outputs = self.cam.outputs
        return grayscale_cam, F.softmax(model_outputs, dim=-1)



class ResNet50Module(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,  # Change to the appropriate number of classes
            lr: float = 1e-3,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
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

        self.num_classes = num_classes
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.loss = nn.BCELoss()

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

        # print(self.resnet50)


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
        # Assuming binary classification, use softmax activation for probabilities
        return F.softmax(self.resnet50(image), dim=-1)


def build_inference_optimizer(params, args):
    optim = torch.optim.Adam(
        params,
        lr=args['lr'],
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args['weight_decay'])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=args['lr_step_size'],
        gamma=args['lr_gamma']
    )

    return optim, scheduler



def build_inference_model(args):
    # * model
    infer_model = ResNet50Module(
        num_classes=args['num_classes'],
        lr=args['lr'],
        lr_step_size=args['lr_step_size'],
        lr_gamma=args['lr_gamma'],
        weight_decay=args['weight_decay'],
        dropout_prob=args['dropout_prob']
    )


    return infer_model
