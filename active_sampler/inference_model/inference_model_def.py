"""
Part of this code is based on or a copy of the Facebook fastMRI code.

Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
# from argparse import ArgumentParser
import torch
# import numpy as np
import torch.nn as nn
# from .mri_module import MriModule
# import torch.nn.functional as F
# import sys
# from pytorch_grad_cam import GradCAM
from torchvision import models  # Assuming you have a ResNet50 model implementation



class ResNet50Module(nn.Module):
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
        feature_map_layer='layer4',
        dropout_prob=args['dropout_prob']
    )


    return infer_model
