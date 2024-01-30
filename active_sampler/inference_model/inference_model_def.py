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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GradCAMPPModel(nn.Module):
    def __init__(self, model, target_layers):
        super(GradCAMPPModel, self).__init__()
        self.model = model
        self.target_layers = target_layers
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers)

    def forward(self, input_tensor, targets=None):
        input_targets = [ClassifierOutputTarget(category) for category in targets.tolist()] if targets is not None else None
        grayscale_cam = torch.tensor(self.cam(input_tensor=input_tensor.requires_grad_(), targets=input_targets)).unsqueeze(1)
        model_outputs = self.cam.outputs
        return grayscale_cam, F.softmax(model_outputs, dim=-1)


class FeatureMapPiler(nn.Module):
    def __init__(self, model, target_layer_names):
        super(FeatureMapPiler, self).__init__()
        self.model = model
        self.target_layer_names = target_layer_names
        self.feature_maps = []

        # Register hooks to the target layers
        self.hook_handlers = []
        self.register_hooks()

    def register_hooks(self):
        def hook(name):
            def hook_fn(module, input, output):
                self.feature_maps.append((name, output))
            return hook_fn

        for name in self.target_layer_names:
            layer = self.find_layer_by_name(self.model, name)
            if layer is not None:
                hook_handler = layer.register_forward_hook(hook(name))
                self.hook_handlers.append(hook_handler)
            else:
                raise ValueError(f"Target layer '{name}' not found in the model.")

    def find_layer_by_name(self, model, target_name):
        for name, module in model.named_modules():
            if name == target_name:
                return module
        return None

    def remove_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()

    def forward(self, x):
        with torch.no_grad():
            self.feature_maps = []  # Clear previous feature maps
            model_outputs = self.model(x)
            self.remove_hooks()

        # Concatenate and resize feature maps
        resized_feature_maps = []
        target_size = x.size()[2:]  # Size of the input image
        for _, feature_map in self.feature_maps:
            resized_feature_map = F.interpolate(feature_map.mean(dim=1).unsqueeze(1), size=target_size, mode='bilinear', align_corners=False)
            print(resized_feature_map)
            resized_feature_maps.append(resized_feature_map)

        # Concatenate along the channel dimension
        concatenated_feature_maps = torch.cat(resized_feature_maps, dim=1)

        return concatenated_feature_maps, F.softmax(model_outputs, dim=-1)





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
        with torch.no_grad():
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
