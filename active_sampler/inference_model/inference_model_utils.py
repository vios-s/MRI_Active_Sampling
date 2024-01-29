import torch
import h5py
import numpy as np
from .inference_model_def import build_inference_model, build_inference_optimizer, GradCAMPPModel, FeatureMapPiler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_infer_model(args, optim=False):
    checkpoint = torch.load(args.infer_model_checkpoint)
    infer_args = checkpoint['hyper_parameters']
    infer_model = build_inference_model(infer_args)

    if not optim:
        # No gradients for this model
        for param in infer_model.parameters():
            param.requires_grad = False

    infer_model.load_state_dict(checkpoint['state_dict'])

    start_epoch = checkpoint['epoch']

    if optim:
        optimizer = build_inference_optimizer(infer_model.parameters(), args)
        optimizer.load_state_dict(checkpoint['optimizer_states'])
        del checkpoint
        return infer_model, infer_args, start_epoch, optimizer

    del checkpoint

    if args.use_feature_map:
        final_infer_model = FeatureMapPiler(infer_model.resnet50.to(device), args.feature_map_layer)
    elif args.use_feature_map == False & args.use_grad_campp:
        target_layers = [getattr(infer_model.resnet50, args.feature_map_layer)[-1]]
        final_infer_model = GradCAMPPModel(model=infer_model.resnet50.to(device), target_layers=target_layers)
    else:
        final_infer_model = infer_model
    print('\n')
    print(f'Successfully load checkpoint from {args.infer_model_checkpoint}!!')

    return infer_args, final_infer_model