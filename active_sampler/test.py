import pathlib

import numpy as np
import argparse
from data.data_loading import create_data_loader
from inference_model.inference_model_utils import load_infer_model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define your command-line arguments
    parser.add_argument('--data_path', type=pathlib.Path, default='./Dataset/test_dataset')
    parser.add_argument('--center_fractions', type=float, nargs='+', default=[0.08, 0.04])
    parser.add_argument('--accelerations', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--resolution', type=list, default=356)
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--infer_model_checkpoint', type=str, default='../classification/log/modif_res50_single20_MT_knee_dropout/checkpoints/epoch=39-step=49400.ckpt')
    parser.add_argument('--use_feature_map', type=bool, default=False)
    parser.add_argument('--feature_map_layer', type=str, default='layer4')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Use args to create DataLoader and loading model
    train_loader = create_data_loader(args, 'train', shuffle=True, display=False)
    val_loader = create_data_loader(args, 'val', shuffle=False, display=True)
    test_loader = create_data_loader(args, 'test', shuffle=False, display=True)
    train_data_range_dict = create_data_range_dict(args, train_loader)
    infer_args, infer_model = load_infer_model(args)

# Training loop
for batch_idx, batch_data in enumerate(train_loader):
    kspace, masked_kspace, mask, zf, target, gt_mean, gt_std, fname, slice_info = batch_data
    label = slice_info['label']

    if args.use_feature_map:
        feature_map, outputs = infer_model(zf, label)
        image_input = feature_map
    else:
        outputs = infer_model(zf)
        image_input = zf[:, 0, :, :]

    print(np.shape(image_input), outputs)

