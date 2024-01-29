import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pl_modules import FastMriDataModule, ResNet50Module
from data.mri_data import fetch_dir
from data.masking import create_mask_for_mask_type
from data.transforms import UnetDataTransform

torch.set_float32_matmul_precision('medium')


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = Path("Dataset/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 16

    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "0129_modif_res50_center01_multi_MT_knee_dropout"

    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--mask_type", default="random", type=str, choices=["random", "equispaced"])
    # parser.add_argument('--accelerations', nargs='+', default=[20], type=int,
    #                      help='Ratio of k-space columns to be sampled. If multiple values are '
    #                      'provided, then one of those is chosen uniformly at random for each volume.')
    # parser.add_argument('--center_fractions', nargs='+', default=[0.125], type=float,
    #                      help='Fraction of low-frequency k-space columns to be sampled. Should '
    #                      'have the same length as accelerations')
    parser.add_argument('--accelerations', nargs='+', default=[4, 6, 8, 10, 20], type=int,
                       help='Ratio of k-space columns to be sampled. If multiple values are '
                       'provided, then one of those is chosen uniformly at random for each volume.')
    parser.add_argument('--center_fractions', nargs='+', default=[0.1, 0.1, 0.1, 0.1, 0.1], type=float,
                       help='Fraction of low-frequency k-space columns to be sampled. Should '
                       'have the same length as accelerations')
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser = ResNet50Module.add_model_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        gpus=num_gpus,
        seed=0,
        batch_size=batch_size,
        default_root_dir=default_root_dir,
        max_epochs=35,
        test_path=None
    )

    args = parser.parse_args()

    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=10,
            verbose=True,
            monitor="val_recall",
            mode="max",
        )
    ]

    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = str(ckpt_list[-1])

    print(args.ckpt_path)

    return args


def main():
    args = build_args()
    pl.seed_everything(args.seed)

    # * data
    # masking
    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    # data transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True)
    test_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True)

    # pl data module
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # * model
    model = ResNet50Module(
        num_classes=2,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
        feature_map_layer='layer4',
        dropout_prob=0.2
    )

    # * trainer
    trainer = pl.Trainer(
        logger=True,
        callbacks=args.callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,

    )

    # * run
    if args.mode == 'train':
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')


if __name__ == '__main__':
    main()
