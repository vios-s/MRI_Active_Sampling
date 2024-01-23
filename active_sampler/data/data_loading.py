# Import statements and other constant definitions here
import os
import sys
from math import floor
from pathlib import Path
from typing import NamedTuple, Dict, Any, Union, Optional, Callable, Tuple, Sequence
from warnings import warn
import xml.etree.ElementTree as etree
import pandas as pd
from collections import Counter
import random
import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
sys.path.append('..')
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, normalize, normalize_instance

# Constants
LIST_PATH = '../Dataset/MT_label_data.csv'

def et_query(root, qlist, namespace="http://www.ismrm.org/ISMRMRD"):
    s = "."
    prefix = "ismrmrd_namespace"
    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)

class FastMRIRawDataSample(NamedTuple):
    fname: Path
    slice_ind: int
    label: int
    metadata: Dict[str, Any]

class SliceDataset(Dataset):
    def __init__(self, root, list_path, data_partition, transform=None, sample_rate=None):
        self.transform = transform
        self.recons_key = "reconstruction_esc"
        self.data_partition = data_partition
        self.raw_samples = []
        label_list = self.read_sample_label(list_path)
        files = list(Path(root).iterdir())
        for fname in sorted(files):
            metadata, num_slices, image_size = self._retrieve_metadata(fname)
            new_raw_samples = []
            if 350 <= image_size[1] <= 400:
                if sample_rate < 1.0:
                    half_slice = 0.5 * num_slices
                    start = floor(half_slice - 0.5 * sample_rate * num_slices)
                    end = floor(half_slice + 0.5 * sample_rate * num_slices)
                    for slice_ind in range(start, end):
                        label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)
                        new_raw_samples.append(raw_sample)
                else:
                    for slice_ind in range(num_slices):
                        label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                        raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)
                        new_raw_samples.append(raw_sample)
                self.raw_samples += new_raw_samples

        label_distribution = self.count_label_distribution()
        over_minor_samples = self.oversample_minority(label_distribution)
        self.raw_samples += over_minor_samples
        random.shuffle(self.raw_samples)
        print("\n")
        print(label_distribution)

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i):
        fname, dataslice, label, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        return self.transform(kspace, mask, target, attrs, fname.name, dataslice, label)

    def _retrieve_metadata(self, fname):
        with h5py.File(fname, 'r') as hf:
            et_root = etree.fromstring(hf['ismrmrd_header'][()])
            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),
                int(et_query(et_root, enc + ["y"])),
                int(et_query(et_root, enc + ["z"])),
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),
                int(et_query(et_root, rec + ["y"])),
                int(et_query(et_root, rec + ["z"])),
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1

            padding_left = enc_size[1] // 2 - enc_limits_center
            padding_right = padding_left + enc_limits_max
            num_slices = hf["kspace"].shape[0]
            image_size = [hf["kspace"].shape[1], hf["kspace"].shape[2]]
            metadata = {
                "padding_left": padding_left,
                "padding_right": padding_right,
                "encoding_size": enc_size,
                "recon_size": recon_size,
                **hf.attrs
            }

        return metadata, num_slices, image_size

    def read_sample_label(self, list_path):
        label_df = pd.read_csv(list_path, header=0, names=['file', 'slice', 'label'])
        return label_df

    def find_label(self, label_list, target_fname, target_slice):
        filtered_rows = label_list.loc[(label_list['file'] == target_fname) & (label_list['slice'] == target_slice)]
        if not filtered_rows.empty:
            return int(filtered_rows['label'].values[0])
        else:
            return int(0)

    def remove_h5_extension(self, fname):
        return os.path.splitext(fname.name)[0]

    def count_label_distribution(self):
        labels = [sample.label for sample in self.raw_samples]
        label_distribution = Counter(labels)
        return label_distribution

    def oversample_minority(self, label_dist):
        oversampled_raw_samples = []
        if self.data_partition == 'train':
            max_samples = max(label_dist.values())
            for label, count in label_dist.items():
                oversample_factor = max_samples // count
                if oversample_factor > 1:
                    minority_samples = [sample for sample in self.raw_samples if sample.label == label]
                    oversampled_raw_samples.extend(minority_samples * (oversample_factor - 1))
        return oversampled_raw_samples

class DataTransform:
    def __init__(self, mask_func, resolution = [640, 356], use_seed=False):
        self.mask_func = mask_func
        self.resolution = resolution
        self.use_seed = use_seed

    def __call__(self, kspace, mask, target, attrs, fname, dataslice, label):
        tensor_kspace = to_tensor(kspace)
        tensor_image = ifft2c(tensor_kspace)
        crop_size = self.resolution
        cropped_clean_image = complex_center_crop(tensor_image, crop_size)
        target_tensor_kspace = fft2c(cropped_clean_image)
        if mask is None and self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask_, _ = apply_mask(target_tensor_kspace, self.mask_func, seed=seed)

        target = complex_abs(ifft2c(target_tensor_kspace))
        zf = complex_abs(ifft2c(masked_kspace))

        zf, zf_mean, zf_std = normalize_instance(zf, eps=1e-11)
        zf = zf.clamp(-6, 6)
        target, gt_mean, gt_std = normalize_instance(target, eps=1e-11)
        target = target.clamp(-6, 6)

        target = target.expand(3, -1, -1)
        zf = zf.expand(3, -1, -1)

        slice_info = {'slice': dataslice, 'label': label}
        return target_tensor_kspace, masked_kspace, mask_, zf, target, gt_mean, gt_std, fname, slice_info

class MaskFunc:
    def __init__(self, center_fractions, accelerations):
        self.center_fractions = center_fractions
        self.accelerations = accelerations

    def __call__(self, shape, seed=None):
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')

        np.random.seed(seed)
        num_cols = shape[-2]

        choice = np.random.randint(0, len(self.accelerations))
        center_fraction = self.center_fractions[choice]
        acceleration = self.accelerations[choice]

        num_low_freqs = int(round(num_cols * center_fraction))
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = np.random.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

        return mask, num_low_freqs

def apply_mask(data, mask_func, seed=None, padding=None):
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:])
    mask, num_low_freqs = mask_func(shape, seed=seed)
    if padding is not None:
        mask[..., :padding[0], :] = 0
        mask[..., padding[1]:, :] = 0

    masked_data = data * mask + 0.0
    return masked_data, mask, num_low_freqs

def create_fastmri_dataset(args, partition):
    if partition == 'train':
        path = args.data_path / 'singlecoil_train'
        use_seed = False
    elif partition == 'val':
        path = args.data_path / 'singlecoil_val'
        use_seed = True
    elif partition == 'test':
        path = args.data_path / 'singlecoil_test'
        use_seed = True
    else:
        raise ValueError(f"partition should be in ['train', 'val', 'test'], not {partition}")

    dataset = SliceDataset(
        root=path,
        list_path=LIST_PATH,
        data_partition=partition,
        transform=DataTransform(MaskFunc(args.center_fractions, args.accelerations), args.resolution, use_seed=use_seed),
        sample_rate=args.sample_rate,
    )

    print(f'{partition.capitalize()} slices: {len(dataset)}')

    return dataset

def create_data_loader(args, partition, shuffle=False, display=False):
    dataset = create_fastmri_dataset(args, partition)

    if partition.lower() == 'train':
        batch_size = args.batch_size
        if not shuffle:
            warn("Currently not shuffling training data! Pass shuffle=True to create_data_loader() to shuffle.")
    elif partition.lower() in ['val', 'test']:
        batch_size = args.val_batch_size
        if display:
            dataset = [dataset[i] for i in range(0, len(dataset), len(dataset) // 16)]
    else:
        raise ValueError(f"'partition' should be in ('train', 'val', 'test'), not {partition}")

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Define your command-line arguments
    parser.add_argument('--data_path', type=Path, default=Path('../Dataset/test_dataset'))
    parser.add_argument('--center_fractions', type=float, nargs='+', default=[0.08, 0.04])
    parser.add_argument('--accelerations', type=int, nargs='+', default=[4, 8])
    parser.add_argument('--resolution', type=list, default=[320,320])
    parser.add_argument('--sample_rate', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Use args to create DataLoader or any other functionality
    train_loader = create_data_loader(args, 'train', shuffle=True, display=False)
    val_loader = create_data_loader(args, 'val', shuffle=False, display=True)
    test_loader = create_data_loader(args, 'test', shuffle=False, display=True)


# Training loop
for batch_idx, batch_data in enumerate(train_loader):
    kspace, masked_kspace, mask, zf, target, gt_mean, gt_std, fname, slice_info = batch_data

    # Your training code goes here...

    # Print batch information
    print(np.shape(kspace),np.shape(mask))
    print(f"Batch {batch_idx + 1}/{len(train_loader)}")
    print(f"File Name: {fname}")
    print(f"Slice Info: {slice_info}")
