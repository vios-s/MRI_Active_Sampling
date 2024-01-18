import os
import torch
import pickle
import h5py
from math import floor
import yaml
import numpy as np
import xml.etree.ElementTree as etree
from pathlib import Path
from warnings import warn
from typing import NamedTuple, Dict, Any, Union, Optional, Callable, Tuple, Sequence
import pandas as pd
from collections import Counter
import random


def et_query(
        root: etree.Element,
        qlist: Sequence[str],
        namespace: str = "http://www.ismrm.org/ISMRMRD",
) -> str:
    """
    ElementTree query function.

    This can be used to query an xml document via ElementTree. It uses qlist
    for nested queries.

    Args:
        root: Root of the xml to search through.
        qlist: A list of strings for nested searches, e.g. ["Encoding",
            "matrixSize"]
        namespace: Optional; xml namespace to prepend query.

    Returns:
        The retrieved data as a string.
    """
    s = "."
    prefix = "ismrmrd_namespace"

    ns = {prefix: namespace}

    for el in qlist:
        s = s + f"//{prefix}:{el}"

    value = root.find(s, ns)
    if value is None:
        raise RuntimeError("Element not found")

    return str(value.text)


def fetch_dir(key: str, data_config_file="fastmri_dirs.yaml") -> Path:
    """_summary_

    Args:
        key (_type_): key to retrieve path from data_config_file. Expected to be in
            ("knee_path", "brain_path", "log_path").
        data_config_file (str, optional): Default path config file to fetch path
            from. Defaults to "fastmri_dirs.yaml".

    Returns:
        Path: _description_
    """

    data_config_file = Path(data_config_file)
    if not data_config_file.is_file():
        default_config = {
            "knee_path": "Dataset/fastMRI/",
            "brain_path": "Dataset/FastMRI/",
            "log_path": "logs/",
        }

        with open(data_config_file, "w") as f:
            yaml.dump(default_config, f)

        data_dir = default_config[key]

        warn(
            f"Path config at {data_config_file.resolve()} does not exist. "
            "A template has been created for you. "
            "Please enter the directory paths for your system to have defaults."
        )
    else:
        with open(data_config_file, "r") as f:
            data_dir = yaml.safe_load(f)[key]

    return Path(data_dir)


class FastMRIRawDataSample(NamedTuple):
    """Basic data type for fastMRI raw data.

    Elements:
        fname: Path for each h5 file, Path
        slice_ind: slice index, int
        metadata: metadata for each volume, Dict
    """
    fname: Path
    slice_ind: int
    label: int
    metadata: Dict[str, Any]


class SliceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: Union[str, Path, os.PathLike],
            list_path: Union[str, Path, os.PathLike],
            challenge: str,
            data_partition: str,
            transform: Optional[Callable] = None,
            use_dataset_cache: bool = False,
            sample_rate: Optional[float] = None,
            volume_sample_rate: Optional[float] = None,
            dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
            num_cols: Optional[Tuple[int]] = None,
            raw_sample_filter: Optional[Callable] = None,

    ):
        """A PyTorch Dataset for loading fastMRI data.

        Args:
            root (Union[str, Path, os.PathLike]): Path to the dataset directory (e.g. knee_path/train, brain_path/test, etc.)
            list_path (Union[str, Path, os.PathLike]): Path to csv file that contains label and metadata
            challenge (str): "singlecoil" or "multicoil".
            transform (Optional[Callable], optional): A callable object that takes a raw data sample as input and returns a transformed version. Defaults to None.
                The transform function should take ['kspace', 'target', 'attributes', 'filename', and 'slice'] as inputs.
                'target' maybe None if the dataset is a test dataset.
            use_dataset_cache (bool, optional): Whether to cache dataset metadata. Useful for large dataset. Defaults to False.
            sample_rate (Optional[float], optional): A float between 0 and 1. Defaults to 1 if None is given. Either sample_rate or volume_sample_rate should be set.
            volume_sample_rate (Optional[float], optional): _description_. The same as sample_rate. Defaults to 1 if None is given.
            dataset_cache_file (Union[str, Path, os.PathLike], optional): A file in which to cache dataset information for faster load times. Defaults to "Dataset/fastMRI/dataset_cache.pkl".
            num_cols (Optional[Tuple[int]], optional): If provided, only slices with the desired number of columns will be considered. Defaults to None.
            raw_sample_filter (Optional[Callable], optional): A callable object that takes an raw_sample metadata as input and return a boolean indicating whether the raw_sample should be included in the dataset. Defaults to None.
            data_partition (str): A callable object that takes an raw_sample metadata as input and return a boolean indicating whether the raw_sample should be includ
        """
        # * Choose between singlecoil and multicoil
        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError(f"Challenge should be singlecoil or multicoil, got {challenge}.")
        # * Check if sample_rate and volume_sample_rate are both set
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError("Only one of sample_rate and volume_sample_rate should be set.")

        self.dataset_cache_file = Path(dataset_cache_file)
        self.transform = transform
        # * set different target for singlecoil or multicoil
        self.recons_key = ("reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss")
        self.data_partition = data_partition
        # * The list of files
        self.raw_samples = []
        # * if there is a filter
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # * set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # Read sample label
        label_list = self.read_sample_label(list_path)
        # check cache
        # if yes, use the metadata from cache
        # if not, regenerate the metadata

        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices, image_size = self._retrieve_metadata(fname)
                new_raw_samples = []
                if (image_size[1] >= 350) & (image_size[1] <= 400):
                    if sample_rate < 1.0:
                        half_slice = 0.5 * num_slices
                        start = floor(half_slice - 0.5 * sample_rate * num_slices)
                        end = floor(half_slice + 0.5 * sample_rate * num_slices)
                        for slice_ind in range(start, end):
                            label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                            raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)
                            if self.raw_sample_filter(raw_sample):
                                new_raw_samples.append(raw_sample)
                    else:
                        for slice_ind in range(num_slices):
                            label = self.find_label(label_list, self.remove_h5_extension(fname), slice_ind)
                            raw_sample = FastMRIRawDataSample(fname, slice_ind, label, metadata)

                            if self.raw_sample_filter(raw_sample):
                                new_raw_samples.append(raw_sample)

                    self.raw_samples += new_raw_samples

            # Calculate label distribution
            label_distribution = self.count_label_distribution()
            over_minor_samples = self.oversample_minority(label_distribution)   # Add oversampled samples to the dataset
            self.raw_samples += over_minor_samples
            random.shuffle(self.raw_samples)

        print("\n")
        print(label_distribution)




    def count_label_distribution(self):
        """
        Count the distribution of labels in the raw_samples.

        Returns:
            Counter: A Counter object containing label counts.
        """
        labels = [sample.label for sample in self.raw_samples]
        label_distribution = Counter(labels)
        return label_distribution

    def oversample_minority(self, label_dist):
        oversampled_raw_samples = []
        if self.data_partition == 'train':
            # Oversample the minority group
            max_samples = max(label_dist.values())
            for label, count in label_dist.items():
                oversample_factor = max_samples // count
                if oversample_factor > 1:
                    # Oversample only the minority group
                    minority_samples = [sample for sample in self.raw_samples if sample.label == label]
                    oversampled_raw_samples.extend(minority_samples * (oversample_factor - 1))
        return oversampled_raw_samples


    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i):
        # * get data from raw_samples and feed into transform
        fname, dataslice, label, metadata = self.raw_samples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice]
            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None
            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice, label)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, dataslice, label)

        return sample

    def _retrieve_metadata(self, fname):
        """_summary_

        Args:
            fname (_type_): 

        Returns:
            _type_: _description_
        """
        with h5py.File(fname, 'r') as hf:
            et_root = etree.fromstring(hf['ismrmrd_header'][()])

            enc = ["encoding", "encodedSpace", "matrixSize"]
            enc_size = (
                int(et_query(et_root, enc + ["x"])),  # 640
                int(et_query(et_root, enc + ["y"])),  # 372
                int(et_query(et_root, enc + ["z"])),  # 1
            )
            rec = ["encoding", "reconSpace", "matrixSize"]
            recon_size = (
                int(et_query(et_root, rec + ["x"])),  # 320
                int(et_query(et_root, rec + ["y"])),  # 320
                int(et_query(et_root, rec + ["z"])),  # 1
            )

            lims = ["encoding", "encodingLimits", "kspace_encoding_step_1"]
            enc_limits_center = int(et_query(et_root, lims + ["center"]))  # 167
            enc_limits_max = int(et_query(et_root, lims + ["maximum"])) + 1  # 334

            padding_left = enc_size[1] // 2 - enc_limits_center  # 372 // 2 - 167 = 19
            padding_right = padding_left + enc_limits_max  # 19 + 334 = 353

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
        # Assuming the CSV file has columns: 'file', 'slice', 'label'
        label_df = pd.read_csv(list_path, header=0, names=['file', 'slice', 'label'])
        return label_df

    def find_label(self, label_list, target_fname, target_slice):
        # Assuming label_list is a DataFrame with columns: 'file', 'slice', 'label'
        filtered_rows = label_list.loc[(label_list['file'] == target_fname) & (label_list['slice'] == target_slice)]
        if not filtered_rows.empty:
            return int(filtered_rows['label'].values[0])
        else:
            return int(0)

    def remove_h5_extension(self, fname):
        return os.path.splitext(fname.name)[0]
