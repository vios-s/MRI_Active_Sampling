import torch
import numpy as np
import sys
from typing import Union, Optional, Tuple, Sequence, NamedTuple

from .masking import MaskFunc

sys.path.append('..')
from utils.coils import rss
from utils.complex import complex_abs
from utils.fft import fft2c, ifft2c
from utils.transform_utils import to_tensor, complex_center_crop, center_crop, normalize, normalize_instance

def apply_mask(data: torch.Tensor, mask_func: MaskFunc, offset: Optional[int]=None, 
                seed: Optional[Union[int, Tuple[int, ...]]]=None, padding: Optional[Sequence[int]]=None):
    
    shape = (1,) * len(data.shape[:-3]) + tuple(data.shape[-3:]) # (1, 1, 640, 372, 2)
    mask, num_low_freqs = mask_func(shape, offset, seed)
    if padding is not None:
        mask[..., :padding[0], :] = 0
        mask[..., padding[1]:, :] = 0
        
    # * add 0.0 removes the sign of the zeros    
    masked_data = data * mask + 0.0 
    
    return masked_data, mask, num_low_freqs


class UnetSample(NamedTuple):
    
    image: torch.Tensor

    mask: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    fname: str
    slice_num: int
    label: list
    max_value: float
    metadata: dict



class UnetDataTransform:
    def __init__(self, challenge:str, mask_func: Optional[MaskFunc]=None, use_seed: bool=True, noise_lvl: float=0.0) -> None:
        assert challenge in ('singlecoil', 'multicoil'), f'Challenge should be either singlecoil or multicoil, got {challenge}'
        
        self.mask_func = mask_func
        self.challenge = challenge
        self.use_seed = use_seed
        self.noise_lvl = noise_lvl
        
    def __call__(self, kspace: np.ndarray, mask: np.ndarray, target: np.ndarray, attrs: dict, fname: str, slice_num: int, label: list):
        """

        Args:
            kspace (np.ndarray): Input k-space of shape (num_coils, rows, cols) for multi-coil data
                or (rows, cols) for single coil data.
            mask (np.ndarray): Mask from the test dataset.
            target (np.ndarray): Target image.
            attrs (Dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name.
            slice_num (int): slice index.
            label (int): classification label
            
        Returns: A tuple containing, 
            image: zero-filled input image, 
            output: the reconstruction
            target: target, 
            mean: the mean used for normalization, 
            std: the standard deviations used for normalization, 
            fname: the filename, 
            slice_num: and the slice number.
            label: classification label
        """
        
        # !1 convert to tensor
        tensor_kspace = to_tensor(kspace)

        # !1.5 add noise
        if self.noise_lvl != 0.0:
            noisy_tensor_kspace = tensor_kspace + torch.randn_like(tensor_kspace) * self.noise_lvl + 0.0
        else:
            noisy_tensor_kspace = tensor_kspace
        
        #! 2 convert to image
        tensor_image = ifft2c(tensor_kspace)
        noisy_image = ifft2c(noisy_tensor_kspace)
        crop_size = [640, 356]
        cropped_clean_image = complex_center_crop(tensor_image, crop_size)
        cropped_noisy_image = complex_center_crop(noisy_image, crop_size)

        #! 4 apply fft2 to get related k-space
        target_tensor_kspace = fft2c(cropped_clean_image)
        noisy_tensor_kspace = fft2c(cropped_noisy_image)
        
        #! 5 apply mask
        if mask is None and self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            noisy_masked_kspace, mask_, _ = apply_mask(noisy_tensor_kspace, self.mask_func, seed=seed)
        else:
            mask_ = mask
            noisy_masked_kspace = noisy_tensor_kspace * mask_ + 0.0

            
        #! 6 get input images
        input_noisy_image = complex_abs(ifft2c(noisy_masked_kspace))
        target_image = ifft2c(target_tensor_kspace)
        
        #! 7 normalize input image
        input_noisy_image, mean, std = normalize_instance(input_noisy_image, eps=1e-11)


        #! 6 normalize target
        if target is not None:
            target_torch = normalize(to_tensor(target), mean, std, eps=1e-11)
            target_torch = target_torch.clamp(-6, 6)
        else:
            target_torch = complex_abs(normalize(target_image, mean, std, eps=1e-11))
            target_torch = target_torch.clamp(-6, 6)

        max_value = attrs["max"] if "max" in attrs else max(target_torch)
        input_noisy_image = input_noisy_image.expand(3, -1, -1)

        return UnetSample(
            image=input_noisy_image,
            mask=mask_,
            target=target_torch,
            mean=mean,
            std=std,
            fname=fname,
            slice_num=slice_num,
            label=label,
            max_value=max_value,
            metadata=attrs
        )
