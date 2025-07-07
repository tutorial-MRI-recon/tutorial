import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob

class MRIReconstructionDataset(Dataset):
    def __init__(self, data_root, split='train'):
        """
        Args:
            data_root (str): Root directory with 'train', 'val', 'test' folders.
            split (str): Which split to load.
        """
        self.data_dir = os.path.join(data_root, split)
        self.file_list = sorted(glob(os.path.join(self.data_dir, '*.npz')))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_path = self.file_list[idx]
        sample = np.load(npz_path)

        # Load arrays
        combined_full = sample['combined_full']      # (320, 320), complex64
        combined_us = sample['combined_us']      # (320, 320), complex64
        us_coil = sample['us_coil']                  # (20, 320, 320), complex64
        sensitivity = sample['sensitivity']          # (20, 320, 320), complex64
        mask = sample['mask']                        # (320, 320), float32

        # Normalize by max |us_coil|
        us_coil_abs = np.abs(us_coil)
        norm_factor = np.max(us_coil_abs).astype(np.float32)
        norm_factor = max(norm_factor, 1e-6)  # avoid divide-by-zero

        # Apply normalization
        combined_us /= norm_factor
        combined_full /= norm_factor
        us_coil /= norm_factor
        #sensitivity /= norm_factor

        # Convert to torch tensors
        combined_full = self._complex_to_tensor(combined_full)    # (2, 320, 320)
        combined_us = self._complex_to_tensor(combined_us)    # (2, 320, 320)
        us_coil = self._complex_to_tensor(us_coil)                # (20, 2, 320, 320)
        sensitivity = self._complex_to_tensor(sensitivity)        # (20, 2, 320, 320)
        mask = torch.from_numpy(mask).float()                     # (320, 320)

        return {
            'combined_full': combined_full,
            'combined_us': combined_us,
            'us_coil': us_coil,
            'sensitivity': sensitivity,
            'mask': mask,
            'norm_factor': torch.tensor(norm_factor),
            'filename': os.path.basename(npz_path)
        }

    def _complex_to_tensor(self, data):
        """
        Converts complex-valued numpy array → torch tensor with shape (..., 2)
        and moves the complex axis to front.
        """
        data = np.stack((data.real, data.imag), axis=-1)  # (..., 2)
        data = np.moveaxis(data, -1, 0)                   # (2, ...)
        return torch.from_numpy(data).float()
