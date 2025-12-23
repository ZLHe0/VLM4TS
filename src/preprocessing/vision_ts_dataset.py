"""PyTorch Dataset for CLIP time series inference."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class CLIPTimeSeriesDataset(Dataset):
    """
    Dataset for loading windowed time series images for CLIP inference.

    Parameters
    ----------
    results_dir : str
        Directory containing the aggregated image .npy file.
        Expected file naming: <base_series_id>_<plot_type>_img.npy
    base_series_id : str
        The base identifier for the time series (e.g., "series").
    plot_type : str
        The plot type used (e.g., "line"). Used to build the filename.
    sample_size : int, optional
        If provided and less than the number of windows, only a random subset of windows is used.
    """

    def __init__(self, results_dir, base_series_id, plot_type='line', sample_size=None, no_anomaly=True):
        self.results_dir = results_dir
        self.base_series_id = base_series_id
        self.plot_type = plot_type

        # Build filename
        self.img_file = os.path.join(results_dir, f"{base_series_id}_{plot_type}_img.npy")

        # Load the aggregated image tensor
        if not os.path.exists(self.img_file):
            raise FileNotFoundError(f"Image file {self.img_file} not found.")
        self.imgs = np.load(self.img_file)  # Shape: [num_windows, C, H, W]
        print(f"Loaded {self.img_file} with shape {self.imgs.shape}.")

        self.num_windows = self.imgs.shape[0]

        # Optionally sample a subset of windows
        self.indices = list(range(self.num_windows))
        if sample_size is not None and sample_size < self.num_windows:
            import random
            self.indices = random.sample(self.indices, sample_size)
            self.indices.sort()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual window index
        win_idx = self.indices[idx]

        # Load image tensor for this window
        img = self.imgs[win_idx]  # shape: [C, H, W]
        img_tensor = torch.from_numpy(img).float()

        # Return only the fields needed by vit4ts.py
        sample = {
            'img': img_tensor,          # Tensor [C, H, W]
            'cls_name': self.base_series_id,  # string
            'window_id': win_idx        # int
        }
        return sample
