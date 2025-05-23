import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random

class CLIPTimeSeriesDataset(Dataset):
    def __init__(self, results_dir, base_series_id, plot_type, sample_size=None, random_seed=None, no_anomaly=False):
        """
        Parameters
        ----------
        results_dir : str
            Directory containing the aggregated image and mask .npy files.
            Expected file naming:
              - For images: <base_series_id>_<plot_type>_img.npy
              - For masks:  <base_series_id>_<plot_type>_mask.npy (if available)
        base_series_id : str
            The base identifier for the time series (e.g., "C-2").
        plot_type : str
            The plot type used (e.g., "line", "gaf", "rp"). This will be used to build the filename.
        sample_size : int, optional
            If provided and less than the number of windows, only a random subset of windows is used.
        random_seed : int, optional
            Seed for reproducibility when sampling.
        no_anomaly : bool, optional
            If True, anomaly masks are not expected and a default zero mask is returned.
        """
        self.results_dir = results_dir
        self.base_series_id = base_series_id
        self.plot_type = plot_type
        self.no_anomaly = no_anomaly
        
        # Build filenames.
        self.img_file = os.path.join(results_dir, f"{base_series_id}_{plot_type}_img.npy")
        self.mask_file = os.path.join(results_dir, f"{base_series_id}_{plot_type}_mask.npy")
        
        # Load the aggregated image tensor.
        if not os.path.exists(self.img_file):
            raise FileNotFoundError(f"Image file {self.img_file} not found.")
        self.imgs = np.load(self.img_file)  # Shape: [num_windows, C, H, W]
        print(f"Loaded {self.img_file} with shape {self.imgs.shape}.")
        
        self.num_windows = self.imgs.shape[0]
        
        # Load mask if available and no_anomaly is False.
        if not no_anomaly and os.path.exists(self.mask_file):
            self.masks = np.load(self.mask_file)  # Shape: [num_windows, H, W]
        else:
            self.masks = None
            print(f"Mask file {self.mask_file} not found or no_anomaly is True. Using default zero mask.")
        
        # Optionally sample a subset of windows.
        self.indices = list(range(self.num_windows))
        if sample_size is not None and sample_size < self.num_windows:
            if random_seed is not None:
                random.seed(random_seed)
            self.indices = random.sample(self.indices, sample_size)
            self.indices.sort()  # Optionally sort them.
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get the actual window index.
        win_idx = self.indices[idx]
        
        # Load image tensor for this window.
        img = self.imgs[win_idx]  # shape: [C, H, W]
        img_tensor = torch.from_numpy(img).float()
        
        # Load corresponding mask if available; otherwise create a zero mask.
        if self.masks is not None:
            mask = self.masks[win_idx]  # shape: [H, W]
        else:
            # Create a default mask of zeros with shape matching the image spatial dimensions.
            _, H, W = img_tensor.shape
            mask = np.zeros((H, W), dtype=np.float32)
        mask_tensor = torch.from_numpy(mask).float()
        
        # Determine anomaly label: if any pixel in mask > 0, anomaly=1.
        anomaly_flag = 1 if mask_tensor.sum() > 0 else 0
        anomaly_tensor = torch.tensor(anomaly_flag, dtype=torch.long)
        
        # The class name is simply the base_series_id.
        cls_name = self.base_series_id
        
        # Set window_id to the window index.
        window_id = win_idx
        
        # Generate a simple text prompt based on the anomaly flag.
        text_prompt = "Anomaly detected" if anomaly_flag == 1 else "Normal operation"
        
        sample = {
            'img': img_tensor,          # Tensor [C, H, W]
            'cls_name': cls_name,       # e.g., "C-2"
            'window_id': window_id,     # integer window index
            'img_mask': mask_tensor,    # Tensor [H, W]
            'anomaly': anomaly_tensor,  # 0 or 1
            'text_prompt': text_prompt  # string prompt
        }
        return sample