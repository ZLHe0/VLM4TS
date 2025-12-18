"""ViT4TS: Vision Transformer for Time Series Anomaly Detection (Orion-compatible)."""

import os
import sys
import tempfile
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preprocessing.preprocess import preprocess_time_series, draw_windowed_images
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from preprocessing.data_utils import orion_to_internal, intervals_from_indices
from models.clip_vision import CLIP_AD
from models.model_utils import (
    build_memory,
    compute_patch_dissimilarity,
    harmonic_aggregation,
    stitch_anomaly_maps,
    align_anomaly_vector,
    compute_detection_intervals,
)


class ViT4TS:
    """
    Zero-shot vision-based anomaly detector for time series (Orion-compatible).

    This detector:
    1. Converts time series to sliding-window visualizations
    2. Extracts multi-scale CLIP vision embeddings
    3. Compares each window to a normal memory bank
    4. Stitches anomaly maps into a final score vector
    5. Returns anomaly intervals in Orion format

    Parameters
    ----------
    window_size : int, optional
        Size of sliding window in time points (default: 224)
    window_step_ratio : float, optional
        Ratio of window_size to step_size (default: 4.0)
    agg_percent : float, optional
        Top percentage for aggregating anomaly maps (default: 0.25)
    patch_size : int, optional
        Patch size for vision transformer (default: 16, must match model)
    model_name : str, optional
        CLIP vision model name (default: 'ViT-B-16')
        Note: Model name, patch_size, and image_size must be compatible
    device : str, optional
        Device for inference: 'auto', 'cuda', 'cpu' (default: 'auto')
    batch_size : int, optional
        Batch size for inference (default: 20)
    image_size : tuple, optional
        Image size in pixels (height, width) (default: (224, 224), must match model)
    dpi : int, optional
        DPI for image generation (default: 100)
    standardize : bool, optional
        Apply detrending and min-max normalization (default: True)
    alpha : float, optional
        The upper quantile for thresholding (e.g. 0.01 ⇒ top 1%).
    verbose : bool, optional
        Print progress messages (default: True)
    """

    def __init__(
        self,
        window_size: int = 224,
        window_step_ratio: float = 4.0,
        agg_percent: float = 0.25,
        patch_size: int = 16,
        model_name: str = "ViT-B-16",
        device: str = "auto",
        batch_size: int = 20,
        image_size: tuple = (224, 224),
        dpi: int = 100,
        standardize: bool = True,
        alpha: float = 0.01,
        verbose: bool = True,
    ):
        self.window_size = window_size
        self.window_step_ratio = window_step_ratio
        self.agg_percent = agg_percent
        self.patch_size = patch_size
        self.model_name = model_name
        self.batch_size = batch_size
        self.image_size = image_size
        self.dpi = dpi
        self.standardize = standardize
        self.alpha = alpha
        self.verbose = verbose

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if self.verbose:
            print(f"ViT4TS initialized with device: {self.device}")

        # Initialize CLIP model
        self.model = CLIP_AD(model_name=self.model_name, device=self.device)
        self.model.eval()

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in time series data.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'timestamp' and 'value' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with 'start', 'end', 'severity' columns
        """
        # 1. Convert Orion format to internal format
        values, timestamps = orion_to_internal(data)
        T_full = len(values)

        if self.verbose:
            print(f"Processing time series with {T_full} points...")

        # 2. Preprocess time series
        if self.standardize:
            values_proc = preprocess_time_series(values)
        else:
            values_proc = values.astype(float)

        # 3. Create temporary directory for windowed images
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = temp_dir
            base_series_id = "series"

            # 4. Generate windowed images
            step_size = int(self.window_size / self.window_step_ratio)
            time_points = np.arange(len(values_proc))
            n_windows = int((T_full - self.window_size) / step_size) + 1

            plot_params = ("-", 1, "*", 0.1, "black", (0, 1) if self.standardize else None)

            if self.verbose:
                print(
                    f"Generating windowed visualizations (window_size={self.window_size}, step_size={step_size})..."
                )

            success = draw_windowed_images(
                base_series_id=base_series_id,
                save_path=results_dir,
                time_series=values_proc,
                time_points=time_points,
                window_size=self.window_size,
                step_size=step_size,
                override=True,
                save_image=False,
                image_size=self.image_size,
                dpi=self.dpi,
                plot_params=plot_params,
            )

            if not success:
                warnings.warn("Failed to generate windowed images. Returning empty results.")
                return pd.DataFrame(columns=["start", "end", "severity"])

            # 5. Run vision model inference
            if self.verbose:
                print("Running vision model inference...")

            anomaly_scores = self._run_inference(results_dir, base_series_id)

            if anomaly_scores is None or len(anomaly_scores) == 0:
                warnings.warn("No anomaly scores generated. Returning empty results.")
                return pd.DataFrame(columns=["start", "end", "severity"])

        # 6. Align anomaly vector
        aligned_scores = align_anomaly_vector(
            anomaly_scores, T_full, self.window_size, step_size, n_windows
        )

        # 7. Convert scores to intervals
        if self.verbose:
            print("Converting anomaly scores to intervals...")

        interval_indices, _, _ = compute_detection_intervals(
            score_vector=aligned_scores,
            alpha=self.alpha,
        )

        intervals = intervals_from_indices(
            interval_indices=interval_indices,
            timestamps=timestamps,
            scores=aligned_scores,
        )

        if self.verbose:
            print(f"Detected {len(intervals)} anomaly intervals")

        return intervals

    def _run_inference(self, results_dir: str, base_series_id: str) -> Optional[np.ndarray]:
        """
        Run CLIP vision model inference on windowed images.

        Parameters
        ----------
        results_dir : str
            Directory containing windowed image tensors
        base_series_id : str
            Base identifier for the series

        Returns
        -------
        np.ndarray or None
            Final anomaly score vector, shape (T,)
        """
        # Create dataset
        dataset = CLIPTimeSeriesDataset(
            results_dir=results_dir,
            base_series_id=base_series_id,
            sample_size=None,
            no_anomaly=True,  # Zero-shot, no labels
            plot_type="line",  # Always use line plots
        )

        if len(dataset) == 0:
            warnings.warn("No windowed images found in dataset.")
            return None

        # Create DataLoader
        test_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Build normal memory bank
        large_memory_normal, mid_memory_normal, patch_memory_normal = build_memory(
            self.model, test_dataloader, self.patch_size, self.device
        )

        results = {"window_id": [], "anomaly_maps": []}

        with torch.no_grad():
            for items in test_dataloader:
                images = items["img"].to(self.device)
                cls_names = items["cls_name"]
                window_ids = items["window_id"]

                results["window_id"].extend(window_ids.tolist())

                b, _, h, w = images.shape

                # Encode images
                (
                    large_scale_tokens,
                    mid_scale_tokens,
                    patch_tokens,
                    _,
                    large_scale,
                    mid_scale,
                ) = self.model.encode_image(images, self.patch_size)

                # Compute anomaly scores
                m_l_normal = compute_patch_dissimilarity(
                    large_memory_normal, large_scale_tokens, cls_names
                )
                m_m_normal = compute_patch_dissimilarity(
                    mid_memory_normal, mid_scale_tokens, cls_names
                )
                m_p_normal = compute_patch_dissimilarity(
                    patch_memory_normal, patch_tokens, cls_names
                )

                m_l_normal = harmonic_aggregation(
                    (b, h // self.patch_size, w // self.patch_size), m_l_normal, large_scale
                ).to(self.device)
                m_m_normal = harmonic_aggregation(
                    (b, h // self.patch_size, w // self.patch_size), m_m_normal, mid_scale
                ).to(self.device)
                m_p_normal = m_p_normal.reshape(
                    (b, h // self.patch_size, w // self.patch_size)
                ).to(self.device)

                normal_vision_score = torch.nan_to_num(
                    (m_l_normal + m_m_normal + m_p_normal) / 3.0,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                final_score = normal_vision_score.unsqueeze(1)

                # Upsample to original image size
                final_score = F.interpolate(final_score, size=(h, w), mode="bilinear")
                final_score = final_score.squeeze(1)

                cpu_map = final_score.detach().cpu()
                results["anomaly_maps"].append(cpu_map)

        # Concatenate anomaly maps
        results["anomaly_maps"] = torch.cat(results["anomaly_maps"], dim=0).numpy()

        # Sort by window ID
        window_ids = np.array(results["window_id"])
        sorted_indices = np.argsort(window_ids)
        sorted_anomaly_maps = results["anomaly_maps"][sorted_indices]

        # Stitch anomaly maps into final vector
        final_anomaly_vector = stitch_anomaly_maps(
            sorted_anomaly_maps, self.window_step_ratio, self.agg_percent
        )

        return final_anomaly_vector