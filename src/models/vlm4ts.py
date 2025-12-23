"""VLM4TS: Two-stage anomaly detection with VLM verification (Orion-compatible)."""

import os
import sys
import json
import base64
import tempfile
import warnings
from typing import Optional, Dict
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

# Add src to path
src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.vit4ts import ViT4TS
from preprocessing.data_utils import orion_to_internal


class VLM4TS:
    """
    Two-stage anomaly detector: ViT4TS screening + VLM verification (Orion-compatible).

    This detector:
    1. Uses ViT4TS to generate high-recall anomaly proposals
    2. Prompts a VLM with the full time series plot and proposals
    3. VLM refines the proposals (removes false positives, adds missed anomalies)
    4. Returns refined anomaly intervals in Orion format

    Parameters
    ----------
    vit4ts_params : dict, optional
        Parameters to pass to ViT4TS. If None, uses defaults.
    alpha : float, optional
        Upper quantile for ViT4TS screening (default: 0.01)
    vlm_model : str, optional
        VLM model name for verification (default: 'gpt-4o')
    api_key : str, optional
        OpenAI API key. If None, loads from environment.
    verbose : bool, optional
        Print progress messages (default: True)
    """

    def __init__(
        self,
        vit4ts_params: Optional[Dict] = None,
        alpha: float = 0.01,
        vlm_model: str = 'gpt-4o',
        api_key: Optional[Dict] = None,
        verbose: bool = True
    ):
        # Initialize ViT4TS
        if vit4ts_params is None:
            vit4ts_params = {}

        # Ensure verbose is passed to ViT4TS
        vit4ts_params['verbose'] = verbose
        vit4ts_params['alpha'] = alpha

        self.vit4ts = ViT4TS(**vit4ts_params)
        self.alpha = alpha
        self.vlm_model = vlm_model
        self.verbose = verbose

        # Initialize OpenAI client
        load_dotenv()
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()  # Uses OPENAI_API_KEY env var

    def detect(self, data: pd.DataFrame, alpha: float = None) -> pd.DataFrame:
        """
        Two-stage anomaly detection with VLM verification.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'timestamp' and 'value' columns
        alpha : float, optional
            Threshold quantile for ViT4TS screening. If None, uses self.alpha.

        Returns
        -------
        pd.DataFrame
            DataFrame with 'start', 'end', 'severity' columns
        """
        # 1. Get ViT4TS proposals
        if self.verbose:
            print("Stage 1: Running ViT4TS screening...")

        # Update alpha temporarily if provided
        original_alpha = self.vit4ts.alpha
        if alpha is not None:
            self.vit4ts.alpha = alpha

        vit_intervals = self.vit4ts.detect(data)
        
        # Restore alpha
        if alpha is not None:
            self.vit4ts.alpha = original_alpha

        if len(vit_intervals) == 0:
            if self.verbose:
                print("ViT4TS found no anomalies. Skipping VLM verification.")
            return vit_intervals

        if self.verbose:
            print(f"ViT4TS detected {len(vit_intervals)} proposal intervals")

        # 2-4. Run verification
        return self.verify_intervals(data, vit_intervals)

    def verify_intervals(self, data: pd.DataFrame, vit_intervals: pd.DataFrame) -> pd.DataFrame:
        """
        Verify and refine anomaly intervals using VLM.

        Parameters
        ----------
        data : pd.DataFrame
            Original time series data
        vit_intervals : pd.DataFrame
            Candidate intervals from screening stage

        Returns
        -------
        pd.DataFrame
            Refined intervals
        """
        if len(vit_intervals) == 0:
            return vit_intervals

        # 2. Generate full time series visualization
        if self.verbose:
            print("Stage 2: Generating visualization for VLM...")

        values, timestamps = orion_to_internal(data)
        # We plot against indices to make it easier for VLM
        img_b64 = self._generate_full_plot(values)

        # 3. Call VLM for verification
        if self.verbose:
            print("Stage 3: Querying VLM for verification...")

        vlm_result = self._query_vlm(img_b64, vit_intervals, timestamps)

        # 4. Convert VLM results to intervals
        if vlm_result is None or 'interval_index' not in vlm_result:
            warnings.warn("VLM verification failed. Returning ViT4TS proposals.")
            return vit_intervals

        interval_indices = vlm_result.get('interval_index', [])
        confidences = vlm_result.get('confidence', [1] * len(interval_indices))
        description = vlm_result.get('abnormal_description', '')

        if len(interval_indices) == 0:
            if self.verbose:
                print("VLM found no anomalies.")
            return pd.DataFrame(columns=['start', 'end', 'severity'])

        # Build intervals DataFrame by mapping indices back to timestamps
        intervals = []
        for i, (start_idx, end_idx) in enumerate(interval_indices):
            # Ensure indices are within bounds
            start_idx = max(0, min(len(timestamps) - 1, int(start_idx)))
            end_idx = max(0, min(len(timestamps) - 1, int(end_idx)))
            
            start_ts = timestamps[start_idx]
            end_ts = timestamps[end_idx]
            
            severity = confidences[i] if i < len(confidences) else 1.0
            intervals.append({
                'start': float(start_ts),
                'end': float(end_ts),
                'severity': float(severity)
            })

        refined_intervals = pd.DataFrame(intervals)

        if self.verbose:
            print(f"VLM refined to {len(refined_intervals)} anomaly intervals")
            if description:
                print(f"\nVLM Analysis:")
                print(f"  {description}")

        return refined_intervals

    def _generate_full_plot(self, values: np.ndarray) -> str:
        """
        Generate a full time series plot against indices and encode as base64.

        Parameters
        ----------
        values : np.ndarray
            Time series values

        Returns
        -------
        str
            Base64-encoded PNG image
        """
        indices = np.arange(len(values))
        dpi = 100
        max_width_px = 1200
        max_height_px = 685

        # Calculate figure size in inches based on desired pixel dimensions
        fig_width = max_width_px / dpi
        fig_height = (max_height_px / 2) / dpi

        # Set font sizes for readability
        title_fontsize = max_width_px // 100
        label_fontsize = max_width_px // 100
        tick_fontsize = max_width_px // 120
        legend_fontsize = max_width_px // 120

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Plot time series
        ax.plot(indices, values, label='Time Series', color='black')
        ax.tick_params(axis='both', labelsize=tick_fontsize)
        
        # Add intermediate ticks for better visual reference
        orig_locs = ax.get_xticks()
        if len(orig_locs) > 1:
            xmin, xmax = ax.get_xlim()
            orig_locs = orig_locs[(orig_locs >= xmin) & (orig_locs <= xmax)]
            mid_locs = (orig_locs[:-1] + orig_locs[1:]) / 2
            mid_locs = mid_locs[(mid_locs > xmin) & (mid_locs < xmax)]
            new_locs = np.sort(np.concatenate([orig_locs, mid_locs]))
            ax.set_xticks(new_locs)
            ax.set_xlim(xmin, xmax)

        ax.set_xlabel("Time", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        ax.set_title("Time Series", fontsize=title_fontsize)
        ax.legend(fontsize=legend_fontsize)
        plt.tight_layout()

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        plt.close()
        buf.seek(0)

        # Encode as base64
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_b64

    def _query_vlm(
        self,
        img_b64: str,
        vit_intervals: pd.DataFrame,
        timestamps: np.ndarray
    ) -> Optional[Dict]:
        """
        Query VLM for anomaly verification.

        Parameters
        ----------
        img_b64 : str
            Base64-encoded time series plot
        vit_intervals : pd.DataFrame
            ViT4TS proposal intervals
        timestamps : np.ndarray
            Timestamp values for index conversion

        Returns
        -------
        dict or None
            VLM response with 'interval_index' and 'confidence' keys
        """
        # Convert timestamp intervals to indices
        detected_indices = []
        for _, row in vit_intervals.iterrows():
            start_ts = row['start']
            end_ts = row['end']
            
            # Find closest indices for timestamps
            start_idx = np.searchsorted(timestamps, start_ts, side='left')
            end_idx = np.searchsorted(timestamps, end_ts, side='right') - 1
            
            # Clamp indices
            start_idx = max(0, min(len(timestamps) - 1, start_idx))
            end_idx = max(0, min(len(timestamps) - 1, end_idx))
            
            detected_indices.append([int(start_idx), int(end_idx)])

        # Construct verification prompt
        base_prompt = """
You are an expert in both time-series analysis and multimodal (vision + language) reasoning.  You will be shown:

1. **A plot of raw time-series data**  
   - X-axis: time step index  
   - Y-axis: signal value over time  

2. **Preliminary “vision-based” anomaly windows**  
   - A list of intervals detected by a coarse, purely visual model  
   - These may include false positives (locally odd but globally normal) and false negatives (statistically or contextually anomalous but visually subtle)

Your goal is to **integrate both sources**—the visual plot and the preliminary windows—and produce a **refined, final anomaly detection** for the entire series.  Specifically:
- **Eliminate** any preliminary windows that look anomalous in isolation but are consistent with the overall trend.  
- **Add** any intervals that the visual model missed but which break temporal continuity or exhibit clear statistical irregularities (spikes, level shifts, abrupt changes).

**Response format**  
Reply **only** with a JSON object containing these fields:

1. `"interval_index"`:  
   An array of `[start, end]` pairs (inclusive indices) for each detected anomaly.  
   ```json
   [[start1, end1], [start2, end2], …]
   ```
   If there are no anomalies, return [].

2. `"confidence"`:
   A parallel array of integers (one per interval) on a 1-3 scale:
   ```json
   [c1, c2, …]
   ```
   - 1 = Low confidence: ambiguous or very subtle deviation (≈50-70% certain)
   - 2 = Medium confidence: clear local irregularity but moderate global uncertainty (≈70-95% certain)
   - 3 = High confidence: strong statistical or contextual evidence of anomaly (>95% certain)
   If no anomalies, return [].

3. `"abnormal_description"`:
   A single paragraph (less than 100 words) summarizing why these intervals are anomalous.

**Important**
- Estimate interval boundaries using the tick marks on the x-axis as precisely as possible.
- The very first segment may appear atypical due to slicing; do not flag it without clear anomaly evidence.  
- Do not include any extra keys or commentary—only the JSON object above.
"""

        vis_line = f"Vision-based model detected intervals (indices): {detected_indices}"
        prompt = base_prompt + "\n" + vis_line

        # Build payload
        payload = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{img_b64}"}
            ],
        }]

        try:
            # Call VLM
            resp = self.client.responses.create(
                model=self.vlm_model,
                input=payload
            )
            raw = resp.output_text.strip()

            # Parse JSON
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                result = json.loads(m.group(0)) if m else {}

            return result

        except Exception as e:
            warnings.warn(f"VLM query failed: {e}")
            return None
