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

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Two-stage anomaly detection with VLM verification.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'timestamp' and 'value' columns

        Returns
        -------
        pd.DataFrame
            DataFrame with 'start', 'end', 'severity' columns
        """
        # 1. Get ViT4TS proposals
        if self.verbose:
            print("Stage 1: Running ViT4TS screening...")

        vit_intervals = self.vit4ts.detect(data)

        if len(vit_intervals) == 0:
            if self.verbose:
                print("ViT4TS found no anomalies. Skipping VLM verification.")
            return vit_intervals

        if self.verbose:
            print(f"ViT4TS detected {len(vit_intervals)} proposal intervals")

        # 2. Generate full time series visualization
        if self.verbose:
            print("Stage 2: Generating visualization for VLM...")

        values, timestamps = orion_to_internal(data)
        img_b64 = self._generate_full_plot(values, timestamps)

        # 3. Call VLM for verification
        if self.verbose:
            print("Stage 3: Querying VLM for verification...")

        vlm_result = self._query_vlm(img_b64, vit_intervals, timestamps)

        # 4. Convert VLM results to intervals
        if vlm_result is None or 'interval_timestamp' not in vlm_result:
            warnings.warn("VLM verification failed. Returning ViT4TS proposals.")
            return vit_intervals

        interval_timestamps = vlm_result.get('interval_timestamp', [])
        confidences = vlm_result.get('confidence', [1] * len(interval_timestamps))
        description = vlm_result.get('abnormal_description', '')

        if len(interval_timestamps) == 0:
            if self.verbose:
                print("VLM found no anomalies.")
            return pd.DataFrame(columns=['start', 'end', 'severity'])

        # Build intervals DataFrame directly from timestamps (no conversion needed)
        intervals = []
        for i, (start_ts, end_ts) in enumerate(interval_timestamps):
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

    def _generate_full_plot(self, values: np.ndarray, timestamps: np.ndarray) -> str:
        """
        Generate a full time series plot and encode as base64.

        Parameters
        ----------
        values : np.ndarray
            Time series values
        timestamps : np.ndarray
            Timestamp values

        Returns
        -------
        str
            Base64-encoded PNG image
        """
        fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
        ax.plot(timestamps, values, linewidth=1, color='black')
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Time Series', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
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
            VLM response with 'interval_timestamp' and 'confidence' keys
        """
        # Use timestamp intervals directly (matching x-axis on plot)
        detected_intervals = []
        for _, row in vit_intervals.iterrows():
            detected_intervals.append([float(row['start']), float(row['end'])])

        # Construct prompt
        base_prompt = """
You are an expert in both time-series analysis and multimodal (vision + language) reasoning. You will be shown:

1. **A plot of raw time-series data**
   - X-axis: timestamp
   - Y-axis: signal value over time

2. **Preliminary "vision-based" anomaly windows**
   - A list of intervals detected by a coarse, purely visual model
   - These may include false positives (locally odd but globally normal) and false negatives (statistically or contextually anomalous but visually subtle)

Your goal is to **integrate both sources**—the visual plot and the preliminary windows—and produce a **refined, final anomaly detection** for the entire series. Specifically:
- **Eliminate** any preliminary windows that look anomalous in isolation but are consistent with the overall trend.
- **Add** any intervals that the visual model missed but which break temporal continuity or exhibit clear statistical irregularities (spikes, level shifts, abrupt changes).

**Response format**
Reply **only** with a JSON object containing these fields:

{
    "interval_timestamp": [[start1, end1], [start2, end2], ...],
    "confidence": [c1, c2, ...],
    "abnormal_description": "..."
}

where:
- "interval_timestamp": an array of [start, end] pairs using the timestamp values from the x-axis. If no anomalies, return [].
- "confidence": a parallel array of integers (1-3 scale). If no anomalies, return [].
- "abnormal_description": a single paragraph (less than 100 words) summarizing why these intervals are anomalous.

**Confidence scale:**
- 1 = Low confidence: ambiguous or very subtle deviation (≈50-70% certain)
- 2 = Medium confidence: clear local irregularity but moderate global uncertainty (≈70-95% certain)
- 3 = High confidence: strong statistical or contextual evidence of anomaly (>95% certain)

**Important:**
- Use the x-axis tick marks (timestamp values) to specify interval boundaries as precisely as possible.
- The very first segment may appear atypical due to slicing; do not flag it without clear anomaly evidence.
- Do not include any extra keys or commentary—only the JSON object above.
"""

        vis_line = f"Vision-based model detected intervals (timestamps): {detected_intervals}"
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
