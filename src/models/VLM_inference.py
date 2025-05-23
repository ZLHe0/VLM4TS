import os
import ast
import json
import base64
import argparse

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

import sys
src_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, src_path)
os.chdir(src_path)
print("system path:", os.getcwd())

def load_file_list(data_dir, dataset_name, file_list_arg):
    if file_list_arg:
        return file_list_arg
    meta_path = os.path.join(data_dir, "datasets_multivariate.csv")
    df = pd.read_csv(meta_path, header=None, names=["dataset","files"])
    row = df[df["dataset"].str.strip() == dataset_name]
    if row.empty:
        raise ValueError(f"Dataset {dataset_name} not found in {meta_path}")
    return ast.literal_eval(row.iloc[0]["files"])

def load_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main(
    data_dir: str,
    dataset_name: str,
    file_list: list[str]|None,
    alpha_list: list[float],
):
    # 1) build list of files
    files = load_file_list(data_dir, dataset_name, file_list)

    # 2) init OpenAI client
    load_dotenv()
    client = OpenAI()

    # 3) fixed part of prompt
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

If there are no anomalies, return [].
	2.	"confidence":
A parallel array of integers (one per interval) on a 1-3 scale:

[c1, c2, …]

	•	1 = Low confidence: ambiguous or very subtle deviation (≈50-70% certain)
	•	2 = Medium confidence: clear local irregularity but moderate global uncertainty (≈70-95% certain)
	•	3 = High confidence: strong statistical or contextual evidence of anomaly (>95% certain)
If no anomalies, return [].

Important
	•	Estimate interval boundaries using the tick marks on the x-axis as precisely as possible.
	•	The very first segment may appear atypical due to slicing; do not flag it without clear anomaly evidence.  
	•	Do not include any extra keys or commentary—only the JSON object above.
"""

    # 4) Loop over each signal
    for file_name in files:
        results_dir = os.path.join("..","results", dataset_name, file_name)
        os.makedirs(results_dir, exist_ok=True)

        # 4a) Load the raw-plot image and base64-encode it
        img_fname = os.path.join(results_dir, f"{file_name}_timeseries_raw.png")
        if not os.path.exists(img_fname):
            print(f"[WARN] missing image {img_fname}, skipping.")
            continue
        img_b64 = load_base64(img_fname)

        # 4b) Load the *vision-based* detections JSON once per file
        det_metrics_path = os.path.join(
            results_dir,
            f"{file_name}_vision_metrics_windowwise.json"
        )
        if not os.path.exists(det_metrics_path):
            print(f"[WARN] missing vision detections {det_metrics_path}, skipping.")
            continue
        det_data = json.load(open(det_metrics_path))

        # 4c) **NEW**: for each alpha, ask the VLM and store its JSON reply
        vlm_results = {}
        for alpha in alpha_list:
            alpha_str = str(alpha)

            # i) pull out the coarse intervals for this alpha
            vis = det_data.get(alpha_str, {})
            detected = vis.get("detected_intervals", [])

            # ii) append that into your prompt
            vis_line = f"Vision-based model detected intervals (alpha={alpha_str}): {detected}"
            prompt   = base_prompt + "\n" + vis_line

            # iii) build the multimodal payload
            payload = [{
                "role":"user",
                "content":[
                    {"type":"input_text",  "text": prompt},
                    {"type":"input_image", "image_url": f"data:image/png;base64,{img_b64}"}
                ],
            }]

            # iv) call the VLM
            resp = client.responses.create(
                model="gpt-4o-2024-08-06",
                input=payload
            )
            raw = resp.output_text.strip()

            # v) parse the JSON (with fallback if needed)
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                import re
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                result = json.loads(m.group(0)) if m else {}

            # vi) stash it under this alpha
            try:
                vlm_results[alpha_str] = result
                print(f"  alpha={alpha_str} → refined intervals: {result.get('interval_index',[])}")
            except Exception as e:
                print(f"[ERROR] failed to parse VLM response for alpha={alpha_str}: {e}")

        # 4d) after looping all alphas, save one JSON mapping alpha → result
        out_path = os.path.join(results_dir, f"{file_name}_gpt_detections.json")
        with open(out_path, "w") as f:
            json.dump(vlm_results, f, indent=2)
        print(f"✔ saved all-alpha results to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Run VLM-based detection on Yahoo A1 time-series plots"
    )
    p.add_argument("--data_dir",        type=str, default="../data/raw/")
    p.add_argument("--dataset_name",    type=str, required=True)
    p.add_argument("--file_list",       type=str, nargs="+", default=None)
    p.add_argument("--alpha_list",      type=float, nargs="+", required=True,
                   help="List of alphas to run VLM refinement over (e.g. 0.1 0.01 0.001)")
    args = p.parse_args()

    main(
        data_dir     = args.data_dir,
        dataset_name = args.dataset_name,
        file_list    = args.file_list,
        alpha_list   = args.alpha_list
    )