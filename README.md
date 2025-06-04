# VLM4TS: Harnessing Vision-Language Models for Time Series Anomaly Detection

This repository implements **ViT4TS** (vision-screening) and **VLM4TS** (VLM-based verification) for unsupervised time-series anomaly detection.

---

## 📦 Repository Structure

```
├── data
│   └── raw                # Place raw benchmark data here (NAB, NASA, Yahoo)
├── results                # Generated detection results and metrics
├── src
│   ├── evaluation         # Scoring, visualization, and utilities
│   ├── models             # ViT4TS and VLM4TS code
│   └── preprocessing      # Data conversion and image rendering
├── run_vit4ts.sh          # ViT4TS screening pipeline
├── run_vlm4ts.sh          # VLM4TS verification pipeline
└── README.md              # You are here
```

---

## 🚀 Quick Start

### 1. Environment Setup

1. Clone this repo.

2.	Create and activate a Python environment (e.g. venv or conda).

3.	Install dependencies (listed in requirements.txt):

4. Data Preparation
   
	1.	Download the benchmark datasets into data/raw/:
    
- NAB: https://github.com/numenta/NAB
- NASA SMAP & MSL: https://github.com/khundman/telemanom
- Yahoo S5: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70
  
	2.	Convert each dataset into the Orion standard format (See https://github.com/sintel-dev/Orion for details).

5. Run ViT4TS (Screening)

```bash
bash run_vit4ts.sh
```

This will:
- Render sliding-window plots and extract multi-scale embeddings.
- Perform cross-patch comparison to generate high-recall anomaly proposals.
- Save proposals, plots, evaluation metrics.

1. Run VLM4TS (Verification)

```bash
bash run_vlm4ts.sh
```

This will:
- Prompt the VLM (e.g. GPT-4o via API) with proposals for global-context verification.
- Output final detections, evaluation metrics.


## 📜 License

This project is released under the MIT License.


