from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import argparse
import ast
import pandas as pd
import numpy as np
import gc
import json

import sys
print("system path:", os.getcwd())
sys.path.insert(0, os.getcwd())
from preprocessing.vision_ts_dataset import CLIPTimeSeriesDataset
from models.model_utils import *
from models.clip_vision import CLIP_AD

def main(dataset_name, file_name, window_step_ratio=4, agg_percent=0.25, patch_size=16, no_anomaly=True,  ### 0406, agg_percent to 0.25
          sample_size=None, plot_types=["line"], save_heatmap=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Available device: {device}")
    model = CLIP_AD(model_name='ViT-B-16-plus-240', device=device)
    for plot_type in plot_types:
        results_dir = f'../results/{dataset_name}/{file_name}/'
        # Create the dataset.
        dataset = CLIPTimeSeriesDataset(results_dir=results_dir, base_series_id=file_name,
                                        sample_size=sample_size, no_anomaly=no_anomaly,
                                            plot_type=plot_type)
        # Create a DataLoader.
        test_dataloader = DataLoader(dataset, batch_size=20, shuffle=True)

        model.eval()
        results = {
            'cls_names': [],
            'window_id': [],
            'imgs_masks': [],
            'anomaly_maps': [],
            'gt_sp': [],
            'pr_sp': []
        }

        # Build normal memory bank
        large_memory_normal, mid_memory_normal, patch_memory_normal = build_memory(
            model, test_dataloader, patch_size, device
        )

        with torch.no_grad():
            for index, items in enumerate(tqdm(test_dataloader)):
                # Load image tensor.
                images = items['img'].to(device)       # shape: [B, C, H, W]
                cls_names = items['cls_name']           # list of strings (e.g., all "0")
                results['cls_names'].extend(cls_names)
                results['window_id'].extend(items['window_id'])
                
                # Process ground-truth masks.
                gt_mask = items['img_mask']             # shape: [B, H, W]
                results['imgs_masks'].append(gt_mask)
                results['gt_sp'].extend(items['anomaly'].detach().cpu().tolist())
                
                b, c, h, w = images.shape
                
                # Encode images.
                (large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens,
                large_scale, mid_scale) = model.encode_image(images, patch_size)            
                
                # ----- Vision-Based Anomaly Map (Normal Memory) -----
                m_l_normal = few_shot(large_memory_normal, large_scale_tokens, cls_names)
                m_m_normal = few_shot(mid_memory_normal, mid_scale_tokens, cls_names)
                m_p_normal = few_shot(patch_memory_normal, patch_tokens, cls_names)
                
                m_l_normal = harmonic_aggregation((b, h // patch_size, w // patch_size), m_l_normal, large_scale).to(device)
                m_m_normal = harmonic_aggregation((b, h // patch_size, w // patch_size), m_m_normal, mid_scale).to(device)
                m_p_normal = m_p_normal.reshape((b, h // patch_size, w // patch_size)).to(device)
         
                normal_vision_score = torch.nan_to_num((m_l_normal + m_m_normal + m_p_normal) / 3.0, nan=0.0, posinf=0.0, neginf=0.0)
                
                # ----- Additional Anomaly Memory (Semi-Supervised) -----
                final_score = normal_vision_score.unsqueeze(1)
                final_z0score = torch.max(torch.max(normal_vision_score, dim=1)[0], dim=1)[0]
                
                # Upsample final anomaly map to original image size.
                final_score = F.interpolate(final_score, size=(h, w), mode='bilinear')
                final_score = final_score.squeeze(1)  # shape: [B, H, W]
                
                results['pr_sp'].extend(final_z0score.detach().cpu().tolist())
                # detach & move to CPU
                cpu_map = final_score.detach().cpu()
                results['anomaly_maps'].append(cpu_map)
                del images, large_scale_tokens, mid_scale_tokens, patch_tokens
                del m_l_normal, m_m_normal, m_p_normal, normal_vision_score, final_score
                torch.cuda.empty_cache()

        # Concatenate batch results.
        results['imgs_masks'] = torch.cat(results['imgs_masks'], dim=0)
        results['anomaly_maps'] = torch.cat(results['anomaly_maps'], dim=0).detach().cpu().numpy()

        anomaly_maps = results['anomaly_maps'] # shape: [num_maps, H, W]
        num_maps = anomaly_maps.shape[0]
        print(f"Total anomaly maps: {num_maps}")
        window_ids = np.array(results['window_id'])
        sorted_indices = np.argsort(window_ids)
        sorted_anomaly_maps = results['anomaly_maps'][sorted_indices]
        # Stitch the anomaly maps to obtain the final anomaly score vector.
        final_anomaly_vector = stitch_anomaly_maps(sorted_anomaly_maps, window_step_ratio, agg_percent)
        base_cls = results['cls_names'][0] if results['cls_names'] else "unknown"
        final_filename_base = os.path.join(results_dir, f"{base_cls}_agg")
        # Save the final anomaly vector as a .npy file.
        np.save(final_filename_base + ".npy", final_anomaly_vector)
        print(f"Saved final anomaly score as {final_filename_base}.npy")
        # Also, save a plot of the final anomaly vector as a PNG image.
        plt.figure(figsize=(12, 4))
        plt.plot(final_anomaly_vector, label="Anomaly Score")
        plt.xlabel("Time Index")
        plt.ylabel("Anomaly Score")
        plt.title(f"Final Anomaly Score ({base_cls}")
        plt.legend()
        plt.savefig(final_filename_base + ".png", bbox_inches='tight')
        plt.close()
        print(f"Saved final anomaly score plot as {final_filename_base}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the main script for inference."
    )

    parser.add_argument("--data_dir", type=str, default="../data/raw/",
                        help="Directory of raw data (e.g., CSV files and metadata).")
    parser.add_argument("--dataset_name", type=str, default="MSL",
                        help="Name of the dataset (default: 'MSL').")
    parser.add_argument("--file_name", type=str, default="C-2",
                        help="Name of the file to process (default: C-2).")
    parser.add_argument("--window_step_ratio", type=float, default=4.0,
                        help="Window step ratio (default: 4).")
    parser.add_argument("--agg_percent", type=float, default=0.25,
                        help="Aggregation percentage for anomaly map reduction (default: 0.10).")
    parser.add_argument("--no_anomaly", action="store_true", default=False,
                        help="Flag indicating that no anomaly labels are available (default: False).")
    parser.add_argument("--plot_types", type=str, nargs="+", default=["line"],
                        help="List of plot types (e.g., 'line', 'gaf'; default: ['line']).")

    args = parser.parse_args()


    # Call process_dataset with the provided parameters (other parameters are set to defaults).
    main(
    dataset_name = args.dataset_name,
    file_name = args.file_name, 
    window_step_ratio = args.window_step_ratio,
    agg_percent = args.agg_percent, 
    no_anomaly = args.no_anomaly,
    plot_types = args.plot_types,
    semi_supervised = args.semi_supervised,
    )
