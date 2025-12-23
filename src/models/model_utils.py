import math
import torch
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from scipy.stats import norm
import pandas as pd


### Multi-scale patching Technique ###
class patch_scale():
    def __init__(self, image_size):
        self.h, self.w = image_size
 
    def make_mask(self, patch_size = 16, kernel_size = 16, stride_size = 16): 
        self.patch_size = patch_size
        self.patch_num_h = self.h//self.patch_size
        self.patch_num_w = self.w//self.patch_size
        self.kernel_size = kernel_size//patch_size
        self.stride_size = stride_size//patch_size
        self.idx_board = torch.arange(0, self.patch_num_h * self.patch_num_w, dtype=torch.float32).reshape((1,1,self.patch_num_h, self.patch_num_w))
        patchfy = torch.nn.functional.unfold(self.idx_board, kernel_size=self.kernel_size, stride=self.stride_size)
        return patchfy

### evaluation utility function ###
def harmonic_aggregation(score_size, similarity, mask):
    b, h, w = score_size
    similarity = similarity.double()
    mask = mask.T.long()              

    score = torch.zeros((b, h*w), device=similarity.device).double()

    for idx in range(h*w):
        patch_idx = [bool(torch.isin(idx+1, mask_patch)) for mask_patch in mask]
        # patch_idx is a Python list of bools of length b
        patch_idx = torch.tensor(patch_idx, device=similarity.device)
        sum_num = patch_idx.sum().item()
        harmonic_sum = torch.sum(1.0 / similarity[:, patch_idx], dim=-1)
        score[:, idx] = sum_num / harmonic_sum

    return score.view(b, h, w)


### Patch Dissimilarity Utility Functions ###
def compute_patch_dissimilarity(
    memory: dict,
    token: torch.Tensor,
    cls_names: list[str],
    row_wise: bool = False
) -> torch.Tensor:
    """
    Compute patch dissimilarity by comparing test patches against memory bank.

    Parameters
    ----------
    memory : dict
        cls_name -> Tensor of shape [L, N, D] or [L, N, 1, D]
    token : torch.Tensor
        [B, N, D] or [B, N, 1, D]
    cls_names : list of str
        length B
    row_wise : bool, default False
        If True, only compare each test patch to memory patches in the same image row.
        Otherwise, compare to all memory patches.

    Returns
    -------
    M : torch.Tensor, shape [B, N]
        0.5 * min dissimilarity per token.
    """
    # Prepare test tokens
    if token.ndim == 4 and token.shape[2] == 1:
        token = token.squeeze(2)  # [B, N, D]
    token_norm = F.normalize(token, dim=-1)  # [B, N, D]
    B, N, D = token_norm.shape

    # Compute medianed memory per sample
    medianed = []
    for cls in cls_names:
        mem = memory[cls]  # [L, N, D] or [L, N, 1, D]
        if mem.ndim == 4 and mem.shape[2] == 1:
            mem = mem.squeeze(2)  # [L, N, D]
        # median over L -> [N, D]
        medianed.append(torch.median(mem, dim=0).values)
    retrieved = torch.stack(medianed, dim=0)          # [B, N, D]
    retrieved_norm = F.normalize(retrieved, dim=-1)   # [B, N, D]

    if not row_wise:
        # Universal matching: compare each token to all memory tokens
        # [B, N, D] @ [B, D, N] -> [B, N, N]
        sim = torch.bmm(token_norm, retrieved_norm.permute(0, 2, 1))
        dissim = 1.0 - sim
        # min over memory patches (last dim)
        M = 0.5 * torch.min(dissim, dim=2).values  # [B, N]
        return M

    # Row-wise matching:
    side = int(math.sqrt(N))
    assert side * side == N, "Number of patches N must be a perfect square for row-wise."
    # Precompute row indices
    row_idx = (torch.arange(N, device=token.device) // side).tolist()

    # Compute full similarity for each sample
    M_rows = []
    for b in range(B):
        sim = torch.matmul(token_norm[b], retrieved_norm[b].T)  # [N, N]
        dissim = 1.0 - sim
        # per-patch, restrict to same-row block
        row_dists = []
        for n in range(N):
            r = row_idx[n]
            start = r * side
            end   = start + side
            row_dists.append(dissim[n, start:end].min())
        M_rows.append(torch.stack(row_dists))
    M = torch.stack(M_rows, dim=0)  # [B, N]
    return 0.5 * M

# Prepare the memory banks for comparison
@torch.no_grad()
def build_memory(model, test_dataloader, patch_size, device):
    """
    Build memory banks for comparison by gathering images.
    (assumed normal) for each class (cls_name) in a test dataloader.
    
    Parameters
    ----------
    model : nn.Module
        A CLIP-based model that has an encode_image(...) method returning:
          (large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_scale, mid_scale)
    test_dataloader : DataLoader
        Yields dictionaries with keys: 'img', 'cls_name', 'window_id', 'img_mask', 'anomaly', 'text_prompt'.
    patch_size : int
        Patch size passed to model.encode_image(...).
    device : torch.device
        The device on which computations are performed.
  
    Returns
    -------
    large_memory, mid_memory, patch_memory : dict
        For example, large_memory["0"] is the concatenation of large-scale tokens from the selected images.
    """
    # Dictionaries to accumulate multi-scale tokens for each cls_name.
    large_memory = defaultdict(list)
    mid_memory   = defaultdict(list)
    patch_memory = defaultdict(list)

    for batch in test_dataloader:
        imgs       = batch['img'].to(device)        # shape: [B, C, H, W]
        cls_names  = batch['cls_name']              # list of strings
        window_ids = batch['window_id']             # list of ints
        
        batch_size = imgs.shape[0]
        for i in range(batch_size):
            cls_name_i  = cls_names[i]
            window_id_i = window_ids[i].item()

            img_tensor = imgs[i].unsqueeze(0)  # shape [1, C, H, W]

            # Encode image.
            (large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens,
                large_scale, mid_scale) = model.encode_image(img_tensor, patch_size)

            # Accumulate in dictionaries keyed by cls_name.
            large_memory[cls_name_i].append(large_scale_tokens)
            mid_memory[cls_name_i].append(mid_scale_tokens)
            patch_memory[cls_name_i].append(patch_tokens)


    # Concatenate tokens for each class.
    for cls_name in large_memory.keys():
        large_memory[cls_name] = torch.cat(large_memory[cls_name], dim=0)
        mid_memory[cls_name]   = torch.cat(mid_memory[cls_name], dim=0)
        patch_memory[cls_name] = torch.cat(patch_memory[cls_name], dim=0)

    return dict(large_memory), dict(mid_memory), dict(patch_memory)


### Aggregation Anomaly Maps Utility Functions ###
def aggregate_anomaly_map(anomaly_map, top_percent):
    """
    Aggregate a 2D anomaly map (shape [H, W]) into a 1D vector of length W by averaging
    the top fraction of values in each column.
    
    Parameters
    ----------
    anomaly_map : np.ndarray
        2D array of anomaly scores with shape (H, W).
    top_percent : float
        Fraction (between 0 and 1) indicating the top portion of values to average in each column.
    
    Returns
    -------
    np.ndarray
        1D anomaly vector of length W.
    """
    H, W = anomaly_map.shape
    vector = np.zeros(W, dtype=float)
    for j in range(W):
        col = anomaly_map[:, j]
        k = max(1, int(np.ceil(H * top_percent)))
        # Sort column values in descending order and take the top k.
        sorted_vals = np.sort(col)[::-1]
        vector[j] = np.mean(sorted_vals[:k])
    return vector

def stitch_anomaly_maps(anomaly_maps, window_step_ratio, agg_percent):
    """
    Stitch overlapping anomaly maps into a final anomaly score vector for the entire time series.
    
    For each anomaly map (of shape [H, W]), reduce it to a 1D vector by aggregating each column
    using the provided aggregation fraction (agg_percent). Then, because windows overlap, average the scores 
    for the same global time index.
    
    Parameters
    ----------
    anomaly_maps : np.ndarray
        Array of shape [num_maps, H, W] containing anomaly scores from each window.
    window_step_ratio : float
        The ratio between the window width and the step size.
        That is, step_size = window_width / window_step_ratio.
    agg_percent : float
        The fraction (between 0 and 1) used to average the top values in each column.
    
    Returns
    -------
    np.ndarray
        A 1D array (final anomaly vector) of length T_final, where
        T_final = step_size * (num_maps - 1) + window_width.
    """
    num_maps, H, W = anomaly_maps.shape
    window_width = W  # Each anomaly map's width is the window width.
    step_size = int(window_width / window_step_ratio)
    T_final = step_size * (num_maps - 1) + window_width

    # For each window, reduce its anomaly map to a 1D vector.
    window_vectors = np.array([aggregate_anomaly_map(anomaly_maps[i], agg_percent)
                                 for i in range(num_maps)])  # shape: [num_maps, window_width]

    # Initialize final_scores and count for overlapping regions.
    final_scores = np.zeros(T_final, dtype=float)
    count = np.zeros(T_final, dtype=int)

    # For each window, map its columns into the final vector.
    for i in range(num_maps):
        start = i * step_size
        end = start + window_width
        final_scores[start:end] += window_vectors[i]
        count[start:end] += 1

    # Average overlapping windows.
    count[count == 0] = 1  # Prevent division by zero.
    final_scores = final_scores / count
    return final_scores

def compute_detection_intervals(
    score_vector,
    alpha,
    method="mean",
    smoothing=True,
    sliding=False,
    anomaly_padding=0
):
    """
    Given an anomaly score vector, compute detection intervals using either
    global thresholding or a sliding-window threshold, and optionally
    smooth the scores first via an exponentially-weighted moving average.

    Parameters
    ----------
    score_vector : array-like, shape (T,)
        The aligned anomaly score vector.
    alpha : float
        The upper quantile for thresholding (e.g. 0.05 ⇒ top 5%).
    method : {'mean', 'median'}, default='mean'
        Whether to compute central tendency + spread as (mean, std) or
        (median, MAD).
    smoothing : bool, default=False
        If True, first replace `score_vector` with its EWMA (alpha = smoothing_alpha).
    smoothing_alpha : float in (0,1), default=0.3
        The alpha for the EWMA if `smoothing=True`.
    sliding : bool, default=False
        If True, compute a local threshold in a sliding window (size T/3, step T/10)
        and mark any point exceeding its window's threshold as anomalous.
        Otherwise use a single global threshold.
    anomaly_padding : int, default=0
        Number of time points to pad before and after each detected interval.

    Returns
    -------
    detection_intervals : list of (start, end) tuples
        Contiguous index ranges (padded) where the (smoothed) score exceeds the threshold.
    threshold : float or None
        The global threshold (if sliding=False), else None.
    scores : np.ndarray
        The (optionally smoothed) score vector.
    """
    scores = np.array(score_vector, dtype=float)
    T = len(scores)

    if smoothing:
        span = max(1, int(len(scores) * 0.01))
        scores = pd.Series(scores).ewm(span=span).mean().values

    # Precompute the Gaussian multiplier
    z = norm.ppf(1 - alpha)

    # 2) Build a boolean mask of anomalies
    anomaly_flags = np.zeros(T, dtype=bool)

    if not sliding:
        # 2a) Global threshold
        if method == "mean":
            central = scores.mean()
            spread  = scores.std()
        elif method == "median":
            central = np.median(scores)
            spread  = np.median(np.abs(scores - central))
        else:
            raise ValueError("method must be 'mean' or 'median'")
        threshold = central + z * spread
        anomaly_flags = scores > threshold

    else:
        # 2b) Sliding‑window threshold
        threshold = 0 # With sliding window based method, threshold varies.
        win = max(1, T // 3)
        step = max(1, T // 10)
        for start in range(0, T, step):
            end = min(start + win, T)
            segment = scores[start:end]
            if method == "mean":
                central = segment.mean()
                spread  = segment.std()
            else:  # median
                central = np.median(segment)
                spread  = np.median(np.abs(segment - central))
            thresh_local = central + z * spread
            # mark any point in [start:end) above its local thresh
            anomaly_flags[start:end] |= (segment > thresh_local)

    # 3) Extract contiguous intervals from the boolean mask
    detection_intervals = []
    in_int = False
    for i, flag in enumerate(anomaly_flags):
        if flag and not in_int:
            in_int = True
            start = i
        elif not flag and in_int:
            in_int = False
            detection_intervals.append((start, i-1))
    if in_int:
        detection_intervals.append((start, T-1))


    # 4) Apply padding if requested
    if anomaly_padding > 0:
        padded = []
        for (s, e) in detection_intervals:
            s_pad = max(0, s - anomaly_padding)
            e_pad = min(T - 1, e + anomaly_padding)
            padded.append((s_pad, e_pad))

        # If nothing was detected, stay empty
        if not padded:
            detection_intervals = []
        else:
            # 5) Merge overlapping or contiguous intervals
            padded.sort(key=lambda x: x[0])
            merged = []
            cur_s, cur_e = padded[0]
            for s_next, e_next in padded[1:]:
                if s_next <= cur_e + 1:   # overlap or contiguous
                    cur_e = max(cur_e, e_next)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s_next, e_next
            merged.append((cur_s, cur_e))

            detection_intervals = merged
    
    return detection_intervals, threshold, scores
    
def align_anomaly_vector(final_vector, T_full, window_size, step_size, n_windows):
    """
    Align the stitched anomaly vector to the original time series length T_full,
    by first interpolating it out to the full covered span of your sliding windows,
    then extrapolating or truncating to exactly T_full.

    Parameters
    ----------
    final_vector : np.ndarray
        The stitched anomaly vector of length L (one score per column in the overlap).
    T_full : int
        The total number of time points in the original series.
    window_size : int
        Number of time points per window.
    step_size : int
        Step size between windows.
    n_windows : int
        Number of windows used in the sliding-window imagery.

    Returns
    -------
    aligned_vector : np.ndarray
        A 1D array of length T_full.
    """
    # 1) compute the total span covered by your windows
    covered_length = window_size + (n_windows - 1) * step_size

    L = len(final_vector)
    # 2) interpolate final_vector (length L) → covered_length
    x_old = np.arange(L)
    x_new = np.linspace(0, L - 1, covered_length)
    interp = np.interp(x_new, x_old, final_vector)

    # 3) now align to T_full
    if covered_length < T_full:
        # linear extrapolation past the end
        if covered_length > 1:
            slope = interp[-1] - interp[-2]
        else:
            slope = 0.0
        extra = T_full - covered_length
        extrap = interp[-1] + slope * np.arange(1, extra + 1)
        aligned = np.concatenate([interp, extrap])
    else:
        # just truncate if we overshot
        aligned = interp[:T_full]

    return aligned