import torch
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
import pdb
import numpy as np

# Setup the working dir
import sys
import os
local_openclip_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, local_openclip_path)
from open_clip import tokenizer

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
def compute_score(image_features, text_features):
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)
    return text_probs

def compute_sim(image_features, text_features):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    similarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return similarity

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


### Few-shot Learning Utility Functions ###
def few_shot(
    memory: dict,
    token: torch.Tensor,
    cls_names: list[str],
    row_wise: bool = False
) -> torch.Tensor:
    """
    Few-shot matching with optional row-wise constraint.

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
        medianed.append(torch.median(mem, dim=0).values)  ## TEST MEAN
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



def downsample_mask(anomaly_mask, target_shape, alpha=0):
    """
    Downsample a binary anomaly mask (values 0 or 1) to a token-level mask
    using a majority vote with adjustable sensitivity. A patch is set as anomalous
    if the average value in its block is greater than alpha.
    
    Parameters
    ----------
    anomaly_mask : torch.Tensor
        Tensor of shape [H, W] with binary values.
    target_shape : tuple
        Desired output shape (target_H, target_W) for the downsampled mask.
    alpha : float, optional
        Threshold value for labeling a block as anomalous. Default is 0.5.
    
    Returns
    -------
    down_mask : torch.Tensor
        Flattened tensor of shape [target_H * target_W] with binary values (0 or 1).
    """
    # Use adaptive average pooling so that each output pixel is the average of a block.
    down_mask = F.adaptive_avg_pool2d(anomaly_mask.unsqueeze(0).unsqueeze(0).float(),
                                      target_shape)
    # Apply the adjustable threshold.
    down_mask = (down_mask > alpha).float()
    down_mask = down_mask.squeeze()  # shape: [target_H, target_W]
    return down_mask.view(-1)        # Flatten to [target_H * target_W]

def refine_patch_mask(tokens: torch.Tensor,
                      mask: torch.Tensor,
                      keep_frac: float = 0.25) -> torch.Tensor:
    """
    Given:
      tokens:    [N, D]  — the per‐patch embeddings (already squeezed)
      mask:      [N]     — the binary downsampled mask (0/1)
      keep_frac: float   — fraction of flagged patches to keep

    Return a new mask of shape [N], where only the top keep_frac fraction
    of originally flagged patches (by dissimilarity to *all* other patches)
    remain =1, the rest =0.
    """
    # If tokens have shape [N, 1, D], squeeze to [N, D]
    if tokens.ndim == 3 and tokens.shape[1] == 1:
        tokens = tokens.squeeze(1)  # now [N, D]
    N, D = tokens.shape
    # 1) normalize
    z = F.normalize(tokens, dim=-1)      # [N, D]
    # 2) pairwise cosine similarity
    sim = z @ z.transpose(0,1)           # [N, N]
    # 3) dissimilarity score per patch (here average across peers)
    dissim = 1.0 - sim                   # [N, N]
    score = dissim.mean(dim=1)           # [N]

    # 4) only consider originally flagged patches
    idxs = torch.nonzero(mask, as_tuple=True)[0]
    if idxs.numel() == 0:
        return mask  # nothing to refine

    # 5) pick top-k
    k = max(1, int(idxs.numel() * keep_frac))
    topk_in_flagged = score[idxs].topk(k, largest=True).indices
    chosen = idxs[topk_in_flagged]

    # 6) build new mask
    new_mask = torch.zeros_like(mask)
    new_mask[chosen] = 1
    return new_mask

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