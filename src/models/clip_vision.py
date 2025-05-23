import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import open_clip
from models.model_utils import patch_scale

def multi_scale_aggregation(patch_tokens, patch_size, mask_helper):
    """
    Given patch tokens (from a vanilla model), generate multi-scale embeddings 
    by pooling tokens according to different masks.
    
    Parameters
    ----------
    patch_tokens : torch.Tensor
        Tensor of shape [B, num_tokens, D] (the vanilla image embeddings).
    patch_size : int
        The patch size used in the ViT.
    mask_helper : patch_scale
        A helper instance used to create pooling masks.
        
    Returns
    -------
    large_scale_tokens : torch.Tensor
         Pooled tokens at a coarse scale. (Shape: [B, mask_num_large, D])
    mid_scale_tokens : torch.Tensor
         Pooled tokens at a mid scale. (Shape: [B, mask_num_mid, D])
    large_mask : torch.Tensor
         The large-scale mask used (of shape [mask_num_large, L_large] where L_large is number of patches per group).
    mid_mask : torch.Tensor
         The mid-scale mask used (of shape [mask_num_mid, L_mid]).
    """
    large_mask = mask_helper.make_mask(kernel_size=48, patch_size=patch_size, stride_size=patch_size).squeeze(0)  # avoid removing extra dims
    mid_mask = mask_helper.make_mask(kernel_size=32, patch_size=patch_size, stride_size=patch_size).squeeze(0)
    
    B, num_tokens, D = patch_tokens.shape

    def pool_tokens(mask, tokens):
        # mask: Tensor of shape [L, mask_num]
        # tokens: [B, num_tokens, D]
        mask = mask.int()  # ensure indices are ints
        L, mask_num = mask.shape
        pooled_list = []
        for i in range(mask_num):
            indices = mask[:,i]  # a vector of length L containing patch indices
            # Select tokens for each image using the indices.
            # This results in shape [B, L, D].
            selected = torch.index_select(tokens, 1, indices.to(tokens.device))
            # Average over the L dimension to produce one token per group.
            pooled = selected.mean(dim=1, keepdim=True)  # shape: [B, 1, D]
            pooled_list.append(pooled)
        # Concatenate pooled tokens across groups: [B, mask_num, D]
        return torch.cat(pooled_list, dim=1)

    large_scale_tokens = pool_tokens(large_mask, patch_tokens)
    mid_scale_tokens = pool_tokens(mid_mask, patch_tokens)
    return large_scale_tokens, mid_scale_tokens, large_mask, mid_mask


class CLIP_AD(nn.Module):
    def __init__(self, model_name='ViT-B-16-plus-240', device=None, image_size=(240, 240)):
        """
        Initialize the CLIP_AD model.
        
        Parameters
        ----------
        model_name : str, optional
             Name of the CLIP model variant.
        device : torch.device, optional
             Device to use (defaults to CUDA if available).
        image_size : tuple, optional
             Expected image size (height, width) used to create pooling masks.
        """
        super(CLIP_AD, self).__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        vision_cfg = {
          "output_tokens": True,  # Enable token output
          "width": 896,
          "image_size": 240,
          "layers": 12,
          "patch_size": 16
        }
        # Create the vanilla CLIP model and its transforms.
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained='laion400m_e31',
                                                                 vision_cfg=vision_cfg)
        self.model = self.model.to(self.device)
        self.image_size = image_size

        # Create a patch-scale helper instance.
        self.mask = patch_scale(image_size=image_size)
    
    def encode_text(self, text):
        return self.model.encode_text(text)
    
    def encode_image(self, image, patch_size, use_mask=True):
        """
        Encode an image and produce multi-scale embeddings as a post-processing step.

        Parameters
        ----------
        image : torch.Tensor
             Input image tensor with shape [B, C, H, W].
        patch_size : int
             The patch size used by the ViT.
        use_mask : bool, optional
             If True, apply multi-scale pooling; if False, simply output vanilla patch tokens.
        
        Returns
        -------
        large_scale_tokens : torch.Tensor
             Aggregated tokens at the large scale.
        mid_scale_tokens : torch.Tensor
             Aggregated tokens at the mid scale.
        patch_tokens : torch.Tensor
             Raw patch tokens, unsqueezed along dimension 2.
        class_tokens : torch.Tensor
             Global image token computed as average of patch tokens.
        large_mask : torch.Tensor
             The large-scale mask used for pooling.
        mid_mask : torch.Tensor
             The mid-scale mask used for pooling.
        """
        image = image.to(self.device, non_blocking=True)
        _, patch_tokens = self.model.encode_image(image)
        
        if use_mask:
            # Obtain multi-scale embeddings via post-processing.
            large_scale_tokens, mid_scale_tokens, large_mask, mid_mask = multi_scale_aggregation(
                patch_tokens, patch_size, self.mask
            )
        else:
            # If not using multi-scale masking, return patch tokens directly.
            large_scale_tokens, mid_scale_tokens, large_mask, mid_mask = None, None, None, None
        
        # Compute a global class token as the average over patch tokens.
        class_tokens = patch_tokens.mean(dim=1, keepdim=True)
        # For consistency, unsqueeze patch_tokens along dimension 2.
        patch_tokens = patch_tokens.unsqueeze(2)
        
        return large_scale_tokens, mid_scale_tokens, patch_tokens, class_tokens, large_mask, mid_mask