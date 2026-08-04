import torch

def generate_pixel_mask(shape, sparsity_level, device):
    """
    Generates a fixed binary mask across all input timesteps.
    sparsity_level: fraction of pixels kept (e.g. 0.05 for 5% sensors)
    shape: (H, W) or (B, H, W)
    Returns: mask of shape (B, 1, H, W)
    """
    if len(shape) == 2:
        H, W = shape
        B = 1
    elif len(shape) == 3:
        B, H, W = shape
    else:
        B, C, H, W = shape # ignore C
        
    mask = torch.rand(B, 1, H, W, device=device) < sparsity_level
    return mask.float()

def apply_sparsity(u_obs, mask):
    """
    Applies the mask to the input observation.
    Zero-fills unobserved pixels and concatenates explicit binary mask.
    u_obs: (B, C, H, W)
    mask: (B, 1, H, W)
    Returns: (B, C+1, H, W)
    """
    u_masked = u_obs * mask
    # Broadcast mask to match batch size if necessary
    if mask.shape[0] != u_obs.shape[0]:
        mask = mask.expand(u_obs.shape[0], -1, -1, -1)
    
    return torch.cat([u_masked, mask], dim=1)
