import torch
import torch.nn.functional as F


# # https://github.com/microsoft/Swin-Transformer/blob/5d2aede42b4b12cb0e7a2448b58820aeda604426/models/swin_transformer.py#L33
@torch.no_grad()
def window_partition(inpt, patch_size, validation, min_activations_per_patch=0, drop_token=0.0, reduce_tokens=False,
                     chunk_len_ms=None, maxTime=None, patch_by_last_k=False, **kargs):
    B, T, H, W, C = inpt.shape
    token_size = patch_size*patch_size*C
    
    x = inpt.view(B, T, H // patch_size, patch_size, W // patch_size, patch_size, C)
    tokens = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, T, -1, patch_size*patch_size, C)   # (B, T, num_tokens, patch_size^2, C)
    
    # Clean patches with not enough information
    min_ammount_of_events_per_patch = int(patch_size*patch_size*min_activations_per_patch/100)
    # init_shape = tokens.shape
    
    if not patch_by_last_k:
        # Drop patches with not enough activated pixels
        tokens[(tokens.sum(-1) != 0).sum(-1) < min_ammount_of_events_per_patch] = 0.0
    else:
        # Drop patches with not enough activated pixels in the last chunk_len_ms step
        tokens[(tokens.max(-1)[0] >= (1 - (chunk_len_ms*1000/maxTime))).sum(-1) < min_ammount_of_events_per_patch] = 0.0
    
    tokens = tokens.contiguous().view(B, T, -1, token_size)     # (B, T, num_tokens, token_size)
    
    pixels_h = torch.arange(0, H, patch_size).repeat_interleave(W//patch_size)
    pixels_w = torch.arange(0, W, patch_size).repeat(H//patch_size)   
    pixels = torch.stack([pixels_h, pixels_w], axis=-1).to(device=tokens.device)
    pixels = pixels.expand(B,T,*pixels.shape)
    
    if not validation and drop_token:
        tokens[torch.rand(tokens.shape[:-1]) < drop_token] = 0.0
    
    if reduce_tokens:
        # Move non activated patches (empty) at the end of the tensor
        t_sums = (tokens.sum(-1) == 0.0)
        inds = t_sums.cpu().argsort(-1).to(device=tokens.device)
        pixels = pixels.gather(-2, inds.unsqueeze(-1).repeat(1,1,1,2))
        tokens = tokens.gather(-2, inds.unsqueeze(-1).repeat(1,1,1,token_size))
        
        # Remove last part of the tensor that has only empty patches
        keep_inds = tokens.sum(-1).sum(0).sum(0) != 0
        pixels = pixels[:,:,keep_inds]
        tokens = tokens[:,:,keep_inds]
        
    return tokens, pixels


@torch.no_grad()
def pad(x, patch_size, pad_value):
    with torch.no_grad():
        B, T, H, W, C = x.shape
        diff_h, diff_w = H % patch_size, W % patch_size
        if diff_h > 0: diff_h = patch_size - diff_h
        if diff_w > 0: diff_w = patch_size - diff_w
        x = F.pad(x, (0,0, diff_w//2,diff_w-(diff_w//2), diff_h//2,diff_h-(diff_h//2), 0,0, 0,0), value=pad_value)
        return x
    