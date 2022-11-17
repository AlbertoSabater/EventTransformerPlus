import torch
from torch import nn
import numpy as np
import torchvision.transforms as T
import math


class DataAugmentation(nn.Module):
    """Module to perform data augmentation"""

    def __init__(self, patch_size, crop_size_perc, rotation_angle, h_flip, shift_crop, shift_pixels, **args):
        super().__init__()
        self.patch_size = patch_size
        self.crop_size_perc = crop_size_perc
        self.rotation_angle = rotation_angle
        self.h_flip = h_flip
        self.shift_crop = shift_crop
        self.shift_pixels = shift_pixels

    
    def random_crop(self, event_frames):
        if self.crop_size_perc == 1: return event_frames
        
        # Define relative crop boundaries
        perc_height = np.random.uniform(self.crop_size_perc, 1)
        perc_heigth_init = np.random.uniform(0, 1-perc_height)
        perc_height_end = perc_heigth_init + perc_height
        perc_width = np.random.uniform(self.crop_size_perc, 1)
        perc_width_init = np.random.uniform(0, 1-perc_width)
        perc_width_end = perc_width_init + perc_width

        # Define absolute crop boundaries
        height, width = event_frames.shape[1:3]
        init_h, end_h = int(perc_heigth_init*height), int(perc_height_end*height)
        init_w, end_w = int(perc_width_init*width), int(perc_width_end*width)

        crop = event_frames[:, init_h:end_h, init_w:end_w, :]
        event_frames = torch.full(event_frames.shape, 0.0, dtype=event_frames.dtype, device=event_frames.device)

        if self.shift_crop:        
            width_h = end_h - init_h
            width_w = end_w - init_w
            shift_h, shift_w = np.random.randint(0, height-width_h), np.random.randint(0, width-width_w)
            
            event_frames[:, shift_h:shift_h+width_h, shift_w:width_w+shift_w, :] = crop
        else: 
            event_frames[:, init_h:end_h, init_w:end_w, :] = crop
        
        return event_frames
        
    
    def _shift(self, pixels):
        mins, maxs = pixels.min(0)[0].min(0)[0], pixels.max(0)[0].max(0)[0]
        max_height, max_width = self.height - (maxs[0] - mins[0]), self.width - (maxs[1] - mins[1])
        if max_height == 0 or max_height == 0: return pixels
        shift_h, shift_w = torch.randint(0, max_height, (1,), device=pixels.device), torch.randint(0, max_width, (1,), device=pixels.device)
        pixels[..., 0] = pixels[..., 0] - mins[0] + shift_h 
        pixels[..., 1] = pixels[..., 1] - mins[1] + shift_w 
        pixels[..., 0] -= shift_h % self.patch_size
        pixels[..., 1] -= shift_w % self.patch_size
        return pixels
    # pixels:  (B, T, num_tokens, 2)
    def random_shift(self, pixels):
        if not self.shift_pixels: return pixels
        with torch.no_grad():
            B = pixels.shape[0]
            for b in range(B):
                pixels[b] = self._shift(pixels[b])
        return pixels
    
    
    def ranfom_rotation(self, event_frames):
        if self.rotation_angle == 0: return event_frames
        rot_ang = np.random.randint(-self.rotation_angle, self.rotation_angle+1)
        event_frames = T.functional.rotate(event_frames, rot_ang, fill=0.0).permute(0,3,1,2).permute(0,2,3,1)
        return event_frames
    
    
    def transform(self, event_frames):
        if self.h_flip and torch.rand(1) > 0.5: event_frames = event_frames.flip(2)
        if self.rotation_angle > 0: event_frames = self.ranfom_rotation(event_frames)
        if self.crop_size_perc < 1.0: event_frames = self.random_crop(event_frames)
        return event_frames
    
    
    def patches_to_ViT_format(self, patches):
        patches = ((patches * 2)-1)/math.sqrt(self.patch_size)
        return patches
    
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, batch_samples):
        with torch.no_grad():
            B = batch_samples['event_frames'].shape[0]
            self.height, self.width = batch_samples['event_frames'].shape[2:4]
            event_frames_l = []  
            for b in range(B):
                event_frames = self.transform(batch_samples['event_frames'][b])
                event_frames_l.append(event_frames)
                
            batch_samples['event_frames'] = torch.stack(event_frames_l)
                
            return batch_samples
                
