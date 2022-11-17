import torch
from torch import nn
import numpy as np
import torchvision.transforms as T


class DataAugmentation(nn.Module):
    """Module to perform data augmentation on torch tensors."""

    def __init__(self, patch_size, crop_size_perc, rotation_angle, h_flip, shift_crop, shift_pixels, 
                 brightness_factor=0.0, hue_factor=0.0, **args):
        super().__init__()
        self.patch_size = patch_size
        self.crop_size_perc = crop_size_perc
        self.rotation_angle = rotation_angle
        self.h_flip = h_flip
        self.shift_crop = shift_crop
        self.shift_pixels = shift_pixels
        self.brightness_factor = brightness_factor
        self.hue_factor = hue_factor

    
    def random_crop(self, depth=None, event_frames=None, images=None):
        if self.crop_size_perc == 1: return event_frames
        
        # Define relative crop boundaries
        perc_height = np.random.uniform(self.crop_size_perc, 1)
        perc_heigth_init = np.random.uniform(0, 1-perc_height)
        perc_height_end = perc_heigth_init + perc_height
        perc_width = np.random.uniform(self.crop_size_perc, 1)
        perc_width_init = np.random.uniform(0, 1-perc_width)
        perc_width_end = perc_width_init + perc_width

        # Define absolute crop boundaries
        height, width = depth.shape[1:3]
        init_h, end_h = int(perc_heigth_init*height), int(perc_height_end*height)
        init_w, end_w = int(perc_width_init*width), int(perc_width_end*width)

        crop_depth = depth[:, init_h:end_h, init_w:end_w, :]
        if event_frames is not None: crop_events = event_frames[:, init_h:end_h, init_w:end_w, :]
        if images is not None: crop_images = images[:, init_h:end_h, init_w:end_w, :]
        depth = torch.full(depth.shape, float('nan'), dtype=depth.dtype, device=depth.device)
        if event_frames is not None: event_frames = torch.full(event_frames.shape, 0.0, dtype=event_frames.dtype, device=event_frames.device)
        if images is not None: images = torch.full(images.shape, 0.0, dtype=images.dtype, device=images.device)

        if self.shift_crop:        
            width_h = end_h - init_h
            width_w = end_w - init_w
            shift_h, shift_w = np.random.randint(0, height-width_h), np.random.randint(0, width-width_w)
            
            depth[:, shift_h:shift_h+width_h, shift_w:width_w+shift_w, :] = crop_depth
            if event_frames is not None: event_frames[:, shift_h:shift_h+width_h, shift_w:width_w+shift_w, :] = crop_events
            if images is not None: images[:, shift_h:shift_h+width_h, shift_w:width_w+shift_w, :] = crop_images
        else: 
            depth[:, init_h:end_h, init_w:end_w, :] = crop_depth
            if event_frames is not None: event_frames[:, init_h:end_h, init_w:end_w, :] = crop_events
            if images is not None: images[:, init_h:end_h, init_w:end_w, :] = crop_images
        
        return depth, event_frames, images
        
    
    def _shift(self, batch, b):
        raise ValueError('Not implemented')
        mins, maxs = batch['depth'][1][b].min(0)[0].min(0)[0], batch['depth'][1][b].max(0)[0].max(0)[0]
        max_height, max_width = self.height - (maxs[0] - mins[0]), self.width - (maxs[1] - mins[1])
        if max_height == 0 or max_height == 0: return batch
        shift_h, shift_w = torch.randint(0, max_height, (1,), device=batch['depth'][1].device), torch.randint(0, max_width, (1,), device=batch['depth'][1].device)
        
        batch['depth'][1][b][..., 0] = batch['depth'][1][b][..., 0] - mins[0] + shift_h 
        batch['depth'][1][b][..., 1] = batch['depth'][1][b][..., 1] - mins[1] + shift_w 
        batch['depth'][1][b][..., 0] -= shift_h % self.patch_size
        batch['depth'][1][b][..., 1] -= shift_w % self.patch_size
        
        if 'event_frames' in batch:
            batch['event_frames'][1][b][..., 0] = batch['event_frames'][1][b][..., 0] - mins[0] + shift_h 
            batch['event_frames'][1][b][..., 1] = batch['event_frames'][1][b][..., 1] - mins[1] + shift_w 
            batch['event_frames'][1][b][..., 0] -= shift_h % self.patch_size
            batch['event_frames'][1][b][..., 1] -= shift_w % self.patch_size
        if 'images' in batch:
            batch['images'][1][b][..., 0] = batch['images'][1][b][..., 0] - mins[0] + shift_h 
            batch['images'][1][b][..., 1] = batch['images'][1][b][..., 1] - mins[1] + shift_w 
            batch['images'][1][b][..., 0] -= shift_h % self.patch_size
            batch['images'][1][b][..., 1] -= shift_w % self.patch_size
        return batch
    def random_shift(self, batch):
        if not self.shift_pixels: return batch
        with torch.no_grad():
            B = batch['depth'][1].shape[0]
            for b in range(B):
                batch = self._shift(batch, b)
        return batch
    
    
    def ranfom_rotation(self, depth=None, event_frames=None, images=None):
        if self.rotation_angle == 0: return depth, event_frames, images
        rot_ang = np.random.randint(-self.rotation_angle, self.rotation_angle+1)
        if depth is not None:               depth =        T.functional.rotate(depth.permute(0,3,1,2), rot_ang, fill=float('nan')).permute(0,2,3,1)
        if event_frames is not None:        event_frames = T.functional.rotate(event_frames.permute(0,3,1,2), rot_ang, fill=0.0).permute(0,2,3,1)
        if images is not None:              images =       T.functional.rotate(images.permute(0,3,1,2),       rot_ang, fill=0.0).permute(0,2,3,1)
        return depth, event_frames, images
    
    def image_color_transformations(self, images):
        images = images.permute(0, 3, 1, 2)
        if self.brightness_factor!= 0: images = T.functional.adjust_brightness(images, np.random.uniform(max(0, 1-self.brightness_factor), 1+self.brightness_factor))
        if self.hue_factor!= 0: images = T.functional.adjust_hue(images, np.random.uniform(-self.hue_factor, self.hue_factor))
        return images.permute(0, 2, 3, 1)
    
    def transform(self, depth, event_frames, images):
        if self.h_flip and torch.rand(1) > 0.5: 
            if depth is not None: depth = depth.flip(2)
            if event_frames is not None: event_frames = event_frames.flip(2)
            if images is not None: images = images.flip(2)
            
        if images is not None and any([self.brightness_factor != 0, self.hue_factor != 0]):
            images = self.image_color_transformations(images)
            
        if self.rotation_angle > 0: 
            depth, event_frames, images = self.ranfom_rotation(depth, event_frames, images)
            
        if self.crop_size_perc < 1.0: 
            depth, event_frames, images = self.random_crop(depth, event_frames, images)

        return depth, event_frames, images
    
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, batch_samples):
        with torch.no_grad():
            B = batch_samples['depth'].shape[0]
            self.height, self.width = batch_samples['depth'].shape[1:3]
            depth_l, event_frames_l, images_l = [], [], []
            for b in range(B):
                depth, event_frames, images = self.transform(
                                                    batch_samples['depth'][b] if 'depth' in batch_samples is not None else None, 
                                                    batch_samples['event_frames'][b] if 'event_frames' in batch_samples is not None else None,
                                                    batch_samples['images'][b] if 'images' in batch_samples is not None else None,
                                                    )
                depth_l.append(depth)
                event_frames_l.append(event_frames)
                images_l.append(images)
            if depth_l[0] is not None: batch_samples['depth'] = torch.stack(depth_l)
            if event_frames_l[0] is not None: batch_samples['event_frames'] = torch.stack(event_frames_l)
            if images_l[0] is not None: batch_samples['images'] = torch.stack(images_l)

            return batch_samples
                
