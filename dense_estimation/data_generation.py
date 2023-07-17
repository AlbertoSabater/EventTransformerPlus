from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import os
import numpy as np
import pickle
import torch
import copy

import sys
sys.path.append('..')
import transformations



# Load pickle file and closes it
def load_pickle(filename):
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return p

class MVSEC(Dataset):

    def __init__(self,
                 dataset_name,
                 path_dataset, sequence,      # Dataset split info
                 chunks_per_depth, k, mT, MT, # frame representation params
                 clip_length_ms,
                 depth_size = 1,        # Number of consecutive depth samples to returns
                 use_events=True, use_images=True,
                  **kwargs,
                 ):
        
        self.dataset_name = dataset_name
        self.path_dataset = path_dataset
        self.sequence = sequence
        print('****', self.path_dataset)
        self.side = ['left']    # Force left information
        self.chunks_per_depth = chunks_per_depth; self.k = k; self.mT = mT; self.MT = MT
        
        # =============================================================================
        # Load sample time-stamps
        # =============================================================================
        self.ts_depths = { s:[] for s in self.side}
        self.sample_ind_by_seq = {}
        ind = 0
        for seq in self.side:
            for i, f in enumerate(sorted(os.listdir(f'{self.path_dataset}/{self.sequence}/{seq}/depth/'))): 
                self.ts_depths[seq].append(int(f.split('_')[1].replace('.pckl','')))
                self.sample_ind_by_seq[ind] = (i, seq); ind += 1
            self.ts_depths[seq] = np.array(sorted(self.ts_depths[seq]))
            
        if use_images:
            self.ts_images = { s:[] for s in self.side}
            for s in self.side:
                for f in sorted(os.listdir(f'{self.path_dataset}/{self.sequence}/{s}/image_raw/')): 
                    self.ts_images[s].append(int(f.split('_')[1].replace('.pckl','')))
                self.ts_images[s] = np.array(sorted(self.ts_images[s]))
            
        if use_events:
            self.event_folder = 'event_frames'
            self.ts_events = { s:[] for s in self.side}
            for s in self.side:
                for f in sorted(os.listdir(f'{self.path_dataset}/{self.sequence}/{s}/{self.event_folder}/')): 
                    self.ts_events[s].append(int(f.split('_')[1].replace('.pckl','')))
                self.ts_events[s] = np.array(sorted(self.ts_events[s]))
                
        acum = 0
        res = {}
        for s in self.side:
            acum += len(self.ts_depths[s])
            res[s] = acum
 
            
        if dataset_name == 'MVSEC': depth_ms = 50
        else: raise ValueError(f'dataset_name [{dataset_name}]')
        
        self.clip_length_ms = clip_length_ms
        self.clip_length_us = self.clip_length_ms*1000
        self.num_chunks = (clip_length_ms // depth_ms)+1          # Number of time-steps (T) per sample. Given by the depth frequency (20Hz / 50 ms)
       
        self.depth_size = self.num_chunks if depth_size == -1 else depth_size
        
        self.use_events, self.use_images = use_events, use_images
        
        if self.depth_size > self.num_chunks:
            print(f'*** depth_size [{self.depth_size}] greater than num_chunks [{self.num_chunks}] -> limiting the later')
            self.depth_size = self.num_chunks
        
        print('*** ', self.dataset_name, 'num_chunks:' , self.num_chunks, '*** depth_size:' , self.depth_size)
    
    
    def __len__(self):
        return int(sum([ len(self.ts_depths[s]) for s in self.side ]))
    
    
    
    """
    Group data (event frames, images) by depth chunks (3 event frames, 2-3 images)
    Make patches from events, images
        Drop event patches -> Get pixels -> Flatten patches
    Return dict with data grouped by type (event frames, images) and depth chunks
    """    
    # Return -> [num_timesteps, num_chunk_events, 2pol], [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    def __getitem__(self, idx):
        idx, side = self.sample_ind_by_seq[idx]
        idx = max(self.depth_size, idx)
        ts_depth = self.ts_depths[side][idx]
        images = event_frames = None
        
        # Load Depth
        if self.clip_length_ms == -1: depth_ts = self.ts_depths[side][np.where((self.ts_depths[side] <= ts_depth))[0]]
        else: 
            if self.depth_size == -1: depth_ts = self.ts_depths[side][np.where((self.ts_depths[side] <= ts_depth))[0]][-self.num_chunks:]
            else: depth_ts = self.ts_depths[side][np.where((self.ts_depths[side] <= ts_depth))[0]][-self.depth_size:]
        depth = [ load_pickle(f'{self.path_dataset}/{self.sequence}/{side}/depth/ts_{ts}.pckl') for ts in depth_ts ]
        depth = torch.stack([ torch.from_numpy(i) for i in depth ])
        
        
        # CROP IN TIME
        # Load grayscale images
        # Check no subtimestep is full of zeros
        if self.use_images:
            if self.clip_length_ms == -1: images_ts = self.ts_images[side][np.where((self.ts_images[side] <= ts_depth))[0]]
            else: images_ts = self.ts_images[side][np.where((self.ts_images[side] <= ts_depth))[0]][-self.num_chunks:]
            images = [ load_pickle(f'{self.path_dataset}/{self.sequence}/{side}/image_raw/ts_{ts}.pckl') for ts in images_ts]
            images = [ torch.from_numpy(img[-1]) if img.shape[0] > 0 else torch.zeros((depth[0].shape)) for img in images ]
            
            images = torch.stack(images).float()
            images = images[..., None]
            images /= 255.
        

        if self.use_events:
            if self.clip_length_ms == -1: event_frames_ts = self.ts_events[side][np.where((self.ts_events[side] <= ts_depth))[0]]
            else: event_frames_ts = self.ts_events[side][np.where((self.ts_events[side] <= ts_depth))[0]][-self.num_chunks:]
            event_frames = [ load_pickle(f'{self.path_dataset}/{self.sequence}/{side}/{self.event_folder}/ts_{ts}.pckl') for ts in event_frames_ts ]
            event_frames = [ ef for ef in event_frames if ef.shape[0] > 0]
            event_frames = [ torch.from_numpy(ef[-1]) for ef in event_frames ]
            if len(event_frames) < self.num_chunks and self.clip_length_ms != -1:
                event_frames = [torch.zeros_like(event_frames[-1])]*(self.num_chunks-len(event_frames)) + event_frames
            event_frames = torch.stack(event_frames).float()
            event_frames = event_frames.view(*event_frames.shape[:3], self.k*2)
            
        depth = depth[..., None]
        
        batch_samples = {'depth': depth, 'depth_ts': depth_ts}
        if self.use_events: 
            if event_frames.sum() == 0: 
                print('AAAAAAA', self.sequence, idx, side, event_frames.sum() == 0, images.sum() == 0)
                raise ValueError('Empty event frame')
            batch_samples['event_frames'] = event_frames
            batch_samples['event_frames_ts'] = event_frames_ts
        if self.use_images: 
            batch_samples['images'] = images
            batch_samples['images_ts'] = images_ts
        
        return batch_samples
    
        

# Return the batch sample indices randomly.
class CustomBatchSampler():
    
    def __init__(self, batch_size, dt, depth_size, sample_repetitions, iterative=False, skip_samples=1):      # , skip_evaluations = 1
        assert batch_size % sample_repetitions == 0
        self.batch_size = batch_size
        self.dt = dt
        self.depth_size = depth_size
        self.sample_repetitions = sample_repetitions
        self.iterative = iterative
        self.skip_samples = skip_samples
        self.num_dt_samples = dt.__len__()
        print(f' * Creating CustomBatchSampler with {self.__len__()} epochs')
        
    def __len__(self):
        if not self.iterative:
            epoch_length = np.ceil(self.num_dt_samples * self.sample_repetitions / self.batch_size / self.depth_size)
            return int(epoch_length)
        else:
            return int(np.ceil(self.num_dt_samples/self.depth_size/self.skip_samples / self.batch_size))
    
    def __iter__(self):
        if not self.iterative:
            batches = []
            for _ in range(self.__len__()):
                batch = np.random.randint(self.num_dt_samples, size=int(self.batch_size // self.sample_repetitions)).repeat(self.sample_repetitions)
                batches.append(batch)
            return iter(batches)
        else:
            batches = np.arange(self.num_dt_samples-1, 0, -self.depth_size)[::self.skip_samples]
            batches = [ l.tolist() for l in np.array_split(batches, self.__len__()) if len(l) > 0 ]
            return iter(batches)
    
    
class MVSEC_DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_params, batch_size, patch_size, min_activations_per_patch, 
                 drop_token, depth_size, sample_repetitions, skip_val_samples, num_workers, pin_memory, 
                 **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_params = data_params
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.min_activations_per_patch = min_activations_per_patch
        self.drop_token = drop_token
        self.depth_size = depth_size
        self.sample_repetitions = sample_repetitions
        self.skip_val_samples = skip_val_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.pixel_dim = data_params['k']*2
        self.original_height, self.original_width = 260, 346
        
        # Set image dimensions after padding
        self.height = self.original_height + patch_size - self.original_height % patch_size if self.original_height % patch_size != 0 else self.original_height
        self.width =  self.original_width + patch_size - self.original_width % patch_size if self.original_width % patch_size != 0 else self.original_width
        
        
        if self.dataset_name == 'MVSEC':
            self.min_dist = [2]
            self.max_dist = [80]
            self.path_dataset = [f'../datasets/MVSEC_final/0601_dataset_frames_k{data_params["k"]}_MT{data_params["MT"]//1000}_mT{50//data_params["k"]}_v0_clear/']
        else: raise ValueError(f'Dataset [{self.dataset_name}] not handled')

        
        
    # Return dict:
        # depth -> B x H x W
        # event_frames  -> B x num_time_steps x num_sub_time_steps (chunks_per_depth) x H x W x K 2 
        # images        -> B x num_time_steps x num_sub_time_steps (2-3) x H x W
    def custom_collate_fn(self, batch_samples): 
        res = {}
        if 'depth' in batch_samples[0]: 
            res['depth'] = torch.stack([ d['depth'] for d in batch_samples ])
            # print('***', res['depth'].shape)
            res['depth'] = transformations.pad(res['depth'], patch_size=self.patch_size, pad_value=float('nan')).float()

        if 'images' in batch_samples[0]: 
            # Complete image stacks with zeros to fit max size
            max_T = max([ img['images'].shape[0] for img in batch_samples ])
            res['images'] = [ img['images'] if img['images'].shape[0] == max_T else torch.cat([torch.zeros(max_T - img['images'].shape[0], *img['images'].shape[1:]), img['images']]) for img in batch_samples ]
            res['images'] = torch.stack([ d for d in res['images'] ])
            res['images'] = transformations.pad(res['images'], patch_size=self.patch_size, pad_value=0.0)

        if 'event_frames' in batch_samples[0]: 
            res['event_frames'] = torch.stack([ d['event_frames'] for d in batch_samples ])
            res['event_frames'] = transformations.pad(res['event_frames'], patch_size=self.patch_size, pad_value=0.0)

        return res


    def train_dataloader(self):
        datasets = []
        for dataset_id, dataset_name in enumerate([self.dataset_name]):
            if dataset_name == 'MVSEC': sequence = 'outdoor_day2'
            else: raise ValueError('dataset_name [{self.dataste_name}] not handled')
            
            path_dataset = self.path_dataset[dataset_id]
            data_params = copy.deepcopy(self.data_params)
            data_params['dataset_name'] = dataset_name

            dt = MVSEC(path_dataset = path_dataset, validation=False, sequence = sequence, **data_params)
            sampler = CustomBatchSampler(self.batch_size, dt, self.depth_size, self.sample_repetitions)
            
            collate_fn = self.custom_collate_fn
            dl = DataLoader(dt, batch_sampler=sampler, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory)
            datasets.append(dl)
        return datasets
    
    def val_dataloader(self):
        # Limit validation on local
        if self.dataset_name in ['MVSEC', 'both']:
            sequences = ['outdoor_day1', 
                          'outdoor_night1', 
                          'outdoor_night2', 
                          'outdoor_night3'
                          ]
        else: raise ValueError('dataset_name [{self.dataste_name}] not handled')
        
        path_dataset = self.path_dataset[0]
        data_params = copy.deepcopy(self.data_params)
        data_params['dataset_name'] = self.dataset_name if self.dataset_name != 'both' else 'MVSEC'
        
        datasets = []
        for seq in sequences:
            dt = MVSEC(path_dataset = path_dataset, validation=True, sequence = seq, **data_params)
            sampler = CustomBatchSampler(self.batch_size, dt, self.depth_size, self.sample_repetitions, 
                                          iterative=True, skip_samples=self.skip_val_samples)
            collate_fn = self.custom_collate_fn
            dl = DataLoader(dt, batch_sampler=sampler, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory)
            datasets.append(dl)
        
        return datasets
        
        
