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

from pathlib import Path
import socket


# Load pickle file and closes it
def load_pickle(filename):
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return p

class EVIMO(Dataset):

    def __init__(self,
                 dataset_name,
                 path_dataset, 
                 side,
                 k, mT, MT, 
                 clip_length_ms,
                 skip_chunk,
                 depth_size = 1,
                   **kwargs,
                 ):
        
        self.dataset_name = dataset_name
        self.path_dataset = Path(path_dataset)
        self.clip_length_ms = clip_length_ms
        self.skip_chunk = skip_chunk
        print('****', self.path_dataset)
        self.side = side
        self.k = k; self.mT = mT; self.MT = MT
        
        # =============================================================================
        # Load sample time-stamps
        # =============================================================================
        self.ts_depths = { s:[] for s in self.side }
        self.sample_ind_by_seq = {}
        
        assert depth_size == 1
        
        # %%
        
        self.chunk_len_ms = 1000/60
        self.num_chunks = int(self.clip_length_ms / self.chunk_len_ms )
        
        
        ind = 0
        for mode in self.side:
            seqs = sorted(list(self.path_dataset.glob(mode + '__*')))
            assert len(seqs) > 0
            for seq in seqs:
                seq_depth_ts = sorted(list((seq / 'depth').iterdir()))[self.num_chunks:]
                assert len(seq_depth_ts) > 0
                if len(seq_depth_ts) == 0: continue
                self.ts_depths[mode].append(seq_depth_ts)
                for i, depth in enumerate(seq_depth_ts): 
                    self.sample_ind_by_seq[ind] = (i, depth)
                    ind += 1
            
        
        print(f'*** {self.dataset_name} | num_chunks: {self.num_chunks} | num_depths {len(self.sample_ind_by_seq)} | num_sequences: {len(self.ts_depths)}')
        print([ (k,len(v)) for k,v in self.ts_depths.items() ])
        print([ (mode,sum([ len(s) for s in seqs ])) for mode,seqs in self.ts_depths.items() ])
    
    
    def __len__(self):
        # return int(sum([ len(self.ts_depths[s]) for s in self.side ]))
        return len(self.sample_ind_by_seq)
    
    
    """
    Group data (event frames, images) by depth chunks (3 event frames, 2-3 images)
    Make patches from events, images
        Drop event patches -> Get pixels -> Flatten patches
    Return dict with data grouped by type (event frames, images) and depth chunks
    """    
    # Return -> [num_timesteps, num_chunk_events, 2pol], [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    def __getitem__(self, idx):
        idx, side = self.sample_ind_by_seq[idx]
        ts_depth = self.sample_ind_by_seq[idx]
        images = event_frames = None
        
        # Load Depth
        depth_ts = ts_depth[1]
        depth = np.load(depth_ts)[None, ...].astype('float32')
        depth = torch.from_numpy(depth)
        
        depth[depth == 0.] = np.nan

        depth_ind = int(depth_ts.stem)
        event_frames_files = []
        i = depth_ind
        while len(event_frames_files) < self.num_chunks:
            eff = depth_ts.parent.parent / f'mono_clms{int(self.chunk_len_ms)}_mt{int(self.mT/1000)}.0_MT{int(self.MT/1000)}' / f'{i:010}.npy'
            i -= 1
            if not eff.exists(): continue
            event_frames_files.append(eff)
        event_frames_files = [ ef for ef in event_frames_files if ef.exists() ]
        event_frames_files = event_frames_files[(len(event_frames_files)%self.skip_chunk+self.skip_chunk-1)%self.skip_chunk::self.skip_chunk]
        event_frames = [ np.load(ev) for ev in event_frames_files ]
        event_frames = [ torch.from_numpy(ef) for ef in event_frames ]
        event_frames = torch.stack(event_frames).float()
        event_frames = event_frames.view(*event_frames.shape[:3], self.k*2)
        
        
        
        depth = depth[..., None]
        
        batch_samples = {'depth': depth}
        batch_samples['event_frames'] = event_frames
        
        return batch_samples
    
    
    
class EVIMO_DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_params, batch_size, patch_size, min_activations_per_patch, 
                 drop_token, 
                  depth_size, 
                 num_workers, pin_memory, 
                  **kwargs
                 ):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_params = data_params
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.min_activations_per_patch = min_activations_per_patch
        self.drop_token = drop_token
        self.depth_size = depth_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.pixel_dim = data_params['k']*2
        self.original_height, self.original_width = 240, 320
        
        # Set image dimensions after padding
        self.height = self.original_height + patch_size - self.original_height % patch_size if self.original_height % patch_size != 0 else self.original_height
        self.width =  self.original_width + patch_size - self.original_width % patch_size if self.original_width % patch_size != 0 else self.original_width
        
        
        if self.dataset_name == 'EVIMO':
            self.min_dist, self.max_dist = 224, 1881
            if socket.gethostname() == 'snowbal': 
                self.path_dataset = Path('/home/asabater/datasets/EVIMO/processed_sequences_mT16667_v0_clear')
            else:                        
                self.path_dataset = Path('/raid/ropert/asabater/datasets/EVIMO/processed_sequences_mT16667_v0_clear')
        else: raise ValueError(f'Dataset [{self.dataset_name}] not handled')

        
        
    # Return dict:
        # depth -> B x H x W
        # event_frames  -> B x num_time_steps x num_sub_time_steps (chunks_per_depth) x H x W x K 2 
        # images        -> B x num_time_steps x num_sub_time_steps (2-3) x H x W
    def custom_collate_fn(self, batch_samples): 
        res = {}
        res['depth'] = torch.stack([ d['depth'] for d in batch_samples ])
        res['depth'] = transformations.pad(res['depth'], patch_size=self.patch_size, pad_value=float('nan')).float()

        res['event_frames'] = torch.stack([ d['event_frames'] for d in batch_samples ])
        res['event_frames'] = transformations.pad(res['event_frames'], patch_size=self.patch_size, pad_value=0.0)
        
        return res


    def train_dataloader(self):
        data_params = copy.deepcopy(self.data_params)
        data_params['dataset_name'] = self.dataset_name

        dt = EVIMO(path_dataset = self.path_dataset / 'train', **data_params) # , sequence = sequence, validation=False, 
        
        collate_fn = self.custom_collate_fn
        dl = DataLoader(dt, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory)    # , batch_sampler=sampler
        return dl
    
    def val_dataloader(self):
        data_params = copy.deepcopy(self.data_params)
        data_params['dataset_name'] = self.dataset_name if self.dataset_name != 'both' else 'MVSEC'
        
        dt = EVIMO(path_dataset = self.path_dataset / 'eval', **data_params) # , sequence = sequence, validation=True
        collate_fn = self.custom_collate_fn
        dl = DataLoader(dt, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory) # batch_sampler=sampler, 
        
        return dl
        
        
