#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:10:14 2023

@author: asabater
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil 

from events_to_frames import process_event_stream

import torch
import sys; sys.path.append('..'); import transformations



height, width = 480, 640
chunk_len_ms = 1000/60
chunk_len_us = chunk_len_ms*1000
k,  maxTime = 3, 256*1000
# minTime = 60000/k
minTime = chunk_len_us

min_patches = 3



path_dataset_org = Path('/mnt/hdd/datasets_hdd/EVIMO/dataset')
# path_dataset_dst = Path('/mnt/hdd/datasets_hdd/EVIMO/processed_sequences')
path_dataset_dst = Path(f'/mnt/hdd/datasets_hdd/EVIMO/processed_sequences_mT{minTime:.0f}_v0_clear')

path_dataset_dst.mkdir(parents=True, exist_ok=True)
(path_dataset_dst / 'train').mkdir(parents=True, exist_ok=True)
(path_dataset_dst / 'eval').mkdir(parents=True, exist_ok=True)


# %%


samples = list((path_dataset_org).glob('samsung_mono/*/*/*'))
samples = [ s for s in samples if 'sanity' not in str(s)]


good_samples, bad_samples = 0, 0
for s in tqdm(samples):
    
    # mode / env, sample_name, sensor
    path_sample = path_dataset_dst / ('eval' if 'eval' in str(s) else 'train') / f'{str(s).split("/")[-3]}__{str(s).split("/")[-1]}'
    print(path_sample)
    

    path_sample.mkdir(parents=True, exist_ok=True)
    path_sample_depth = path_sample / 'depth'
    path_sample_events = path_sample / f'mono_clms{int(chunk_len_ms)}_mt{minTime//1000}_MT{maxTime//1000}'
    path_sample_depth.mkdir(parents=True, exist_ok=True)
    path_sample_events.mkdir(parents=True, exist_ok=True)
    
    
    mono = np.concatenate([np.load(s / 'dataset_events_xy.npy'), 
                                    (np.load(s / 'dataset_events_t.npy')[:,None]*1000*1000).astype('uint'),
                                    np.load(s / 'dataset_events_p.npy')[:,None],
                                    ], axis=-1).astype('uint32')
    # Reduce size
    mono = mono[(mono[:,0] % 2 == 0) & (mono[:,1] % 2 == 0)]
    mono[:,0] = mono[:,0] // 2
    mono[:,1] = mono[:,1] // 2
    mono_depth = np.load(s / 'dataset_depth.npz')
    mono_keys = sorted(list(mono_depth.keys()))
    meta = np.load(s / 'dataset_info.npz', allow_pickle=True)['meta'].item()['frames']
    mono_ts = np.array(sorted([ d['ts'] for d in np.load(s / 'dataset_info.npz', allow_pickle=True)['meta'].item()['frames'] ]))
    # mono_ts_keys = sorted([ d['gt_frame'] for d in np.load(s / 'dataset_info.npz', allow_pickle=True)['meta'].item()['frames'] ])
    

    if len(mono_ts) != len(mono_keys):
        bad_samples += 1
    else: good_samples += 1
    
    
    # Goal: mono_ts == frames
    
    # Remove events after last depth_ts
    last_depth_ts = mono_ts[-1]
    mono = mono[mono[:,2]/1000000 < last_depth_ts]

    # # Remove depths (keys) after last event
    # last_event_ts = ((mono[:,2]-1)/1000000).max()
    # depth_good_inds = ~(mono_ts <= (last_event_ts + 1000/59/1000))   # Items to remove
    # for i in sorted(np.where(depth_good_inds)[0], reverse=True): 
    #     del mono_keys[mono_keys.index(f'depth_{i:>010}')]
    # mono_ts = mono_ts[~depth_good_inds]
    
    # # Remove depths before first event
    # first_event_ts = ((mono[:,2]-1)/1000000).min()
    # depth_good_inds = ~(mono_ts >= (first_event_ts))   # Items to remove
    # for i in sorted(np.where(depth_good_inds)[0], reverse=True): 
    #     del mono_keys[mono_keys.index(f'depth_{i:>010}')]
    # mono_ts = mono_ts[~depth_good_inds]
    
    
    print(f' | mono_depth {len(mono_depth)} | mono_keys {len(mono_keys)} | mono_ts {len(mono_ts)}')
    frames, _, tw_ends = process_event_stream(mono, height//2, width//2, chunk_len_us, k, minTime, maxTime, 
                                     pos_fifo=None, neg_fifo=None, return_tw_ends=True)
    tw_ends = tw_ends / 1000000
    
    # Remove depths (keys) after last event frame
    # Remove depth (keys) before first event frame
    first_frame_ts, last_frame_ts = tw_ends[0], tw_ends[-1]
    good_depth_ts = (mono_ts > first_frame_ts) & (mono_ts <= (last_frame_ts+chunk_len_ms/1000))
    for i in sorted(np.where(~good_depth_ts)[0], reverse=True): 
        mk = f'depth_{i:>010}'
        if mk in mono_keys: del mono_keys[mono_keys.index(mk)]
    mono_ts = mono_ts[good_depth_ts]
    
    # Remove frames before first depth
    first_depth_ts = mono_ts[0]
    depth_frame_inds = tw_ends > first_depth_ts    
    
    
    print(f' | mono_depth {len(mono_depth)} | mono_keys {len(mono_keys)} | mono_ts {len(mono_ts)} | frames {len(frames)}')
    
    assert len(mono_ts) == len(frames) 


    
    # store depth
    for i in range(1, len(frames)):

        mk = f'depth_{i:>010}'

        # Convert to patches
        events_pad = transformations.pad(torch.tensor(frames[i][None,None,].reshape(1, 1, height//2, width//2, k*2).copy()), patch_size=12, pad_value=0.0)
        patches, pixels = transformations.window_partition( events_pad, patch_size=12, validation=False,
                       min_activations_per_patch=7.5, 
                       drop_token=0.0, 
                        chunk_len_ms=1000/60, maxTime=maxTime, 
                        patch_by_last_k=True,
                        reduce_tokens=True)   
        num_patches = (patches.sum(-1) != 0).sum(-1)

        if num_patches[0][0] < min_patches: continue

        
        # Save depth
        if mk in mono_keys:
            depth_i = mono_depth[mk]
            depth_i = depth_i[::2, ::2]
            np.save(path_sample_depth / f'{i+1:>010}', depth_i)
        
        # Save events
        np.save(path_sample_events / f'{i+1:>010}', frames[i].astype('float32'))


    # break
    del frames
    del mono
    del depth_i

    



print(f'good_samples {good_samples} | bad_samples {bad_samples}')


