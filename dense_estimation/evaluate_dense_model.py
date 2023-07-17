import sys; sys.path.append('..')
from data_generation import MVSEC_DataModule
import transformations
from trainer import log_to_abs, mean_error

from EvT_DepthEstimation import DepthDecoder

import json
import torch
from tqdm import tqdm
import time
import numpy as np
from ptflops import get_model_complexity_info
import os

import copy



def load_model_evaluation_hparams(path_model):
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    all_params['model_params']['depth_size'] = 1        # Load just the last depth map for each subsequence
    all_params['data_params']['depth_size'] = 1         # Load just the last depth map for each subsequence
    all_params['data_params']['skip_val_samples'] = 1   # Return all depth_maps
    all_params['data_params']['sample_repetitions'] = 1               # Do not repeat the loaded sequences
    all_params['data_params']['skip_val_samples'] = 1               # Do not skip the loaded sequences
    all_params['data_params']['num_workers'] = 8
    all_params['data_params']['pin_memory'] = True
    return all_params

dists = [10, 20, 30]
seq_info = {'D1': 0, 'N1': 1, 'N2': 2, 'N3': 3}
# device = 'cpu'
device = 'cuda'


path_model = '../trained_models/dense_models/eve'
# path_model = '../trained_models/dense_models/img_eve'



all_params = load_model_evaluation_hparams(path_model)
data_params = all_params['data_params']

from pathlib import Path



"""
performance -> mean_error: per subsequence
complexity: -> FLOPs: per time-window, modified predictions
efficeincy: -> time: per time-window
"""
evaluation_mode = 'performance'               # mean_error@d | Increase batch_size, faster execution
# evaluation_mode = 'efficiency'              # time / DeltaT | batch_size = 1, calcualte eff. stats just for a single time-window (50ms)
# evaluation_mode = 'complexity'              # FLOPs / DeltaT | batch_size = 1, calcualte eff. stats just for a single time-window (50ms)

if evaluation_mode == 'performance':
    data_params['batch_size'] = 16 
    # data_params['batch_size'] = 32 
    # data_params['batch_size'] = 48
    stats = { seq: {f'mean_error_{d}':[]  for d in dists } for seq in seq_info.keys() }

    def print_stats(stats, seq=None):
        seqs = ([seq] if seq else list(seq_info.keys()))
        if len(seqs) > 1:
            stats['global'] = { k:sum([ stats[seq][k] for seq in seq_info.keys() ], []) for k in stats[seqs[0]] }
            seqs.append('global')
            
        for seq in seqs:
            print('\n\n' + '*'*10, seq, '*'*10)
            print('*'*24)
            if f'mean_error_{dists[0]}' in stats[seq]: 
                s = []
                for dist in dists:
                    s.append(f'{np.mean(stats[seq][f"mean_error_{dist}"]):.2f}')
                print('mean_error individual:', ' & '.join(s))
        
        
elif evaluation_mode == 'efficiency':   # Time to process a single time-window
    data_params['batch_size'] = 1 
    data_params['clip_length_ms'] = 49  # sequence length same as the time-window
    stats = { seq: {'num_patches_events':[], 'num_patches_images':[], 'times':[] } for seq in seq_info.keys() }

    def print_stats(stats, seq=None):
        seqs = ([seq] if seq else list(seq_info.keys()))
        if len(seqs) > 1:
            stats['global'] = { k:sum([ stats[seq][k] for seq in seq_info.keys() ], []) for k in stats[seqs[0]] }
            seqs.append('global')
        
        for seq in seqs:
            print('\n\n' + '*'*10, seq, '*'*10)
            print('*'*24)
            if 'num_patches_events' in stats[seq]: print(f'Avg. number of event patches per frame: {np.mean(stats[seq]["num_patches_events"]):.2f}')
            if 'num_patches_images' in stats[seq]: print(f'Avg. number of image patches per frame: {np.mean(stats[seq]["num_patches_images"]):.2f}')
            if 'times' in stats[seq]: print(f'Avg. processing time per DeltaT: {np.mean(stats[seq]["times"]):.4f} ms')
        
        
elif evaluation_mode == 'complexity':
    data_params['batch_size'] = 1  
    data_params['clip_length_ms'] = 49  # sequence length same as the time-window
    stats = { seq: {} for seq in seq_info.keys() }
    for seq in seq_info.keys():
        stats[seq].update({ 'flops':[], 'params':[] })

    def print_stats(stats, seq=None):
        seqs = ([seq] if seq else list(seq_info.keys()))
        if len(seqs) > 1:
            stats['global'] = { k:sum([ stats[seq][k] for seq in seq_info.keys() ], []) for k in stats[seqs[0]] }
            seqs.append('global')
            
        for seq in seqs:
            print('\n\n' + '*'*10, seq, '*'*10)
            print('*'*24)
            if 'flops' in stats[seq]: print(f'Avg. FLOPs per DeltaT: {np.mean(stats[seq]["flops"])*1e-9:.4f} G')
            if 'params' in stats[seq]: print(f'Avg. params DeltaT: {np.mean(stats[seq]["params"])*1e-6:.4f} M')
            
else:
    raise ValueError('Evaluation model unknown')


        
# Create data loaders, 1 per video recording
dm = MVSEC_DataModule(**data_params, data_params=data_params)
dl_list = dm.val_dataloader()
    
for seq, seq_num in seq_info.items():

    # Get sequence data loader    
    dl = dl_list[seq_num]
    print('\n\n***************', seq_num, seq)

    # Load model
    # TODO: .
    path_weights = path_model + f'/weights/{seq}.ckpt'
    # def get_best_weigths(path_model, metric, mode):
    #     assert mode in ['min', 'max']
    #     mode = max if mode == 'max' else min
    #     w = [ s for s in os.listdir(os.path.join(path_model, 'weights')) if metric in s ]
    #     path_weights = mode(w, key=lambda x: [ float(s[len(metric)+1:len(metric)+1+7]) for s in x.split('-') if s.startswith(metric) ][0])
    #     return os.path.join(path_model, 'weights',path_weights)
    # path_weights = get_best_weigths(path_model, f'{seq}_mean_error_20', 'min')
    print(path_weights)

    model = DepthDecoder(**copy.deepcopy(all_params['model_params'])).to(device)
    state_dict = torch.load(path_weights)['state_dict']
    state_dict = { k.replace('model.', ''):v for k,v in state_dict.items() }            # TODO: .
    model.load_state_dict(state_dict)
    model.eval()

    # Get stats
    for i, batch in enumerate(tqdm(dl)):
        for k in batch.keys():  batch[k] = batch[k].to(device)
    
        if 'event_frames' in batch and batch['event_frames'] is not None:
            batch['event_frames'] = transformations.window_partition( batch['event_frames'], data_params['patch_size'], validation=True,
                           min_activations_per_patch=data_params['min_activations_per_patch'], 
                           drop_token=data_params['drop_token'], 
                                chunk_len_ms=50, maxTime=data_params['MT'], 
                                patch_by_last_k=data_params['patch_by_last_k'],
                            reduce_tokens=True)
        if 'images' in batch and batch['images'] is not None:
            batch['images'] = transformations.window_partition(batch['images'], data_params['patch_size'], validation=True,
                           min_activations_per_patch=data_params['min_activations_per_patch'], 
                           drop_token=data_params['drop_token'], 
                                chunk_len_ms=50, maxTime=data_params['MT'], 
                                patch_by_last_k=False,
                            reduce_tokens=True)
    
        # Calculate activated patches
        if evaluation_mode == 'efficiency':
            stats[seq]['num_patches_events'] += [ float(i) for b in range(batch['event_frames'][0].shape[0]) for i in (batch['event_frames'][0][b].sum(-1) != 0).sum(-1).cpu().detach().tolist() ]
            if 'images' in batch: stats[seq]['num_patches_images'] += [ float(i) for b in range(batch['event_frames'][0].shape[0]) for i in (batch['images'][0][b].sum(-1) != 0).sum(-1).cpu().detach().tolist() ]
        num_time_steps = batch['event_frames'][0].shape[1]

        if evaluation_mode in ['performance', 'efficiency']:
            # Get prediction
            with torch.no_grad():
                t = time.time()
                pred_log = model(batch)[..., None]
                if evaluation_mode == 'efficiency': stats[seq]['times'].append(float((time.time() - t)/num_time_steps))        
            
            if evaluation_mode == 'performance':
                # Calculate stats
                    y = batch['depth']
                    non_nan_values = ~torch.isnan(y)
                    min_dist = data_params['abs_log_params']['min_dist'][0]
                    max_dist = data_params['abs_log_params']['max_dist'][0]
                    y = torch.clip(y, min_dist, max_dist)
                    pred = log_to_abs(pred_log, min_dist=min_dist, max_dist=max_dist)
                
                    logs = {}
                    for d in dists:
                        mask_d = (non_nan_values) & (y < d)
                        stats[seq][f'mean_error_{d}'] += [ float(mean_error(pred[b][mask_d[b]], y[b][mask_d[b]]).cpu().detach().numpy()) for b in range(pred.shape[0]) ]
                        
            if evaluation_mode == 'performance':
                pred.to('cpu'); pred = None; del pred
                y.to('cpu'); y = None; del y
        
        elif evaluation_mode in ['complexity']:
            # Get FLOPs
            macs, params = get_model_complexity_info(model, 
                                        ({'event_data': batch},),
                                      input_constructor=lambda x: x[0],
                                      as_strings=False,
                                      print_per_layer_stat=False, verbose=False)
    
            stats[seq]['flops'].append(float(macs*2/num_time_steps))
            stats[seq]['params'].append(float(params))
            
        else: 
            raise ValueError()

    model.to('cpu'); model = None; del model
    state_dict = None; del state_dict
    del batch
    del dl
    torch.cuda.empty_cache()
        
    print_stats(stats, seq)



print(); print()
print('*'*80)
print('*'*80)
print('*'*80)
print(evaluation_mode)
print_stats(stats)

print()
print(path_model)
print('  &  '.join([ ' & '.join([ f'{np.mean(stats[seq][f"mean_error_{d}"]):.2f}' for d in dists ]) for seq in seq_info.keys() ]))
print(dm.path_dataset)
    
