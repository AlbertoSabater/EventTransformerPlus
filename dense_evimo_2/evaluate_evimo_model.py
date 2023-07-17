#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 17:59:21 2023

@author: asabater
"""

import sys; sys.path.append('..')
# from data_generation import MVSEC_DataModule
import transformations
from trainer import log_to_abs, mean_error
from dense_estimation.EvT_DepthEstimation import DepthDecoder
from data_generation import EVIMO_DataModule


import json
import torch
from tqdm import tqdm
import time
import numpy as np
from ptflops import get_model_complexity_info
import os

import copy
import pandas as pd

import matplotlib.pyplot as plt
import pickle



def load_model_evaluation_hparams(path_model):
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    all_params['model_params']['depth_size'] = 1        # Load just the last depth map for each subsequence
    all_params['data_params']['depth_size'] = 1         # Load just the last depth map for each subsequence
    all_params['data_params']['skip_val_samples'] = 1   # Return all depth_maps
    all_params['data_params']['sample_repetitions'] = 1               # Do not repeat the loaded sequences
    all_params['data_params']['skip_val_samples'] = 1               # Do not skip the loaded sequences
    all_params['data_params']['num_workers'] = 8
    all_params['data_params']['pin_memory'] = False
    return all_params


def plot_training_stats(path_model):
    def load_csv_logs_as_df(path_model):
        log_file = path_model / 'train_log/version_0/metrics.csv'
        logs = pd.read_csv(log_file)
        for i, row in logs[logs.epoch.isna()].iterrows():
            candidates = logs[(~logs.epoch.isna()) & (logs.step >= int(row.step))].epoch.min()
            logs.loc[i, 'epoch'] = candidates
        return logs
    
    logs = load_csv_logs_as_df(path_model)
    # logs = logs[~logs['val_mean_error'].isna()]
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(dpi=200)
    ax.set_ylim(0,250)
    for k in ['val_mean_error', 'train_mean_error']:
        data = logs[~np.isnan(logs[k])]
        plt.plot(data['step'], data[k])
    ax.set_title('/'.join(str(path_model).split('/')[-2:]))
    
    
# %%


def get_mean_error_stats(path_model):
    
    # perf_file = path_model / 'mean_error.pckl'
    # if os.path.isfile(perf_file):
    #     # print('Loading stats')
    #     total_stats = pickle.load(open(perf_file, 'rb'))
    # else:
    
    # def get_best_weigths(path_model, metric, mode):
    #     assert mode in ['min', 'max']
    #     mode = max if mode == 'max' else min
    #     w = [ s for s in (path_model / 'weights').iterdir() if metric in str(s) ]
    #     path_weights = mode(w, key=lambda x: [ float(s[len(metric)+1:len(metric)+1+7]) for s in str(x).split('-') if s.startswith(metric) ][0])
    #     # return os.path.join(path_model, 'weights',path_weights)
    #     return path_weights
    # path_weights = get_best_weigths(path_model, 'val_mean_error', 'min')
    # print(path_weights)
        

    all_params = load_model_evaluation_hparams(str(path_model))
    all_params['data_params']['batch_size'] = 16
    data_params = all_params['data_params']
        


    total_stats = {}
    for mode in ['imo', 'imo_ll',  'sfm', 'sfm_ll']:

        model = DepthDecoder(**copy.deepcopy(all_params['model_params'])).to(device)
        path_weights = path_model / f'weigths/{mode}.ckpt'
        print(path_weights)
        state_dict = torch.load(path_weights)['state_dict']
        state_dict = { k.replace('model.', ''):v for k,v in state_dict.items() }            # TODO: .
        model.load_state_dict(state_dict)
        model.eval()

        data_params['side'] = [mode]
        dm = EVIMO_DataModule(**data_params, data_params=data_params)
        dl = dm.val_dataloader()
        
        total_stats[mode] = []
        
        stats = {'mean_error': []}
        for i, batch in enumerate(tqdm(dl)):
            for k in batch.keys():  batch[k] = batch[k].to(device)
        
            batch['event_frames'] = transformations.window_partition( batch['event_frames'], data_params['patch_size'], validation=True,
                           min_activations_per_patch=data_params['min_activations_per_patch'], 
                           drop_token=data_params['drop_token'], 
                                chunk_len_ms=1000/60, maxTime=data_params['MT'], 
                                patch_by_last_k=data_params['patch_by_last_k'],
                            reduce_tokens=True)

        
            with torch.no_grad():
                t = time.time()
                pred_log = model(batch)[..., None]
    
                y = batch['depth']
                non_nan_values = ~torch.isnan(y)
                min_dist = data_params['abs_log_params']['min_dist']
                max_dist = data_params['abs_log_params']['max_dist']
                y = torch.clip(y, min_dist, max_dist)
                pred = log_to_abs(pred_log, min_dist=min_dist, max_dist=max_dist)
            
                total_stats[mode] += [ float(mean_error(pred[b][non_nan_values[b]], y[b][non_nan_values[b]]).cpu().detach().numpy()) for b in range(pred.shape[0]) ]

                
                
        model.to('cpu'); model = None; del model
        state_dict = None; del state_dict
        del batch
        del dl
        torch.cuda.empty_cache()
            
        
        print(f'{mode:<6} |  {np.mean(total_stats[mode]):.2f}')    
        
        # pickle.dump(total_stats, open(perf_file, 'wb'))
    return total_stats
        
    


# %%

if __name__ == '__main__':
    
    from pathlib import Path

    # device = 'cpu'
    device = 'cuda'
    
    
    path_model = Path('../trained_models/dense_evimo_2/eve/')
    total_stats = get_mean_error_stats(path_model)
    print(f'{path_model.parent.stem:<9}/{path_model.stem} ||', f'all: {np.mean(sum(total_stats.values(), [])):.2f} |', ' | '.join([ f'{k}: {np.mean(v):.2f}' for k,v in total_stats.items() ]))    




