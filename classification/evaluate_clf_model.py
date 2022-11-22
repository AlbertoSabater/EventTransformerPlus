import torch
import json
import time
import copy
from tqdm import tqdm
import numpy as np

from EvT_CLF import EvT_CLF
from data_generation import Stream_CLF_DataModule
import transformations

from ptflops import get_model_complexity_info
from sklearn.metrics import accuracy_score

import sys; sys.path.append('..')



# # 99.24 
# path_model = '../trained_models/clf_models/DVS128/'

# # 97.57
# path_model = '../trained_models/clf_models/DVS128_11CLS/'

# # 92.34
# path_model = '../trained_models/clf_models/SL_Animals_3Sets/'

# # 94.39
path_model = '../trained_models/clf_models/SL_Animals_4Sets/'



device = 'cuda'
# device = 'cpu'

"""
performance -> mean_error: per subsequence
complexity: -> FLOPs: per time-window, modified predictions
efficeincy: -> time: per time-window
"""
evaluation_mode = 'performance'               # accuracy | Increase batch_size, faster execution
# evaluation_mode = 'efficiency'              # time / DeltaT | batch_size = 1, calcualte eff. stats just for a single time-window (50ms)
# evaluation_mode = 'complexity'              # FLOPs / DeltaT | batch_size = 1, calcualte eff. stats just for a single time-window (50ms)


        
def load_model_evaluation_hparams(path_model):
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    all_params['data_params']['sample_repetitions'] = 1               # Do not repeat the loaded sequences
    all_params['data_params']['num_workers'] = 8
    all_params['data_params']['pin_memory'] = False
    return all_params

all_params = load_model_evaluation_hparams(path_model)
data_params = all_params['data_params']


if evaluation_mode == 'performance':
    data_params['batch_size'] = 16 
    data_params['clip_length_ms'] = 1298 if 'DVS128' in data_params['dataset_name'] else 1792
    stats = {'acc': None}
    def print_stats(stats):
        print('\n\n', '*'*24)
        if 'acc' in stats: print(f'Accuracy: {stats["acc"]*100:.2f}')

elif evaluation_mode == 'efficiency':   # Time to process a single time-window
    data_params['batch_size'] = 1 
    data_params['clip_length_ms'] = data_params['chunk_len_ms']-1  # process a single time-window
    stats = {'num_patches_events':[], 'times':[]}
    def print_stats(stats):
        print('\n\n', '*'*24)
        if 'num_patches_events' in stats: print(f'Avg. number of event patches per frame: {np.mean(stats["num_patches_events"]):.2f}')
        if 'times' in stats: print(f'Avg. processing time per DeltaT: {np.mean(stats["times"])*1000:.4f} ms')

elif evaluation_mode == 'complexity':
    data_params['batch_size'] = 1  
    data_params['clip_length_ms'] = data_params['chunk_len_ms']-1  # process a single time-window
    stats = { 'flops':[], 'params':[] }
    def print_stats(stats):
        print('\n\n', '*'*24)
        if 'num_patches_events' in stats: print(f'Avg. number of event patches per frame: {np.mean(stats["num_patches_events"]):.2f}')
        if 'flops' in stats: print(f'Avg. FLOPs per DeltaT: {np.mean(stats["flops"])*1e-9:.4f} G')
        if 'params' in stats: print(f'Avg. params DeltaT: {np.mean(stats["params"])*1e-6:.4f} M')
      
else:
    raise ValueError('Evaluation model unknown')



# Load model and weights
path_weights = path_model + '/weights/model_weights.ckpt'
model = EvT_CLF(**copy.deepcopy(all_params['model_params'])).eval().to(device)
state_dict = torch.load(path_weights)['state_dict']
model.load_state_dict(state_dict)
state_dict = None; del state_dict

# Load Data Loader
dm = Stream_CLF_DataModule(**data_params, data_params=data_params)
dl = dm.val_dataloader()


preds, gt = [], []
for i, batch in enumerate(tqdm(dl)):
    for k in batch.keys():  batch[k] = batch[k].to(device)

    batch['event_frames'] = transformations.window_partition(batch['event_frames'], data_params['patch_size'], validation=True,
                   min_activations_per_patch=data_params['min_activations_per_patch'], 
                   drop_token=data_params['drop_token'], 
                    chunk_len_ms=data_params['chunk_len_ms'], maxTime=data_params['MT'], 
                    patch_by_last_k=data_params['patch_by_last_k'],
                   reduce_tokens=True)

    num_time_steps = batch['event_frames'][0].shape[1]

    if evaluation_mode in ['performance', 'efficiency']:
        # Get prediction
        with torch.no_grad():
            t = time.time()
            label = model(batch)
            
            if evaluation_mode == 'efficiency':
                stats['times'].append(time.time() - t)
                # Calculate activated patches
                stats['num_patches_events'] += [ float(i) for b in range(batch['event_frames'][0].shape[0]) for i in (batch['event_frames'][0][b].sum(-1) != 0).sum(-1).cpu().detach().tolist() ]
            
            preds += label.argmax(1)
            gt += batch['labels']
                
    
    elif evaluation_mode in ['complexity']:
        print(num_time_steps)
        inds = batch['event_frames'][0][0].sum(-1).sum(-1) != 0
        if num_time_steps == 0: 
            print('No event information, continue'); continue
        if not inds.all(): 
            print('Skipping step'); continue
        batch['event_frames'] = (batch['event_frames'][0][:,inds], batch['event_frames'][1][:,inds])
        
        events, pixels = batch['event_frames']
        num_time_steps = events.shape[1]
            
        macs, params = get_model_complexity_info(
                                        model, 
                                           ({'event_data': batch},),
                                         input_constructor=lambda x: x[0],
                                         as_strings=False,
                                         print_per_layer_stat=False, verbose=False)
        flops = 2*macs
        stats['flops'].append(flops/num_time_steps); stats['params'].append(params)

    else: 
        raise ValueError()

if evaluation_mode == 'performance':
    stats['acc'] = accuracy_score(torch.stack(preds).cpu().numpy(), torch.stack(gt).cpu().numpy())



model.to('cpu'); model = None; del model
del batch
del dl
torch.cuda.empty_cache()
    
print_stats(stats)

