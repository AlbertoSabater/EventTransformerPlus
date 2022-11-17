import torch
# from data_generation import Event_DataModule

# from pytorch_lightning.metrics import Accuracy
# from torchmetrics.functional import accuracy

from sklearn.metrics import confusion_matrix
import pandas as pd
import json
from tqdm import tqdm
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# from v1_tore_oldNet_newDataLoader.data_generation import Stream_CLF_DataModule
# from v1_tore_oldNet_newDataLoader.data_augmentation import DataAugmentation




def get_best_weigths(path_model, metric, mode):
    assert mode in ['min', 'max']
    mode = max if mode == 'max' else min
    w = [ s for s in os.listdir(os.path.join(path_model, 'weights')) if metric in s ]
    path_weights = mode(w, key=lambda x: [ float(s[len(metric)+1:len(metric)+1+7]) for s in x.split('-') if s.startswith(metric) ][0])
    return os.path.join(path_model, 'weights',path_weights)


def load_csv_logs_as_df(path_model):
    log_file = path_model + '/train_log/version_0/metrics.csv'
    logs = pd.read_csv(log_file)
    logs.loc[logs.epoch.isnull(), 'epoch'] = np.arange(len(logs.loc[logs.epoch.isnull(), 'epoch']))     # np.arange(logs.epoch.max()+1)
    logs = logs.groupby('epoch', as_index=False).max()
    return logs
    
def plot_training_evolution(path_model, plot_metric='mean_error'):

    logs = load_csv_logs_as_df(path_model)
    
    lr_col = [ c for c in logs.columns if 'lr' in c ][0]
    lr = logs[~logs[lr_col].isna()][lr_col]
    # cols = [ c for c in logs.columns if c and '_exp' in c ]
    # cols = [ c for c in logs.columns if plot_metric in c ]
    # cols = [ c for c in logs.columns if c.endswith(plot_metric) ]

    fig, ax1 = plt.subplots(figsize=(12,7), dpi=200)
    ax2 = ax1.twinx()
    # ax3 = ax1.twinx()

    # for c in [ c for c in logs.columns if 'val_' in c and 'acc' not in c ]:
        # loss = logs[~logs[c].isna()][c]
        # ax1.plot(loss.values, label=c)
    # for c in cols:
    ax1.plot(logs.loc[:,'val_acc'], label='val_acc')

        
    # ax2.plot(val_acc.values, 'g')
    ax2.plot(lr.values, 'r')
    
    # if 'ASL' in path_model: ax2.set_ylim(0.95, 1)  # Acc lims
    # else: ax2.set_ylim(0.5, 1)  # Acc lims

    ax1.set_ylabel('losses', fontsize=18)   # , color='b'
    ax2.set_ylabel('lr', color='r', fontsize=18)
    
    ax1.set_ylim(0.5*100, 1.0*100)

    # ax2.hlines(val_acc.max(), 0, len(val_acc.values), color='g', linestyle='--', alpha=0.7)
    # ax1.hlines(logs[~logs['val_loss_total'].isna()]['val_loss_total'].min(), 0, len(val_acc.values), color='y', linestyle='--', alpha=0.7)

    # ax3.spines['right'].set_position(('outward', 60))
    fig.tight_layout()

    _,stats,_ = get_evaluation_results(path_model)
    # print(stats)

    plt.title(f"{' | '.join(path_model.split('/')[-3:])}\n" +\
              f' val_acc {logs["val_acc"].max():.4f}' +\
              # f' | min_lr {logs["lr-AdamW"].iloc[-1]}'
              f' | FLOPs {stats["flops"] if "flops" in stats else -1:.3f} G' +\
              f' | #patches {stats["avg_patches"] if "avg_patches" in stats else -1:.0f}'
              f' | #params {stats["params"] if "params" in stats else -1:.3f}'
                  , fontsize=16)
    # plt.title('{} | Acc.: {:.4f} | Loss: {:.4f}'.format(' | '.join(path_model.split('/')[-3:]), val_acc.max(), 
    #           logs[~logs['val_loss_total'].isna()]['val_loss_total'].min()), fontsize=16)
    ax1.legend()
    
    plt.show()


def get_evaluation_results(path_model, force=True, device='cpu'):  # , path_weights, skip_validation=False,
    
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    stats_filename = path_model + '/stats_validation.json'
    cm_filename = path_model + '/confussion_matrix.pckl'
    if not force and os.path.isfile(stats_filename):
        df_cm = pickle.load(open(cm_filename, 'rb')) if os.path.isfile(cm_filename) else None
        return all_params, json.load(open(stats_filename, 'r')),df_cm
    
    stats_filename = path_model +'/stats.json'
    if os.path.isfile(stats_filename): stats = json.load(open(stats_filename, 'r'))
    else: stats = {}
    # print(stats)
    logs = load_csv_logs_as_df(path_model)
    # cols = [ c for c in logs.columns if '_exp' in c ]
    # cols = logs.columns
    # for c in cols: stats[c] = logs[c].min()
    stats['val_acc'] = logs['val_acc'].max()
    stats['val_loss_total'] = logs['val_loss_total'].min()
    stats['num_epochs'] = int(logs['epoch'].max())
    stats['lr0'] = logs['lr-AdamW'][0]
    stats['min_lr'] = logs['lr-AdamW'].min()
    stats['max_lr'] = logs['lr-AdamW'].max()
    # data_params = all_params['data_params']
        
        # def_val = 0.0
        # stats['validation_val_acc'] = def_val
        # stats['sequence_ms'] = def_val
        # stats['chunk_ms'] = def_val
        # stats['events_per_chunk'] = def_val
        # stats['ms/ms'] = def_val
    df_cm = None

        
    return all_params, stats, df_cm

def get_evaluation_results_depth(path_model, force=True, device='cpu'):  # , path_weights, skip_validation=False,
    
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    stats_filename = path_model + '/stats_validation.json'
    cm_filename = path_model + '/confussion_matrix.pckl'
    if not force and os.path.isfile(stats_filename):
        df_cm = pickle.load(open(cm_filename, 'rb')) if os.path.isfile(cm_filename) else None
        return all_params, json.load(open(stats_filename, 'r')),df_cm
    
    stats = {}
    logs = load_csv_logs_as_df(path_model)
    # cols = [ c for c in logs.columns if '_exp' in c ]
    cols = logs.columns
    # cols = logs.columns
    # for c in cols: stats[c] = logs[c].min()
    for c in cols:
        if 'delta' not in c: stats[c] = logs[c].min()
        else: stats[c] = logs[c].max()
    # stats['val_acc'] = logs['val_acc'].max()
    # stats['val_loss_total'] = logs['val_loss_total'].min()
    stats['num_epochs'] = int(logs['epoch'].max())
    # data_params = all_params['data_params']
        
        # def_val = 0.0
        # stats['validation_val_acc'] = def_val
        # stats['sequence_ms'] = def_val
        # stats['chunk_ms'] = def_val
        # stats['events_per_chunk'] = def_val
        # stats['ms/ms'] = def_val
    df_cm = None

        
    return all_params, stats, df_cm


def get_time_accuracy(path_model, device):
    from torchmetrics import Accuracy
    from v1_tore_oldNet_newDataLoader import transformations
    from v4_tore_newNet_newDataset_newDataLoader.trainer_v4 import EvNetModel


    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    all_params['model_params']['skip_conn_backbone'] = all_params['model_params'].get('skip_conn_backbone', False)

    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Stream_CLF_DataModule(**data_params, data_params=data_params)
    dl = dm.val_dataloader()
    
    path_weights = get_best_weigths(path_model, 'val_acc', 'max')
    model = EvNetModel.load_from_checkpoint(path_weights, **all_params, map_location=torch.device('cpu')).eval().to(device)
    data_aug = DataAugmentation(**data_params)

    total_time = []
    preds, gt = [], []
    # for polarity, pixels, labels in tqdm(dl):
    for batch in tqdm(dl):
        # polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        batch['event_frames'] = transformations.window_partition(batch['event_frames'], data_params['patch_size'], validation=True,
                       min_activations_per_patch=data_params['min_activations_per_patch'], 
                       drop_token=data_params['drop_token'], 
                        chunk_len_ms=data_params['chunk_len_ms'], maxTime=data_params['MT'], 
                        patch_by_last_k=data_params['patch_by_last_k'],
                       reduce_tokens=True)
        if batch['event_frames'][0].sum() == 0: continue
        if data_params.get('patches_to_vit', False): batch['event_frames'] = (data_aug.patches_to_ViT_format(batch['event_frames'][0]), data_aug.random_shift(batch['event_frames'][1]))
 
        inds = batch['event_frames'][0][0].sum(-1).sum(-1) != 0
        if not inds.all(): print('Skipping step')
        batch['event_frames'] = (batch['event_frames'][0][:,inds], batch['event_frames'][1][:,inds])
    
        # print('\n', batch['event_frames'][0].shape, batch['event_frames'][0].sum(0).sum(-1).sum(-1))
        
        t = time.time()
        label = model.backbone(batch)
        total_time.append((time.time() - t)/batch['event_frames'][0].shape[1])
        preds.append(label.argmax())
        gt.append(batch['labels'][0])
        # break
    
    acc = Accuracy()(torch.stack(preds), torch.stack(gt))
    time_per_step = np.mean(total_time)*1000
    
    return acc, time_per_step


def get_flops(model, path_model):
    # https://github.com/sovrasov/flops-counter.pytorch
    from ptflops import get_model_complexity_info
    from v1_tore_oldNet_newDataLoader import transformations

    # TODO: calculate avg num of activated patches
    
    all_params = json.load(open(path_model + '/all_params.json', 'r'))
    all_params['model_params']['skip_conn_backbone'] = all_params['model_params'].get('skip_conn_backbone', False)

    data_params = all_params['data_params']
    data_params['batch_size'] = 1
    data_params['pin_memory'] = False
    data_params['sample_repetitions'] = 1
    dm = Stream_CLF_DataModule(**data_params, data_params=data_params)
    dl = dm.val_dataloader()
    
    
    total_flops, total_macs, total_params, total_act_patches = [], [], [], []
    total_time_flops = []
    for batch in tqdm(dl):
        # polarity, pixels, labels = polarity.to(device), pixels.to(device), labels.to(device)
        
        batch['event_frames'] = transformations.window_partition(batch['event_frames'], data_params['patch_size'], validation=True,
                       min_activations_per_patch=data_params['min_activations_per_patch'], 
                       drop_token=data_params['drop_token'], 
                        chunk_len_ms=data_params['chunk_len_ms'], maxTime=data_params['MT'], 
                        patch_by_last_k=data_params['patch_by_last_k'],
                       reduce_tokens=True)
        
        inds = batch['event_frames'][0][0].sum(-1).sum(-1) != 0
        if not inds.all(): print('Skipping step')
        batch['event_frames'] = (batch['event_frames'][0][:,inds], batch['event_frames'][1][:,inds])
    
        events, pixels = batch['event_frames']
        num_time_steps = events.shape[1]
        if batch['event_frames'][0].sum() == 0: continue
            
        t = time.time()
        macs, params = get_model_complexity_info(model.backbone, 
                                           ({'event_data': batch},),
                                         input_constructor=lambda x: x[0],
                                         as_strings=False,
                                         print_per_layer_stat=False, verbose=False)
        total_time_flops.append(time.time() - t)
        # params = float(params.split()[0])
        # # macs = float(macs.split()[0]) / polarity.shape[0]
        # macs = float(macs.split()[0])
        flops = 2*macs
        # total_act_patches.append(num_patches.cpu())
        total_flops.append(flops/num_time_steps); total_macs.append(macs/num_time_steps); total_params.append(params)
        total_act_patches.append((batch['event_frames'][0].sum(-1) != 0).sum(-1).cpu())

    total_act_patches = torch.mean(torch.cat(total_act_patches, axis=1).float())
    
    return np.mean(total_flops)*1e-9, np.mean(total_macs)*1e-9, np.mean(total_params)*1e-6, total_act_patches, np.mean(total_time_flops)





if __name__ == '__main__':
    
    # device = 'cuda:0'
    device = 'cpu'

    # path_model = '/mnt/hdd/ml_results/EvT_v2_depth/001_/MVSEC/001/0228_2208_model_0'
    path_model = '/mnt/hdd/ml_results/EvT_v2_depth/001_/MVSEC/001/'
    path_model += os.listdir(path_model)[0]
    

    path_weights = get_best_weigths(path_model, 'N3_mean_error', 'min')
    
    logs = load_csv_logs_as_df(path_model)
    plot_training_evolution(path_model)

    # all_params, stats, df_cm = get_evaluation_results(path_model, path_weights, force=True, device=device)
    # print(path_weights)
    # print(' * Accuracy: {:.4f} | {:.4f}'.format(stats['training_val_acc'], stats['validation_val_acc']))


