import json
import numpy as np
import copy

from torch.optim import AdamW
import torch

from pytorch_lightning import Trainer, LightningModule
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from data_generation import EVIMO_DataModule

import sys
sys.path.append('..')
import transformations

from dense_estimation.EvT_DepthEstimation import DepthDecoder
from dense_estimation.data_augmentation import DataAugmentation

import training_utils

from kornia.filters.sobel import spatial_gradient




def abs_to_log(x, min_dist, max_dist):
    x = torch.clip(x, min_dist, max_dist)
    x = torch.log(x) - float(np.log(min_dist))
    x = x / float(np.log(max_dist) - np.log(min_dist))
    return x
    
def log_to_abs(x, min_dist, max_dist):
    x = x * float(np.log(max_dist) - np.log(min_dist))
    x = torch.exp(x) * min_dist
    x = torch.clip(x, min_dist, max_dist)
    return x
    

def mean_error(y_input, y_target):
    abs_diff = torch.abs(y_target-y_input)
    return abs_diff.mean()

def scale_invariant_loss(y_input, y_target, weight = 1.0, n_lambda = 0.5):
    log_diff = y_input - y_target
    v =  weight * ((log_diff**2).mean()-(n_lambda*(log_diff.mean())**2))
    return v

class MultiScaleGradient(torch.nn.Module):
    def __init__(self, start_scale = 1, num_scales = 4):
        super(MultiScaleGradient,self).__init__()
        self.start_scale = start_scale
        self.num_scales = num_scales
        self.multi_scales = [torch.nn.AvgPool2d(self.start_scale * (2**scale), self.start_scale * (2**scale)) for scale in range(self.num_scales)]

    def forward(self, prediction, target):
        prediction, target = prediction[...,0], target[...,0]
        loss_value = 0
        diff = prediction - target
        _,_,H,W = target.shape
        for m in self.multi_scales:
            # input and type are of the type [B x C x H x W]
            # Use kornia spatial gradient computation
            delta_diff = spatial_gradient(m(diff))
            is_nan = torch.isnan(delta_diff)
            is_not_nan_sum = (~is_nan).sum()
            # output of kornia spatial gradient is [B x C x 2 x H x W]
            loss_value += torch.abs(delta_diff[~is_nan]).sum()/is_not_nan_sum*target.shape[0]*2
        return (loss_value/self.num_scales)
multi_scale_grad_loss_fn = MultiScaleGradient()



class EvNetModel(LightningModule):

    def __init__(self, model_params, data_params, optim_params):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_params = model_params
        self.data_params = data_params
        self.optim_params = optim_params

        self.model = DepthDecoder(**model_params)
        self.data_aug = DataAugmentation(**data_params)


    def forward(self, x):
        pred_depth = self.model(x)
        return pred_depth

    def configure_optimizers(self):
        # Import base optimizer
        base_optim = AdamW
        optim = base_optim(self.parameters(), **self.optim_params['optim_params'])
        if 'scheduler' in self.optim_params: 
            if self.optim_params['scheduler']['name'] == 'lr_on_plateau': 
                sched = lr_scheduler.ReduceLROnPlateau(optim, **self.optim_params['scheduler']['params'])
            elif self.optim_params['scheduler']['name'] == 'one_cycle_lr': 
                sched = lr_scheduler.OneCycleLR(optim, max_lr=self.optim_params['optim_params']['lr'],  **self.optim_params['scheduler']['params'])
            return {'optimizer': optim, 'lr_scheduler': sched}  # , 'monitor': self.optim_params['scheduler']['monitor']
        return optim
    
    
    def step(self, batch, min_dist, max_dist):
        pred_log = self(batch)[..., None]     # Add channel dimension (1)
        
        y = batch['depth']
        non_nan_values = ~torch.isnan(y)
        y = torch.clip(y, min_dist, max_dist)

        logs = {}
        
        """ depth to log -> evaluate losses """
        y_log = abs_to_log(y, min_dist=min_dist, max_dist=max_dist)
        logs['scale_invariant_log_loss'] =  scale_invariant_loss(pred_log[non_nan_values], y_log[non_nan_values])
        logs['multi_scale_grad_loss_fn'] =  multi_scale_grad_loss_fn(pred_log, y_log)
        logs['loss'] = sum([ w*logs[l] for w,l in self.optim_params['loss_monitor'] ])
        
        """ pred to abs -> evaluate metrics [abs_rel, squ_rel, rms_linear, scale_invariant, mean_depth_error, mean_diff]
                        -> evaluate by threshold [10,20,30] """
                        
        pred = log_to_abs(pred_log, min_dist=min_dist, max_dist=max_dist)
        logs['mean_error'] =            mean_error(pred[non_nan_values], y[non_nan_values])
            
        return logs
    

    def training_step(self, batch, batch_idx):
        
        batches = batch
        total_loss = []
        batch = self.data_aug(batch)

        batch['event_frames'] = transformations.window_partition( batch['event_frames'], self.data_params['patch_size'], validation=False,
                       min_activations_per_patch=self.data_params['min_activations_per_patch'], 
                       drop_token=self.data_params['drop_token'], 
                        chunk_len_ms=1000/60, maxTime=self.data_params['MT'], # TODO: . 
                        patch_by_last_k=self.data_params['patch_by_last_k'],
                        reduce_tokens=True)

            
        # Shift
        batch = self.data_aug.random_shift(batch)
        
        losses = self.step(batch, min_dist=self.data_params['abs_log_params']['min_dist'], max_dist=self.data_params['abs_log_params']['max_dist'])
        for k,v in losses.items():
            self.log(f'train_{k}', v, prog_bar=True, logger=True, sync_dist=True)

        total_loss.append(losses['loss'])
        return torch.stack(total_loss).mean()

    # def validation_step(self, batch, batch_idx, dataloader_idx=-1, *args):
    def validation_step(self, batch, batch_idx, *args):
        batch['event_frames'] = transformations.window_partition( batch['event_frames'], self.data_params['patch_size'], validation=True,
                       min_activations_per_patch=self.data_params['min_activations_per_patch'], 
                       drop_token=self.data_params['drop_token'], 
                            chunk_len_ms=1000/60, maxTime=self.data_params['MT'], 
                            patch_by_last_k=self.data_params['patch_by_last_k'],
                        reduce_tokens=True)

        losses = self.step(batch, min_dist=self.data_params['abs_log_params']['min_dist'], max_dist=self.data_params['abs_log_params']['max_dist'])
        for k,v in losses.items():
            if any([ s == k for s in ['mean_error', 'loss'] ]):
                self.log(f'val_{k}', v, prog_bar=True, logger=True, sync_dist=True)
            else:
                self.log(f'val_{k}', v, prog_bar=False, logger=True, sync_dist=True)

        return losses['loss']
    


def train(folder_name, path_results, data_params, model_params,
          training_params, optim_params, callbacks_params, logger_params):

    path_model = training_utils.create_model_folder(path_results, folder_name)
    
    callbacks = []
    for k, params in callbacks_params:
        if k == 'early_stopping': callbacks.append(EarlyStopping(**params))
        if k == 'lr_monitor': callbacks.append(LearningRateMonitor(**params))
        if k == 'model_chck': 
            params['dirpath'] = params['dirpath'].format(path_model)
            callbacks.append(ModelCheckpoint(**params))
        
    loggers = []
    if 'csv' in logger_params: 
        logger_params['csv']['save_dir'] = logger_params['csv']['save_dir'].format(path_model)
        loggers.append(CSVLogger(**logger_params['csv']))

    
# =============================================================================
# Train
# =============================================================================
    data_params['depth_size'] = model_params['depth_size']
    
    dm = EVIMO_DataModule(data_params=data_params, **data_params)
    
    model_params['events_dim'] = dm.pixel_dim*data_params['patch_size']*data_params['patch_size']
    model_params['images_dim'] = data_params['patch_size']*data_params['patch_size']
    if optim_params['scheduler']['name'] == 'one_cycle_lr': 
        print(f" - Setting max_epochs to [{optim_params['scheduler']['params']['epochs']}], according to [one_cycle_lr]")
        training_params['max_epochs'] = optim_params['scheduler']['params']['epochs']
    model_params['height'], model_params['width'] = dm.height, dm.width
    
    model_params['use_events'] = data_params['use_events']
    model_params['use_images'] = data_params['use_images']
    model_params['patch_size'] = data_params['patch_size']
    
    data_params['abs_log_params'] = {'min_dist': dm.min_dist, 'max_dist': dm.max_dist}
    

    
    if 'pos_encoding' in model_params and model_params['pos_encoding']['params'].get('shape', -1) == -1:
        model_params['pos_encoding']['params']['shape'] = (dm.height, dm.width)
        print(f" - Setting positional_encoding shape to [{(dm.height, dm.width)}]")
    if model_params['downsample_pos_enc'] == -1: 
        print(f" - Setting downsample_pos_enc to [{data_params['patch_size']}], according to [patch_size]")
        model_params['downsample_pos_enc'] = data_params['patch_size']
    
    if optim_params['scheduler']['name'] == 'one_cycle_lr':
        optim_params['scheduler']['params']['steps_per_epoch'] = 1
    
    model = EvNetModel(model_params=copy.deepcopy(model_params), 
                       data_params = copy.deepcopy(data_params),
                       optim_params=copy.deepcopy(optim_params),
                       )
    
    trainer = Trainer(**training_params, callbacks=callbacks, logger=loggers,
                      )

    
    # Save all params
    json.dump({'data_params': data_params, 'model_params': model_params,  'training_params': training_params,
               'optim_params': optim_params, 'callbacks_params': callbacks_params, 'logger_params': logger_params},
              open(path_model+'all_params.json', 'w'))
    
    
    trainer.fit(model, dm)
    trainer.save_checkpoint(path_model+"/weights/final_weights.ckpt")
    
    print(' ** Train finished:', path_model)

    return path_model



if __name__ == '__main__':
    
    path_results = '../trained_models/new/'
        
    reference_model = '../trained_models/dense_models/evimo_eve/'
    train_params = json.load(open(reference_model + '/all_params.json', 'r'))
        
    folder_name = 'evimo_eve_test/'
        
    train(folder_name, path_results, **train_params)


