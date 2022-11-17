from torch import nn
import torch
from pytorch_lightning import Trainer, LightningModule
from torch.optim import lr_scheduler

import sys
sys.path.append('..')
from data_generation import Stream_CLF_DataModule

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import training_utils
import json
from torch.optim import AdamW
import torchmetrics

from EvT_CLF import EvT_CLF

from data_augmentation import DataAugmentation

import sys
sys.path.append('..')
import transformations


class EvNetModel(LightningModule):

    def __init__(self, model_params, data_params, optim_params, loss_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.model_params = model_params
        self.optim_params = optim_params
        self.data_params = data_params
        
        # Initialize Backbone
        self.backbone = EvT_CLF(**model_params)
        
        self.loss_weights = loss_weights
        self.init_optimizers()
        self.data_aug = DataAugmentation(**data_params)
        
        
    def init_optimizers(self):
        if self.optim_params['loss']['name'] == 'ce':
            self.criterion = nn.CrossEntropyLoss(weight = self.loss_weights, **self.optim_params['loss']['params'])
        elif self.optim_params['loss']['name'] == 'nll':
            if 'label_smoothing' in self.optim_params['loss']['params']: del self.optim_params['loss']['params']['label_smoothing']
            self.criterion = nn.NLLLoss(weight = self.loss_weights, **self.optim_params['loss']['params'])
        else: raise ValueError('Loss not handled')
        
        if self.optim_params['use_ols']:
            from online_label_smoothing import OnlineLabelSmoothing
            self.criterion = OnlineLabelSmoothing(hard_loss=self.criterion, alpha=0.5, n_classes=self.model_params['decoder_params']['opt_classes'], smoothing=0.1)
        
        self.accuracy = torchmetrics.Accuracy()
        

    def forward(self, batch):
        # Get updated latent vectors
        clf_logits = self.backbone(batch)
        return clf_logits
    
        
    def configure_optimizers(self):
        # Import base optimizer
        base_optim = AdamW
        optim = base_optim(self.parameters(), **self.optim_params['optim_params'])
        if 'scheduler' in self.optim_params: 
            if self.optim_params['scheduler']['name'] == 'lr_on_plateau': 
                sched = lr_scheduler.ReduceLROnPlateau(optim, **self.optim_params['scheduler']['params'])
            elif self.optim_params['scheduler']['name'] == 'one_cycle_lr': 
                sched = lr_scheduler.OneCycleLR(optim, max_lr=self.optim_params['optim_params']['lr'],  **self.optim_params['scheduler']['params'])
            return {'optimizer': optim, 'lr_scheduler': sched}      # , 'monitor': self.optim_params['monitor']
        return optim
    
    
    # Forward data and calculate loss and acc
    def step(self, batch):
        y = batch['labels']
        clf_logits = self(batch)
        loss_clf = 0.0
        logs = {}
        loss_clf = self.criterion(clf_logits, y)
        preds = torch.argmax(clf_logits, dim=-1)
        acc = self.accuracy(preds, y)
        logs['loss_clf'] = loss_clf
        logs['acc'] = acc
        logs['loss_total'] = loss_clf
        return logs
        

    def training_step(self, batch, batch_idx):
        batch = self.data_aug(batch)
        batch['event_frames'] = transformations.window_partition( batch['event_frames'], self.data_params['patch_size'], validation=False,
                       min_activations_per_patch=self.data_params['min_activations_per_patch'], 
                       drop_token=self.data_params['drop_token'], 
                        chunk_len_ms=self.data_params['chunk_len_ms'], maxTime=self.data_params['MT'], 
                        patch_by_last_k=self.data_params['patch_by_last_k'],
                       reduce_tokens=True)
        batch['event_frames'] = (batch['event_frames'][0], self.data_aug.random_shift(batch['event_frames'][1]))
        losses = self.step(batch)
        for k,v in losses.items():
            self.log(f'train_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return losses['loss_total']


    def validation_step(self, batch, batch_idx):
        batch['event_frames'] = transformations.window_partition( batch['event_frames'], self.data_params['patch_size'], validation=True,
                       min_activations_per_patch=self.data_params['min_activations_per_patch'], 
                       drop_token=self.data_params['drop_token'], 
                        chunk_len_ms=self.data_params['chunk_len_ms'], maxTime=self.data_params['MT'], 
                        patch_by_last_k=self.data_params['patch_by_last_k'],
                       reduce_tokens=True)
        losses = self.step(batch)
        for k,v in losses.items():
            self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        
        return losses['loss_total']



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
    dm = Stream_CLF_DataModule(**data_params, data_params=data_params)
    model_params['events_dim'] = dm.pixel_dim*data_params['patch_size']*data_params['patch_size']
    model_params['decoder_params']['opt_classes'] = dm.num_classes
    model_params['decoder_params']['softmax'] = optim_params['loss']['name'] == 'nll'
    
    if optim_params['scheduler']['name'] == 'one_cycle_lr': 
        print(f" - Setting max_epochs to [{optim_params['scheduler']['params']['epochs']}], according to [one_cycle_lr]")
        training_params['max_epochs'] = optim_params['scheduler']['params']['epochs']

    
    if 'pos_encoding' in model_params and model_params['pos_encoding']['params'].get('shape', -1) == -1:
        model_params['pos_encoding']['params']['shape'] = (dm.height, dm.width)
        print(f" - Setting positional_encoding shape to [{(dm.height, dm.width)}]")
    if model_params['downsample_pos_enc'] == -1: 
        print(f" - Setting downsample_pos_enc to [{data_params['patch_size']}], according to [patch_size]")
        model_params['downsample_pos_enc'] = data_params['patch_size']
    
    if optim_params['scheduler']['name'] == 'one_cycle_lr':
        optim_params['scheduler']['params']['steps_per_epoch'] = 1
        
        
    model = EvNetModel(model_params, data_params, optim_params)

    trainer = Trainer(**training_params, callbacks=callbacks, logger=loggers)
    
    # Save all params
    json.dump({'data_params': data_params, 'model_params': model_params, 'training_params': training_params,
               'optim_params': optim_params, 'callbacks_params': callbacks_params, 'logger_params': logger_params},
              open(path_model+'all_params.json', 'w'))
    
    
    trainer.fit(model, dm)
    
    print(' ** Train finished:', path_model)

    sys.stdout.flush()
    return path_model




if __name__ == '__main__':
    
    reference_model = '../trained_models/clf_models/DVS128/'
    train_params = json.load(open(reference_model + '/all_params.json', 'r'))
    
    train_params['data_params']['batch_size'] = 48
    train_params['data_params']['num_workers'] = 4
    
    path_results = '../trained_models/'
    
    train('', path_results, **train_params)
    
    
    
    
