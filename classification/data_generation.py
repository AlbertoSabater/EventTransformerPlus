from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

import os
import numpy as np
import pickle
import torch
import sys
sys.path.append('..')
import transformations


# Load pickle file and closes it
def load_pickle(filename):
    with open(filename, 'rb') as f:
        p = pickle.load(f)
    return p
        
class Stream_CLF_Dataset(Dataset):

    def __init__(self,
                 dataset_name,
                 path_dataset,
                 chunk_len_ms, k, mT, MT,
                 clip_length_ms,
                 validation, 
                 h_flip, crop_size_perc, rotation_angle,
                 height, width,
                 exclude_samples = [],
                 **kwargs,
                 ):
        
        self.dataset_name = dataset_name
        self.validation = validation
        self.path_dataset = path_dataset
        self.height = height
        self.width = width
        
        self.clip_length_ms = clip_length_ms; self.k = k; self.mT = mT; self.MT = MT
        
        self.samples = {}
        num_sample = 0
        for s in os.listdir(path_dataset):
            cont = False
            for exc in exclude_samples:
                if exc in s: 
                    cont = True
                    break
            if cont: continue
            label = int(s.split('_')[-1])
            if 'SL_Animals' in self.dataset_name: label -= 1
            s = os.path.join(path_dataset, s)
            self.samples[num_sample] = {'path': s, 'label': label, 'len': len(os.listdir(s))}
            num_sample += 1
            # break
        self.num_samples = len(self.samples)
        
        self.h_flip = h_flip
        self.crop_size_perc = crop_size_perc
        self.rotation_angle = rotation_angle
        
        self.clip_length_ms = clip_length_ms
        self.clip_length_us = self.clip_length_ms*1000
        self.num_chunks = (clip_length_ms // chunk_len_ms)+1          # Number of time-steps (T) per sample. Given by the depth frequency (20Hz / 50 ms)
       
        print(f'** [{self.num_samples}] samples | samples of [{self.clip_length_ms}] ms | sequences of [{self.num_chunks}] chunks | chunks of [{chunk_len_ms}] ms')
        print(self.path_dataset)
        
    def get_label_dict(self):
        label_dict = {}
        for i,d in self.samples.items():
            label = d['label']
            if label not in label_dict: label_dict[label] = []
            label_dict[label].append(i)
        return label_dict
    
    
    def __len__(self):
        return self.num_samples
    

    # Return -> [num_timesteps, num_chunk_events, 2pol], [num_timesteps, num_chunk_events, 2pix_xy], [num_timesteps]
    def __getitem__(self, idx):
        
        label, sample_len, sample_path = torch.tensor(self.samples[idx]['label']), self.samples[idx]['len'], self.samples[idx]['path']
        
        if self.clip_length_ms == -1:
            init_t, end_t = 0, sample_len-1
        else:
            if not self.validation:
                init_t = np.random.randint(0, max(1,sample_len-self.num_chunks))
            else:
                diff = sample_len - self.num_chunks
                init_t = max(0, diff // 2)
            end_t = min(init_t + self.num_chunks - 1, sample_len-1)
        

        event_frames, total_min_max = [], []
        for t in range(init_t, end_t+1):
            ef, min_max = load_pickle(sample_path + f'/{t:04}.pckl')
            event_frames.append(torch.from_numpy(ef))
            total_min_max.append(min_max)
        total_min_max = np.array(total_min_max)
        if len(event_frames) == 0: print('*'*120)

        if end_t - init_t + 1 < self.num_chunks and self.clip_length_ms != -1:
            event_frames = [torch.zeros((self.height, self.width, self.k, 2))]*(self.num_chunks - end_t + init_t - 1) + event_frames
        event_frames = torch.stack(event_frames)
        
        event_frames = event_frames.view(*event_frames.shape[:3], self.k*2)
        event_frames = event_frames.float()
        
        batch_samples = {'label': label, 'event_frames': event_frames}
        
        return batch_samples
    


# Return the batch sample indices randomly.
class CustomBatchSampler():
    
    def __init__(self, batch_size, dt, sample_repetitions, iterative=False):      # , skip_evaluations = 1
        assert batch_size % sample_repetitions == 0
        self.batch_size = batch_size
        self.dt = dt
        self.sample_repetitions = sample_repetitions
        self.iterative = iterative
        self.num_dt_samples = dt.__len__()
        print(f' * Creating CustomBatchSampler with {self.__len__()} epochs')
        
    def __len__(self):
        if not self.iterative:
            epoch_length = np.ceil(self.num_dt_samples * self.sample_repetitions / self.batch_size)
            return int(epoch_length)
        else:
            return int(np.ceil(self.num_dt_samples/self.batch_size))
    
    def __iter__(self):
        if not self.iterative:
            batches = []
            for _ in range(self.__len__()):
                batch = np.random.randint(self.num_dt_samples, size=int(self.batch_size // self.sample_repetitions)).repeat(self.sample_repetitions)
                batches.append(batch)
            return iter(batches)
        else:
            batches = np.arange(self.num_dt_samples-1, 0, -1)
            batches = [ l.tolist() for l in np.array_split(batches, self.__len__()) ]
            return iter(batches)
        

    
class Stream_CLF_DataModule(LightningDataModule):
    def __init__(self, dataset_name, data_params, batch_size, patch_size, 
                 sample_repetitions, num_workers, pin_memory, **kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.data_params = data_params
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.sample_repetitions = sample_repetitions
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.pixel_dim = data_params['k']*2
        
        if 'SL_Animals' in self.dataset_name:
            self.original_height, self.original_width = 128,128
            self.num_classes = 19
            if '3Sets' in self.dataset_name: data_params['exclude_samples'] = ['_S3_']
            elif '4Sets' in self.dataset_name: pass
            else: raise ValueError('.')
            self.path_dataset = "../datasets/SL_Animals/dataset_4sets_frames"
        elif 'DVS128' in self.dataset_name:
            self.original_height, self.original_width = 128,128
            if '11Cls' in self.dataset_name: 
                self.num_classes = 11; data_params['exclude_samples'] = []
            elif '10Cls' in self.dataset_name: 
                self.num_classes = 10; data_params['exclude_samples'] = ['_10']
            else: raise ValueError('')
            self.path_dataset = "../datasets/DvsGesture/dataset_frames"
        else: raise ValueError(f'Dataset [{self.dataset_name}] not handled')
        print(self.path_dataset)

        # Set image dimensions after padding
        self.height = self.original_height + patch_size - self.original_height % patch_size if self.original_height % patch_size != 0 else self.original_height
        self.width =  self.original_width + patch_size - self.original_width % patch_size if self.original_width % patch_size != 0 else self.original_width
        self.data_params['height'], self.data_params['width'] = self.original_height, self.original_width
        
        # self.to_patches = True
        
        
    # Return dict:
        # depth -> B x H x W
        # event_frames  -> B x num_time_steps x num_sub_time_steps (chunks_per_depth) x H x W x K 2 
        # images        -> B x num_time_steps x num_sub_time_steps (2-3) x H x W
        # images_vi     -> B x num_time_steps x num_sub_time_steps (2-3) x H x W
    def custom_collate_fn(self, batch_samples): 
        res = {
                'event_frames': torch.stack([ d['event_frames'] for d in batch_samples ]),
                'labels': torch.stack([ d['label'] for d in batch_samples ]).long(),
            }
        res['event_frames'] = transformations.pad(res['event_frames'], patch_size=self.patch_size, pad_value=0.0)
        return res

    def train_dataloader(self):
        dt = Stream_CLF_Dataset(path_dataset = self.path_dataset+'/train/', validation=False, **self.data_params)
        sampler = CustomBatchSampler(self.batch_size, dt, self.sample_repetitions)
        dl = DataLoader(dt, batch_sampler=sampler, collate_fn=self.custom_collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return dl

    
    def val_dataloader(self):
        dt = Stream_CLF_Dataset(path_dataset = self.path_dataset+'/test/', validation=True, **self.data_params)
        dl = DataLoader(dt, batch_size=(self.batch_size//2)+1, shuffle=False, collate_fn=self.custom_collate_fn, num_workers=self.num_workers, pin_memory=self.pin_memory)
        return dl
        
    
