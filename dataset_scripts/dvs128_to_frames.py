import aermanager
from aermanager.aerparser import load_events_from_file
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os

import events_to_frames



chunk_len_ms = 24
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
k = 3
minTime, maxTime = chunk_len_us/k, 256*1000

# Define source and destination folders
path_dataset_src = '../datasets/DvsGesture/'
path_dataset_dst = '../datasets/DvsGesture/dataset_frames/'

# Create destination folders
if not os.path.isdir(path_dataset_dst + 'train'): os.makedirs(path_dataset_dst + 'train')
if not os.path.isdir(path_dataset_dst + 'test'): os.makedirs(path_dataset_dst + 'test')

# Load train/test splits
train_files, test_files = 'trials_to_train.txt', 'trials_to_test.txt'
with open(path_dataset_src + train_files, 'r') as f: train_files = f.read().splitlines()
with open(path_dataset_src + test_files, 'r') as f: test_files = f.read().splitlines()



num_sample = 0
for events_file in tqdm(train_files + test_files):
    
    if events_file == '': continue
    # Load events
    labels = pd.read_csv(path_dataset_src + events_file.replace('.aedat', '_labels.csv'))
    shape, total_events = load_events_from_file(path_dataset_src + events_file, parser=aermanager.parsers.parse_dvs_ibm)
    total_events = np.array([total_events['x'], total_events['y'], total_events['t'], total_events['p']]).transpose()
    
    filename_dst = path_dataset_dst + '{}/' + '{}_' + events_file.replace('.aedat', '_{}')
    
    # Load user samples
    # Class 0 for non action
    time_segment_class = []     # [(t_init, t_end, class)]
    prev_event = 0
    for _,row in labels.iterrows():
        if (row.startTime_usec-1 - prev_event) > 0: time_segment_class.append((prev_event, row.startTime_usec-1, 0, (row.startTime_usec-1 - prev_event)/1000))
        if ((row.endTime_usec - row.startTime_usec)) > 0: time_segment_class.append((row.startTime_usec, row.endTime_usec, row['class'], (row.endTime_usec - row.startTime_usec)/1000))
        prev_event = row.endTime_usec + 1 
    time_segment_class.append((prev_event, np.inf, 0, np.inf))
    
    
    for init, end, label, seq_len in time_segment_class:
        if label == 0: continue
        label -= 1
        # Events for each sample
        sample_events = total_events[(total_events[:,2] >= init) & (total_events[:,2] <= end)]
    
        if events_file in train_files: filename_row = filename_dst.format('train', f'{num_sample:04}', label)
        elif events_file in test_files:  filename_row = filename_dst.format('test', f'{num_sample:04}', label)
        if not os.path.isdir(filename_row): os.mkdir(filename_row)
    
        # Get frames from events
        total_frames, min_max_values = events_to_frames.process_event_stream(sample_events, height, width, chunk_len_us, k, minTime, maxTime)
        total_frames = total_frames.astype(np.float16)        
        
        # Store frames
        if events_file in train_files:
            for t in range(total_frames.shape[0]):
                pickle.dump((total_frames[t], min_max_values[t]), open(filename_row + f'/{t:04}.pckl', 'wb'))
        elif events_file in test_files:
            for t in range(total_frames.shape[0]):
                pickle.dump((total_frames[t], min_max_values[t]), open(filename_row + f'/{t:04}.pckl', 'wb'))
        else: raise ValueError(f'events_file [{events_file}] not found')      
        
        num_sample += 1

        
print('**\nStored in:', path_dataset_dst)

