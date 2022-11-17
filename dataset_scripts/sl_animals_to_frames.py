import os
import pandas as pd
import numpy as np
import pickle

import aermanager
from aermanager.aerparser import load_events_from_file
from tqdm import tqdm

import events_to_frames


np.random.seed(0)


chunk_len_ms = 48
chunk_len_us = chunk_len_ms*1000
height, width = 128, 128
k = 3
minTime, maxTime = chunk_len_us/k, 256*1000

# Define source and destination folders
path_dataset = '../datasets/SL_Animals/'
path_dataset_dst = path_dataset + '/dataset_{}_frames/'
print('***', 'Storing in', path_dataset_dst)

# Create destination folders
for mode in ['train', 'test']:
    for s in ['3Sets', '4sets']:
        d = path_dataset_dst.format(s) + mode
        if not os.path.isdir(d): os.makedirs(d)

# Read filenames and define training/test splits
files = os.listdir(path_dataset + 'allusers_aedat/')
# Test split files used in this work
test_samples_4sets = [ 'user10_indoor.aedat', 'user12_indoor.aedat', 'user14_indoor.aedat', 'user17_indoor.aedat', 'user19_sunlight.aedat', 
                      'user24_sunlight.aedat', 'user29_imse.aedat', 'user30_imse.aedat', 'user34_imse.aedat', 'user35_dc.aedat', 
                      'user36_dc.aedat', 'user37_dc.aedat', 'user38_dc.aedat', 'user42_dc.aedat', 'user57_dc.aedat' ]
train_samples_4sets = [ f for f in files if f not in test_samples_4sets ]
# from sklearn.model_selection import train_test_split
# train_samples_4sets, test_samples_4sets = train_test_split(files, test_size=0.25, random_state=0,
#                                       stratify=[ f[:-6].split('_')[-1] for f in files ])


for events_file in tqdm(files):
    # Load events
    shape, events = load_events_from_file(path_dataset + 'allusers_aedat/' + events_file, parser=aermanager.parsers.parse_dvs_128)
    labels = pd.read_csv(path_dataset + 'tags_updated_19_08_2020/' + events_file.replace('.aedat', '.csv'))

    filename_dst = path_dataset_dst + '{}/' + events_file.replace('.aedat', '_{}_{}')

    for _,row in labels.iterrows():
        
        # Events for each sample
        sample_events = events[row.startTime_ev:row.endTime_ev]
        
        total_events = np.array([sample_events['y'], sample_events['x'], sample_events['t'], sample_events['p']]).transpose()

        # Get frames from events
        total_frames, min_max_values = events_to_frames.process_event_stream(total_events, height, width, chunk_len_us, k, minTime, maxTime)
        total_frames = total_frames.astype(np.float16)

        # Check sample set
        if '_sunlight' in events_file:  val_set = 'S4' # S4 indoors with frontal sunlight
        elif '_indoor' in events_file:  val_set = 'S3' # S3 indoors neon light
        elif '_dc' in events_file:      val_set = 'S2' # S2 natural side light
        elif '_imse' in events_file:    val_set = 'S1' # S1 natural side light
        else: raise ValueError('Set not handled')


        # Define sample filename
        if events_file in train_samples_4sets: filename_row = filename_dst.format('4sets', 'train', val_set, row['class'])
        elif events_file in test_samples_4sets:  filename_row = filename_dst.format('4sets', 'test',  val_set, row['class'])
        else: raise ValueError(f'events_file [{events_file}] not found')
        if not os.path.isdir(filename_row): os.mkdir(filename_row)
        
        # Store frames
        if events_file in train_samples_4sets:
            for t in range(total_frames.shape[0]):
                pickle.dump((total_frames[t], min_max_values[t]), open(filename_row + f'/{t:04}.pckl', 'wb'))
        elif events_file in test_samples_4sets:
            for t in range(total_frames.shape[0]):
                pickle.dump((total_frames[t], min_max_values[t]), open(filename_row + f'/{t:04}.pckl', 'wb'))
        else: raise ValueError(f'events_file [{events_file}] not found')
        
print('**** Stored in ', path_dataset_dst)
        
