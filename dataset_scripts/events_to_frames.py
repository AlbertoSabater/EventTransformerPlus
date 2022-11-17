import numpy as np


# total_events -> [(x,y,t,p)]
def process_event_stream(total_events, height, width, chunk_len_us, k, minTime, maxTime, pos_fifo=None, neg_fifo=None):
    # Remove consecutive events -> make sure that events for a certain pixel are at least minTime away
    if minTime > 0: 
        total_events = total_events[::-1]   # Reverse to extract higher time-events -> unique sort later in increase order again
        orig_time = total_events[:,2].copy()    # Save a copy of the original time-stamps
        total_events[:,2] = total_events[:,2] - np.mod(total_events[:,2], minTime)  # Binarize by minTime
        uniq, inds = np.unique(total_events, return_index=True, axis=0)     # Extract unique binarized events
        total_events = total_events[inds]
        total_events[:,2] = orig_time[inds]         # Roll back to the original time-stamps

            
    min_max_values = []
    # List time-window timestamp endings
    tw_ends = np.arange(int(total_events[:,2].max()), int(total_events[:,2].min()), -chunk_len_us)[::-1]
    
    tw_init = -np.inf
    
    # True to return the FIFOs 
    return_mem = pos_fifo is not None and not neg_fifo is None
    
    # FIFOs of size K for the positive and negative events
    if pos_fifo is None: pos_fifo = np.full((height, width, k), -np.inf, dtype=np.float32)
    if neg_fifo is None: neg_fifo = np.full((height, width, k), -np.inf, dtype=np.float32)

    frames = []
    time_steps = []
    for tw_num, current_tw_end in enumerate(tw_ends):
        
        # Select events within the slice
        tw_events_inds = (total_events[:,2] > tw_init) & (total_events[:,2] <= current_tw_end)
        
        # Get pos/neg frame representations
        new_pos, new_neg, min_max = get_representation(total_events, tw_events_inds, k, height, width)
        if new_pos is None or (new_pos.sum() == 0.0 and new_neg.sum() == 0.0): 
            if tw_num != 0: print(f'*** {tw_num} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[1]}')   # ', cat_id, num_sample, sampleLoop
            # print(f'*** {sampleLoop} | Empty window: p0 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[0]} | p1 {total_events[(tw_events_inds) & (total_events[:,3]==0)].shape[1]}')   # ', cat_id, num_sample, sampleLoop
            continue
        
        # Update fifos. Append new events, move zeros to the beggining and retain last k events for each pixel/polarity
        pos_fifo = np.sort(np.concatenate([pos_fifo, new_pos], axis=2), axis=2)[:,:,-k:]
        neg_fifo = np.sort(np.concatenate([neg_fifo, new_neg], axis=2), axis=2)[:,:,-k:]
            
        # Build frame by stacking positive and negative fifo representations
        frame = np.stack([neg_fifo, pos_fifo], axis=-1)
        frames.append(frame)
        
        tw_init = current_tw_end
        min_max_values.append(min_max)
        time_steps.append(current_tw_end)

    if len(frames) == 0: return []
    frames = np.stack(frames)
    time_steps = np.array(time_steps)
            
    # Make each window in the range (0, maxTime)
    diff = maxTime - time_steps
    diff = diff[:,None,None,None,None]
    frames = frames + diff           # Make newer events to have higher value than the older ones
    frames[frames < 0] = 0
    frames = frames / maxTime             # Make newer events to have a value close to 1 and older ones a value close to 0
    if return_mem: return (frames, min_max_values), (pos_fifo, neg_fifo)
    else: return frames, min_max_values


def evetns_to_frame_v0(events, unique_coords_pos, unique_indexes_pos, height, width, k):
    # Initialize positive frame
    new_pos = np.full((height, width, k), 0)      # Initialize frame representation
    if not len(unique_coords_pos) == 0: 
        agg_pos = np.split(events[:, 2], unique_indexes_pos[1:])
        # Get only the last k events for each coordinate
        agg_k_pos = [ pix_agg[-k:] for pix_agg in agg_pos ]
        # List of the last k events per pixel
        agg_k_pos = np.array([ np.pad(pix_agg, (k-len(pix_agg), 0)) for pix_agg in agg_k_pos ])
        new_pos[unique_coords_pos[:,1], unique_coords_pos[:,0]] = agg_k_pos
    return new_pos

# Create a frame representation of the given events
def evetns_to_frame(events, unique_coords_pos, unique_indexes_pos, height, width, k):
    # Initialize frame
    new_pos = np.full((height * width * k), 0)      # Initialize frame representation
    if not len(unique_coords_pos) == 0: 
        true_inds, k_inds = [], []
        prev_ind = -1
        # true_inds: calculate the positions of events belonging to each coordinate
        # k_inds: calculate the position k of each event
        for num_item, i in enumerate(unique_indexes_pos):
            current_true_ind = 1 + np.arange(max(prev_ind, i-k, 0), i)
            true_inds.append(current_true_ind)
            k_inds.append(np.arange(k-len(current_true_ind), k))
            prev_ind = i
        true_inds = np.concatenate(true_inds)
        k_inds = np.concatenate(k_inds)

        events = events[true_inds]
        # Transform pixel and k array coordinates to ravel array position
        true_coords = np.concatenate([events[:,[1,0]], k_inds[:,None]], axis=1, dtype=int)
        true_coords_inds = np.ravel_multi_index(true_coords.transpose(), (height,width, k))
        
        # Add time-stamp information to the empty ravel frame
        new_pos[true_coords_inds] = events[:,2]
    # Reshape ravel frame
    new_pos = new_pos.reshape(height, width, k)
    return new_pos



# Transform a list of events into a positive and negative frame
# Frames contains the last k events (their time-stamp) from total_events for each pixel
# This frame only contains event information from the events (total_events) of the current time-window
# Older events from the FIFOs will be added later if needed
# total_events -> [(x,y,t,p)]
def get_representation(total_events, tw_events_inds, k, height, width):
    # Select events for the current time-window
    pos_inds = (tw_events_inds) & (total_events[:,3]==1)
    pos_events = total_events[pos_inds]
    # Sort events by y, x, timestamp
    pos_events = pos_events[np.lexsort((pos_events[:,2], pos_events[:,1], pos_events[:,0]))]
    # Aggregate events per pixel. Get unique event coordinates -> avoid duplicates
    unique_coords_pos, unique_indexes_pos = np.unique(pos_events[:, :2], return_index=True, axis=0)
    # new_pos = evetns_to_frame_v0(pos_events, unique_coords_pos, unique_indexes_pos, height, width, k)
    new_pos = evetns_to_frame(pos_events, unique_coords_pos, unique_indexes_pos, height, width, k)
        
    # Select  events for the current time-window
    neg_inds = (tw_events_inds) & (total_events[:,3]==0)
    neg_events = total_events[neg_inds]
    # Sort events by y, x, timestamp
    neg_events = neg_events[np.lexsort((neg_events[:,2], neg_events[:,1], neg_events[:,0]))]
    # Aggregate events per pixel. Get unique event coordinates -> avoid duplicates
    unique_coords_neg, unique_indexes_neg = np.unique(neg_events[:, :2], return_index=True, axis=0)
    # new_neg = evetns_to_frame_v0(neg_events, unique_coords_neg, unique_indexes_neg, height, width, k)
    new_neg = evetns_to_frame(neg_events, unique_coords_neg, unique_indexes_neg, height, width, k)
    

    # More recent samples are close to zero 0
    if len(unique_coords_pos) == 0 and  len(unique_coords_neg) == 0:
        mins = maxs = (0,0)
    elif len(unique_coords_pos) == 0:
        mins, maxs = unique_coords_neg.min(0), unique_coords_neg.max(0)
    elif len(unique_coords_neg) == 0:
        mins, maxs = unique_coords_pos.min(0), unique_coords_pos.max(0)
    else:
        mins, maxs = np.concatenate([unique_coords_pos, unique_coords_neg], axis=0).min(0), np.concatenate([unique_coords_pos, unique_coords_neg], axis=0).max(0)
    min_max = (int(mins[1]), int(maxs[1]), int(mins[0]), int(maxs[0]))
    
    return new_pos, new_neg, min_max


