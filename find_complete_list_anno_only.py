# -*- coding: utf-8 -*-
import numpy as np

def fcl_anno_only(pair_list, song_length):
    """
    Finds annotations for all pairs of repeats found in previous step
    
    Args
    ----
    pair_list: 
        list of pairs of repeats
        WARNING: bandwidths must be in ascending order
        
    song_length: int
        number of audio shingles in song
        
    Returns
    -------
    out_lst:
        list of pairs of repeats with smaller repeats added and with
        annotation markers
        
    """
    
    # find list of unique repeat lengths
    bw_found = np.unique(pair_list[:,4])
    bw_num = bw_found.shape[0]
    
    full_lst = np.array([])
    
    # remove longest bandwidth row if it is the length of the full song
    if song_length == bw_found[b-1]:
        
    
    # add annotation markers to each pair of repeats
    for j in range(1, bw_length + 1):
        bandwidth = 
        # isolate pairs of repeats of desired length
        
        # get annotations for this bandwidth
    return out_lst

