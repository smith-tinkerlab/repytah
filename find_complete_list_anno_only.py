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
    
    # remove longest bandwidth row if it is the length of the full song
    if song_length == bw_found[bw_num - 1]:
        pair_list[-1,:] = []
        bw_found[-1] = []
        bw_num = (bw_num - 1)
    p = pair_list.shape[0]
    
    # add annotation markers to each pair of repeats
    full_list = []
    for j in range(bw_num):
        bandwidth = bw_found[j]
        # isolate pairs of repeats of desired length
        bsnds = np.amin(np.nonzero(pair_list[:,4] == bandwidth))
        bends = np.nonzero(pair_list[:,4] > bandwidth)
        
        if np.size(bends) > 0:
            bends = np.amin(bends)
        else:
            bends = p
        
        bw_mat = np.array((pair_list[bsnds:bends,]))
        bw_mat_length = bw_mat.shape[0]
        
        temp_anno_mat = np.concatenate((bw_mat, (np.zeros((bw_mat_length,1)))),
                                       axis = 1).astype(int)

        # get annotations for this bandwidth
        temp_anno_list = add_annotations(temp_anno_mat, song_length)
        full_list.append(temp_anno_list)
        
    out_list = np.concatenate(full_list)
        
    return out_list

