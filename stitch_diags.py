# -*- coding: utf-8 -*-
"""
Stitches information from thresholded diagonal matrix into a single
    row (where each entry represents a time step and the entry 
    indicates the group that step is part of)

ARGS:
    thresh_diags: Binary matrix with 1 at each pair (SI,SJ) and 0
        elsewhere. WARNING: must be symmetric
        
    z_or_n: Binary indicator determining whether NAN or 0 is used
        for temporary variables. 0 means zeroes and 1 means NANs
    
RETURNS:
    song_pattern: Row where each entry represents a time step and
        the group that time step is a member of
    
"""

import numpy as np

def stitch_diags(thresh_diags, z_or_n):
    
    num_rows = thresh_diags.shape[0]
    
    if z_or_n == 0:
        p_base = np.zeros((1,num_rows), dtype = int)
    elif z_or_n == 1:
        #will I have to do a for loop to change an array of 
        #zeros into an array of NaNs? 
    
    # initializing group number
    pattern_num = 1
    
    col_sum = thresh_diags.sum(axis = 0)
    
    check_inds = col_sum.nonzero()
    
    # creates vector of song length
    p_mask = np.ones(1, num_rows)
    p_out = (col_sum == 0)
    p_mask = p_mask - p_out
    
    while check_inds.size != 0:
        
        # takes first entry in check_inds
        i = check_inds[0]
        
        # takes the corresponding row from thresh_diags
        temp_row = thresh_diags[i,:]
        
        # finds all time steps that i is close to
        inds = temp_row.nonzero()
        
        if inds.size != 0:
            while inds.size != 0:
                
                # takes sum of rows corresponding to inds and
                # multiplies the sums against p_mask
                c_mat = 
    
    return song_pattern


