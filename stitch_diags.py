# -*- coding: utf-8 -*-

import numpy as np

def stitch_diags(thresh_diags):
    
    """
    Stitches information from thresholded diagonal matrix into a single
        row

    ARGS
    ----
    thresh_diags: array
        Binary matrix with 1 at each pair (SI,SJ) and 0 elsewhere. 
        WARNING: must be symmetric
    
    RETURNS
    -------
    song_pattern: array
        Row where each entry represents a time step and the group 
        that time step is a member of
    
    """
    
    num_rows = thresh_diags.shape[0]
    
    p_base = np.zeros((1,num_rows), dtype = int)

    # initializing group number
    pattern_num = 1
    
    col_sum = thresh_diags.sum(axis = 0)
    
    check_inds = col_sum.nonzero()
    check_inds = check_inds[0]
    
    # creates vector of song length
    p_mask = np.ones((1, num_rows))
    p_out = (col_sum == 0)
    p_mask = p_mask - p_out
    
    while np.size(check_inds) != 0:
        
        # takes first entry in check_inds
        i = check_inds[0]
        
        # takes the corresponding row from thresh_diags
        temp_row = thresh_diags[i,:]
        
        # finds all time steps that i is close to
        inds = temp_row.nonzero()
        
        if np.size(inds) != 0:
            while np.size(inds) != 0:
                
                # takes sum of rows corresponding to inds and
                # multiplies the sums against p_mask
                c_mat = np.sum(thresh_diags[inds,:], axis = 1)
                
                # finds nonzero entries of c_mat
                c_inds = c_mat.nonzero()
                
                # gives all elements of c_inds the same grouping 
                # number as i
                p_base[0,c_inds] = pattern_num
                
                # removes all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                p_mask[0,c_inds] = 0
                
                # resets inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                
            # updates grouping number to prepare for next group
            pattern_num = pattern_num + 1
            
        # removes i from check_inds
        check_inds = np.setdiff1d(check_inds, i)
        
    song_pattern = p_base
    
    return song_pattern


