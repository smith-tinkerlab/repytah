# -*- coding: utf-8 -*-
import numpy as np


def __find_song_pattern(thresh_diags):
    """
    Stitches information from thresh_diags matrix into a single
        row, song_pattern, that shows the timesteps containing repeats

    Args
    ----
    thresh_diags: array
        binary matrix with 1 at each pair (SI,SJ) and 0 elsewhere. 
        WARNING: must be symmetric
    
    Returns
    -------
    song_pattern: array
        row where each entry represents a time step and the group 
        that time step is a member of
    """
    song_length = thresh_diags.shape[0]
    
    # Initialize song pattern base
    pattern_base = np.zeros((1,song_length), dtype = int)

    # Initialize group number
    pattern_num = 1
    
    col_sum = thresh_diags.sum(axis = 0)

    check_inds = col_sum.nonzero()
    check_inds = check_inds[0]
    
    # Creates vector of song length
    pattern_mask = np.ones((1, song_length))
    pattern_out = (col_sum == 0)
    pattern_mask = pattern_mask - pattern_out
    
    while np.size(check_inds) != 0:
        # Takes first entry in check_inds
        i = check_inds[0]
        
        # Takes the corresponding row from thresh_diags
        temp_row = thresh_diags[i,:]
        
        # Finds all time steps that i is close to
        inds = temp_row.nonzero()
        
        if np.size(inds) != 0:
            while np.size(inds) != 0:
                # Takes sum of rows corresponding to inds and
                # multiplies the sums against p_mask
                c_mat = np.sum(thresh_diags[inds,:], axis = 0)
                c_mat = c_mat*pattern_mask
                
                # Finds nonzero entries of c_mat
                c_inds = c_mat.nonzero()
                c_inds = c_inds[1]
                
                # Gives all elements of c_inds the same grouping 
                # number as i
                pattern_base[0,c_inds] = pattern_num
                
                # Removes all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                pattern_mask[0,c_inds] = 0
                
                # Resets inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                inds = np.delete(inds,0)
                
            # Updates grouping number to prepare for next group
            pattern_num = pattern_num + 1
            
        # Removes i from check_inds
        check_inds = np.setdiff1d(check_inds, i)
        
    song_pattern = pattern_base
    
    return song_pattern
