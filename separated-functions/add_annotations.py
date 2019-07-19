# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sps

def find_song_pattern(thresh_diags):
    """
    Stitches information from thresholded diagonal matrix into a single
        row

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


def add_annotations(input_mat, song_length):
    """
    Adds annotations to pairs of repeats in input matrix

    Args
    ----
    input_mat: array
        list of pairs of repeats. The first two columns refer to 
        the first repeat of the pair. The third and fourth columns refer
        to the second repeat of the pair. The fifth column refers to the
        repeat lengths. The sixth column contains any previous annotations,
        which will be removed.
        
    song_length: int
        number of audio shingles in the song.
    
    Returns
    -------
    anno_list: array
        list of pairs of repeats with annotations marked
    """
    num_rows = input_mat.shape[0]
    
    # Removes any already present annotation markers
    input_mat[:, 5] = 0
    
    # Find where repeats start
    s_one = input_mat[:,0]
    s_two = input_mat[:,2]
    
    # Creates matrix of all repeats
    s_three = np.ones((num_rows,), dtype = int)
    
    up_tri_mat = sps.coo_matrix((s_three, (s_one, s_two)),
                                shape = (song_length, song_length)).toarray()
    
    low_tri_mat = up_tri_mat.conj().transpose()
    
    full_mat = up_tri_mat + low_tri_mat
    
    # Stitches info from input_mat into a single row
    song_pattern = find_song_pattern(full_mat)
    
    # Adds annotation markers to pairs of repeats
    for i in song_pattern[0]:
        pinds = np.nonzero(song_pattern == i)
        
        #One if annotation not already marked, 0 if it is
        check_inds = (input_mat[:,5] == 0)
        
        for j in pinds[1]:
            
            # Finds all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j) + (s_two == j)
            mark_inds = (mark_inds > 0)
            mark_inds = check_inds * mark_inds
            
            # Adds found annotations to the relevant time steps
            input_mat[:,5] = (input_mat[:,5] + i * mark_inds)
            
            # Removes pairs of repeats with annotations from consideration
            check_inds = check_inds ^ mark_inds
     
    temp_inds = np.argsort(input_mat[:,5])
    
    # Creates list of annotations
    anno_list = input_mat[temp_inds,]
    
    return anno_list
    
