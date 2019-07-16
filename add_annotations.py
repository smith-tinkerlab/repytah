# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sps

def stitch_diags(thresh_diags):
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
    
    # initialize song pattern base
    pattern_base = np.zeros((1,song_length), dtype = int)

    # initialize group number
    pattern_num = 1
    
    col_sum = thresh_diags.sum(axis = 0)
    #col_sum = np.delete(col_sum, 0)
    #col_sum = np.append(col_sum, 0)
    
    check_inds = col_sum.nonzero()
    check_inds = check_inds[0]
    
    # creates vector of song length
    pattern_mask = np.ones((1, song_length))
    pattern_out = (col_sum == 0)
    pattern_mask = pattern_mask - pattern_out
    
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
                c_mat = np.sum(thresh_diags[inds,:], axis = 0)
                c_mat = c_mat*pattern_mask
                
                # finds nonzero entries of c_mat
                c_inds = c_mat.nonzero()
                c_inds = c_inds[1]
                
                # gives all elements of c_inds the same grouping 
                # number as i
                pattern_base[0,c_inds] = pattern_num
                #pattern_base[c_inds] = pattern_num
                
                # removes all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                pattern_mask[0,c_inds] = 0
                
                # resets inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                inds = np.delete(inds,0)
                
            # updates grouping number to prepare for next group
            pattern_num = pattern_num + 1
            
        # removes i from check_inds
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
    
    # removes any already present annotation markers
    input_mat[:, 5] = 0
    
    # find where repeats start
    s_one = input_mat[:,0]
    s_two = input_mat[:,2]
    
    # creates matrix of all repeats
    s_three = np.ones((num_rows,), dtype = int)
    
    up_tri_mat = sps.coo_matrix((s_three, (s_one, s_two)),
                                shape = (song_length, song_length)).toarray()
    
    low_tri_mat = up_tri_mat.conj().transpose()
    
    full_mat = up_tri_mat + low_tri_mat
    
    
    # stitches info from input_mat into a single row
    song_pattern = stitch_diags(full_mat)
    
    # adds annotation markers to pairs of repeats
    for i in song_pattern[0]:
        pinds = np.nonzero(song_pattern == i)
        
        #one if annotation not already marked, 0 if it is
        check_inds = (input_mat[:,5] == 0)
        
        for j in pinds[1]:
            
            # finds all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j) + (s_two == j)
            mark_inds = (mark_inds > 0)
            mark_inds = check_inds * mark_inds
            
            # adds found annotations to the relevant time steps
            input_mat[:,5] = (input_mat[:,5] + i * mark_inds)
            
            # removes pairs of repeats with annotations from consideration
            check_inds = check_inds ^ mark_inds
     
    temp_inds = np.argsort(input_mat[:,5])
    
    # creates list of annotations
    anno_list = input_mat[temp_inds,]
    
    return anno_list
    