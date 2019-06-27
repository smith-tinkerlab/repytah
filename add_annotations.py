# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sps

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
    
    # gets maximum of each column
    #sp_max = song_pattern.max(0)
    
    # adds annotation markers to pairs of repeats
    for i in song_pattern:
        pinds = np.nonzero(song_pattern == i)
        
        #one if annotation not already marked, 0 if it is
        check_inds = (input_mat[:,5] == 0)
        
        for j in pinds:
            
            # finds all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j) + (s_two == j)
            mark_inds = (mark_inds > 0)
            mark_inds = check_inds * mark_inds
            
            # adds found annotations to the relevant time steps
            input_mat[:,5] = (input_mat[:,5] + i * mark_inds)
            
            # removes pairs of repeats with annotations from consideration
            check_inds = check_inds - mark_inds
            
    (unused, temp_inds) = np.sort(input_mat[:,5])
    
    # creates list of annotations
    anno_list = input_mat[temp_inds,]
    
    return anno_list
    