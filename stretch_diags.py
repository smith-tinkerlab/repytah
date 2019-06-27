# -*- coding: utf-8 -*-

import numpy as np

def stretch_diags(thresh_diags, band_width):
    """
    Stretches entries of matrix of thresholded diagonal onsets
        into diagonals of given length
    
    ARGS: 
        thresh_diags: Binary matrix where entries equal to 1 
            signal the existence of a diagonal
    
        band_width: Length of encoded diagonals
    
    RETURNS:
        stretch_diag_mat: Logical matrix with diagonals of length 
            band_width starting at each entry prescribed in 
            thresh_diag
    
    """
    # creates size of returned matrix
    n = thresh_diags.shape[0] + band_width - 1
    
    temp_song_marks_out = np.zeros(n)
    
    (jnds, inds) = thresh_diags.nonzero()
    
    subtemp = np.identity(band_width)
    
    # expands each entry in thresh_diags into diagonal of
    # length band width
    for i in range(inds.shape[0]):
        tempmat = np.zeros((n,n))
        
        tempmat[inds[i]:(inds[i] + band_width), 
                jnds[i]:(jnds[i] + band_width)] = subtemp
        
        temp_song_marks_out = temp_song_marks_out + tempmat
                
    # ensures that stretch_diag_mat is a binary matrix
    stretch_diag_mat = (temp_song_marks_out > 0)
    
    return stretch_diag_mat