# -*- coding: utf-8 -*-
"""
Stretches entries of matrix of thresholded diagonal onsets
    into diagonals of given length
    
ARGS: 
    thresh_diags: Binary matrix where entries equal to 1 
        signal the existence of a diagonal
    
    band_width: Length of encoded diagonals
    
RETURNS:
    stretch_diag_mat: Binary matrix with diagonals of length 
        band_width starting at each entry prescribed in 
        thresh_diag
    
"""

import numpy as np

def stretch_diags(thresh_diags, band_width):
    
    # creates size of returned matrix
    n = thresh_diags.shape[1] + band_width - 1
    
    temp_song_marks_out = np.zeros(n)
    
    (inds, jnds) = thresh_diags.nonzero()
    
    subtemp = np.identity(band_width)
    
    # expands each entry in thresh_diags into diagonal of
    # length band width
    for i in range(1, band_width.shape[1]):
        tempmat = np.zeros(n)
        
        tempmat[inds[i]:(inds[i] + band_width - 1), 
                jnds[i]:(jnds[i] + band_width - 1)] = subtemp
        
        temp_song_marks_out = temp_song_marks_out + tempmat
                
    # ensures that stretch_diag_mat is a binary matrix
    stretch_diag_mat = (temp_song_marks_out > 0)
    
    return stretch_diag_mat