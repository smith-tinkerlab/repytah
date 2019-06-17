# -*- coding: utf-8 -*-
"""
FUNCTION:
    
INPUT: 
    
OUTPUT:
    
"""

import numpy as np

def stretch_diags(thresh_diags, band_width):
    
    n = 
    
    temp_song_marks_out = np.zeros(n)
    
    (inds, jnds) = thresh_diags.nonzero()
    
    subtemp = np.identity(band_width)
    
    
    for i in range(1, bw.shape[1]):
        tempmat = np.zeros(n)
        
        tempmat[inds[i]:(inds[i] + band_width - 1), 
                jnds[i]:(jnds[i] + band_width - 1)] = subtemp
        
                temp_song_marks_out = temp_song_marks_out + tempmat
                
    
    stretch_diag_mat = (temp_song_marks_out > 0)
    
    return stretch_diag_mat