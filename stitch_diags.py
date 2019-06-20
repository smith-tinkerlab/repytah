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
    
    #stuff happens
    
    return song_pattern
