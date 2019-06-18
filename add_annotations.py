# -*- coding: utf-8 -*-
"""
Adds annotations to pairs of repeats in input matrix

ARGS:
    input_mat: List of pairs of repeats. The first two columns refer to 
        the first repeat of the pair. The third and fourth columns refer
        to the second repeat of the pair. The fifth column refers to the
        repeat lengths. The sixth column contains any previous annotations,
        which will be removed.
        
    song_length: Number of audio shingles in the song.
    
RETURNS:
    anno_list: List of pairs of repeats with annotations marked
"""

import numpy as np

def add_annotations(input_mat, song_length):
    
    num_columns = input_mat.shape[0]
    
    # removes any previous annotations
    input_mat[:, 5] = 0
    
    # find where repeats start
    
    return anno_list