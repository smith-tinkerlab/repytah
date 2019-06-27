# coding: utf-8

"""
FUNCTION: SEPARATE_ALL_ANNOTATIONS expands PATTERN_ROW into a matrix, so that there
is one group of repeats per row.

INPUT: 
    1. K_MAT -- List of pairs of repeats of length BAND_WIDTH with 
                annotations marked. The first two columns refer to the
                first repeat of the pair, the second two refer to the
                second repeat of the pair, the fifth column refers to the
                length of the repeats, and the sixth column contains the
                annotation markers.
    2. SN -- Song length, which is the number of audio shingles
    3. BAND_WIDTH -- The length of repeats encoded in K_MAT
    4. PATTERN_ROW -- Row that marks where non-overlapping repeats
                       occur, marking the annotation markers for the
                       start indices and 0's otherwise

OUTPUT: 
    1. PATTERN_MAT -- Matrix representation of K_MAT with one row for 
                        each group of repeats
    2. PATTERN_KEY -- Vector containing the lengths of the repeats
                        encoded in each row of PATTERN_MAT
    3. ANNO_ID_LST -- Vector containing the annotation markers of the 
                        repeats encoded in each row of PATTERN_MAT
"""

import numpy as np

def by_annotation_marker(k_mat, sn, band_width, pattern_row): 

    #k_mat = np.array([[1, 1, 3, 3, 1, 1],
                      #[4, 4, 6, 6, 1, 2]]) 
    k_mat = np.array([[1, 1, 3, 3, 1, 1]])
    sn = 6
    band_width = 1
    #pattern_row = np.array([1, 0, 1, 2, 0, 2])
    pattern_row = np.array([1, 0, 1, 0, 0, 0])
    
    anno_lst = k_mat[:,5] 
    anno_max = np.max(anno_lst)
    pattern_mat = np.zeros((anno_max, sn), dtype = np.intp)

    if anno_max > 1: 
        for a in anno_lst: 
            columns = np.append(k_mat[a-1,0]-1, k_mat[a-1,2]-1)
            pattern_mat[a -1, columns] = 1
            pattern_key = band_width * np.ones((anno_max, 1)).astype(int)
    else: 
        pattern_mat = pattern_row 
        pattern_key = band_width
        #anno_id_lst = [:anno_max] isnt this just anno_lst? 

    return pattern_mat 





