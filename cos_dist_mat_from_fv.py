# -*- coding: utf-8 -*-
"""
FUNCTION:
    COS_DIST_MAT_FROM_FV: creates audio shingles from feature vectors
    and outputs self dissimilarity matrix
    
INPUT: 
    1. MATRIX_FEATUREVECS: Matrix of feature vectors where each
    column is a timestep
    2. NUM_FV_PER_SHINGLE: Value for audio shingle size
    
OUTPUT:
    SELF_DISSIM_MAT: Self dissimilarity matrix with paired
    cosine distances between shingles

"""

import numpy as np
import scipy.spatial.distance as spd

def cos_dist_mat_from_fv(matrix_featurevecs, num_fv_per_shingle):
    
    [num_rows, num_columns] = matrix_featurevecs.shape
    
    if num_fv_per_shingle == 1:
        mat_as = matrix_featurevecs
        
    else:
        mat_as = np.zeros(((num_rows * num_fv_per_shingle),
                           (num_columns - num_fv_per_shingle + 1)))
        
        for i in range(1, num_fv_per_shingle):
            mat_as[((i-1) * num_rows):((i * num_rows)),] = matrix_featurevecs[:,
                   (i-1):((num_columns - num_fv_per_shingle + i))]
            
sdm_row = spd.pdist(mat_as, 'cosine')
self_dissim_mat = spd.squareform(sdm_row)
