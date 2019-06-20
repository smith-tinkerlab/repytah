# -*- coding: utf-8 -*-
"""
Creates audio shingles from feature vectors, finds cosine 
    distance between shingles, and returns self dissimilarity matrix
    
ARGS: 
    matrix_featurevecs: Matrix of feature vectors where each
        column is a timestep
        
    num_fv_per_shingle: Number of feature vectors per audio
        shingle
    
RETURNS:
    self_dissim_mat: Self dissimilarity matrix with paired
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
        
        for i in range(1, num_fv_per_shingle + 1):
            # use feature vectors to create an audio shingle
            # for each time step and represent these shingles
            # as vectors by stacking the relevant feature
            # vectors on top of each other
            mat_as[((i-1) * num_rows):((i * num_rows)),] = matrix_featurevecs[:,
                   (i-1):((num_columns - num_fv_per_shingle + i))]
            
    sdm_row = spd.pdist(mat_as, 'cosine')
    self_dissim_mat = spd.squareform(sdm_row)
    
    return self_dissim_mat
