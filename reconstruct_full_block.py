# coding: utf-8

"""
FUNCTION: RECONSTRUCT_FULL_BLOCK creates a binary matrix with a block of 1's for 
each repeat encoded in PATTERN_MAT whose length is encoded in PATTERN_KEY

INPUT: 
    1. PATTERN_MAT: Binary matrix with 1's where repeats begin and 0's otherwise
    2. PATTERN_KEY: Vector containing the lengths of the repeats encoded in each 
    row of PATTERN_MAT

OUTPUT: 
    PATTERN_BLOCK: Binary matrix representation for PATTERN_MAT with blocks of 1's 
    equal to the length's prescribed in PATTERN_KEY
"""

import numpy as np

def reconstruct_full_block(pattern_mat, pattern_key):
    
    pattern_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]])
    pattern_key = np.array([3, 5, 8, 8 ])
    
    P = pattern_mat
    K = pattern_key
    sn = pattern_mat.shape[1] #number of columns
    sb = pattern_mat.shape[0] #number of rows 
    
    pattern_block = np.zeros((sb,sn)).astype(int)  

    for i in range(sb):
        p_row = P[i,:]
        bw = K[i]
        block_mat = np.zeros((bw, sn))
        block_mat[0,:] = p_row
        for b in range(2,bw + 1): 
            x = p_row[0:(1 - b)]
            y = np.zeros((1,( b  - 1)))
            z = np.append(y,x)
            block_mat[b - 1,:] = z
        pattern_block[i,:] = np.sum(block_mat, axis = 0)
    
    return pattern_block 

