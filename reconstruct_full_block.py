#!/usr/bin/env python
# coding: utf-8

import numpy as np

def reconstruct_full_block(pattern_mat, pattern_key): 
    """
    Creates a binary matrix with a block of 1's for 
        each repeat encoded in pattern_mat whose length 
        is encoded in patern_key

    Args
    ----
    pattern_mat: np.array
        binary matrix with 1's where repeats begin 
        and 0's otherwise
     
    pattern_key: np.array
        vector containing the lengths of the repeats 
        encoded in each row of pattern_mat

    Returns
    -------
    pattern_block: np.array
        binary matrix representation for pattern_mat 
        with blocks of 1's equal to the length's 
        prescribed in pattern_key
    """
    # Number of beats in pattern_mat (columns)
    sn = pattern_mat.shape[1]

    # Number of repeated structures in pattern_mat (rows)
    sb = pattern_mat.shape[0]

    # Pre-allocating a sn by sb array of zeros 
    pattern_block = np.zeros((sb,sn)).astype(int)  

    for i in range(sb):

        # Retrieve all of row i of pattern_mat 
        repeated_struct = pattern_mat[i,:]

        # Retrieve the length of the repeats encoded in row i of pattern_mat 
        length = pattern_key[i]

        # Pre-allocate a section of size length x sn for pattern_block
        sub_section = np.zeros((length, sn))

        # Replace first row in block_zeros with repeated_structure 
        sub_section[0,:] = repeated_struct

        # Creates pattern_block: Sums up each column after sliding repeated 
        # structure i to the right bw - 1 times 
        for b in range(2, length + 1): 

            # Retrieve repeated structure i up to its (1 - b) position 
            sub_struct_a = repeated_struct[0:(1 - b)]

            # Row vector with number of entries not included in sub_struct_a  
            sub_struct_b = np.zeros((1,( b  - 1)))

            # Append sub_struct_b in front of sub_struct_a 
            new_struct = np.append(sub_struct_b, sub_struct_a)

            # Replace part of sub_section with new_struct 
            sub_section[b - 1,:] = new_struct

        # Replaces part of pattern_block with the sums of each column in sub_section 
        pattern_block[i,:] = np.sum(sub_section, axis = 0)

    return pattern_block
    
# Calling function with example inputs 
reconstruct_full_block(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ], 
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]]), np.array([3, 5, 8, 8 ]))



