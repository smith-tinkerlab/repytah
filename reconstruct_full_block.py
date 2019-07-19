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
    #Find number of beats (columns) in pattern_mat
    
    #Check size of pattern_mat (in cases where there is only 1 pair of
    #repeated structures)
    if (pattern_mat.ndim == 1): 
        #Convert a 1D array into 2D array 
        #From https://stackoverflow.com/questions/3061761/numpy-array-dimensions
        pattern_mat = pattern_mat[None, : ]
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
    else: 
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
        
    #Assign number of repeated structures (rows) in pattern_mat to sb 
    sb = pattern_mat.shape[0]
    
    #Pre-allocating a sn by sb array of zeros 
    pattern_block = np.zeros((sb,sn)).astype(int)  
    
    #Check if pattern_key is in vector row 
    if pattern_key.ndim != 1: 
        #Convert pattern_key into a vector row 
        length_vec = np.array([])
        for i in pattern_key:
            length_vec = np.append(length_vec, i).astype(int)
    else: 
        length_vec = pattern_key 
    
    for i in range(sb):
        #Retrieve all of row i of pattern_mat 
        repeated_struct = pattern_mat[i,:]
    
        #Retrieve the length of the repeats encoded in row i of pattern_mat 
        length = length_vec[i]
    
        #Pre-allocate a section of size length x sn for pattern_block
        sub_section = np.zeros((length, sn))
    
        #Replace first row in block_zeros with repeated_structure 
        sub_section[0,:] = repeated_struct
        
        #Creates pattern_block: Sums up each column after sliding repeated 
        #sastructure i to the right bw - 1 times 
        for b in range(2, length + 1): 
    
            #Retrieve repeated structure i up to its (1 - b) position 
            sub_struct_a = repeated_struct[0:(1 - b)]
    
            #Row vector with number of entries not included in sub_struct_a  
            sub_struct_b = np.zeros((1,( b  - 1)))
    
            #Append sub_struct_b in front of sub_struct_a 
            new_struct = np.append(sub_struct_b, sub_struct_a)
            
            #Replace part of sub_section with new_struct 
            sub_section[b - 1,:] = new_struct
    
    #Replaces part of pattern_block with the sums of each column in 
    #sub_section 
    pattern_block[i,:] = np.sum(sub_section, axis = 0)
    
    return pattern_block
    

    








