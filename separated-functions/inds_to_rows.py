#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def __inds_to_rows(input_inds_mat, row_length):
    """
    Converts a list of indices to row(s) with 1's where an 
    index occurs and 0's otherwise
    
    Args
    ----
        input_inds_mat: np.array 
            matrix of one or two rows, containing the 
            starting indices 
            
        row_length: int 
            length of the rows 
            
    Returns
    -------
        new_mat: np.array 
            matrix of one or two rows, with 1's where 
            the starting indices and 0's otherwise 
    """
    mat_rows = input_inds_mat.shape[0]
    new_mat = np.zeros((mat_rows,row_length))

    for i in range(0, mat_rows + 1):
        inds = input_inds_mat[i,:]
        new_mat[i,inds] = 1;

    return new_mat



