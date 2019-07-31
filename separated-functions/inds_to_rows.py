#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def __inds_to_rows(start_mat, row_length):
    """
    Expands a vector containing the starting indices of a piece or two of a 
    repeat into a matrix representation recording when these pieces occur in 
    the song with 1's. All remaining entries are marked with 0's. 
    
    Args
    ----
        start_mat: np.array 
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
    mat_rows = start_mat.shape[0]
    new_mat = np.zeros((mat_rows,row_length))

    for i in range(0, mat_rows + 1):
        inds = start_mat[i,:]
        new_mat[i,inds] = 1;

    return new_mat



