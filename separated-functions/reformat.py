#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:03:56 2019

@author: lizettecarpenter
"""
import numpy as np

def reformat(pattern_mat, pattern_key):
    """Transforms a binary array with 1's where repeats start and 0's
    otherwise into an a list of repeated stuctures. This list consists of
    information about the repeats including length, when they occur and when
    they end. 
    
    Every row has a pair of repeated structure. The first two columns are 
    the time steps of when the first repeat of a repeated structure start and 
    end. Similarly, the second two columns are the time steps of when the 
    second repeat of a repeated structure start and end. The fourth colum is 
    the length of the repeated structure. 
    
    reformat.py may be helpful when writing example inputs for aligned 
    hiearchies.
    
    Args
    ----
        pattern_mat: np.array 
            binary array with 1's where repeats start and 0's otherwise 
        
        pattern_key: np.array 
            array with the lengths of each repeated structure in pattern_mat
            
    Returns
    -------
        info_mat: np.array 
            array with the time steps of when the pairs of repeated structures 
            start and end organized 

    """

    #Pre-allocate output array with zeros 
    info_mat = np.zeros((pattern_mat.shape[0], 5))
    
    #Retrieve the index values of the repeats in pattern_mat 
    results = np.where(pattern_mat == 1)
    
    #1. Find the starting indices of the repeated structures row by row 
    for r in range(pattern_mat.shape[0]):
        #Find where the repeats start  
        r_inds = (pattern_mat[r] == 1) 
        inds = np.where(r_inds)
        
        #Retrieve the starting indices of the repeats 
        s_ij = inds[0] 
        
        #Seperate the starting indices of the repeats 
        i_ind = s_ij[0]
        j_ind = s_ij[1]
        
        #2. Assign the time steps of the repeated structures into  info_mat
        for x in results[0]:
            #If the row equals the x-value of the repeat
            if r == x:
                info_mat[r, 0] = i_ind + 1
                info_mat[r, 1] = i_ind + pattern_key[r] 
                info_mat[r, 2] = j_ind + 1 
                info_mat[r, 3] = j_ind + pattern_key[r]
                info_mat[r, 4] = pattern_key[r]
                
    return info_mat 