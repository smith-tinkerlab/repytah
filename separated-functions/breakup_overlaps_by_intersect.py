#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from inspect import signature 

def breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, thresh_bw):
    """
    Extract repeats in input_patter_obj that has the starting indices of the 
    repeats, into the essential structure componets using bw_vec, that has the 
    lengths of each repeat. The essential structure components are the 
    smallest building blocks that form the basis for every repeat in the song. 
    
    Args
    ----
        input_pattern_obj: np.array 
            binary matrix with 1's where repeats begin 
            and 0's otherwise 
        
        bw_vec: np.array 
            vector containing the lengths of the repeats
            encoded in input_pattern_obj
        
        thresh_bw: number
            the smallest allowable repeat length 
        
    Returns
    -------
        pattern_no_overlaps: np.array 
            binary matrix with 1's where repeats of 
            essential structure components begin 
        
        pattern_no_overlaps_key: np.array 
            vector containing the lengths of the repeats
            of essential structure components in
            pattern_no_overlaps 
    """
    sig = signature(breakup_overlaps_by_intersect)
    params = sig.parameters 
    if len(params) < 3: 
        T = 0 
    else: 
        T = thresh_bw
    
    #Initialize input_pattern_obj 
    PNO = input_pattern_obj
    
    #Sort the bw_vec and the PNO so that we process the biggest pieces first
    
    #Part 1: Sort the lengths in bw_vec in descending order 
    sort_bw_vec = np.sort(bw_vec)
    desc_bw_vec = sort_bw_vec[::-1]
    
    #Part 2: Sort the indices of bw_vec in descending order 
    bw_inds = np.argsort(desc_bw_vec, axis = 0)
    row_bw_inds = np.transpose(bw_inds)
    
    
    PNO = PNO[(row_bw_inds),:]
    PNO = PNO.reshape(4,19)
    
    T_inds = np.nonzero(bw_vec == T) 
    T_inds = np.array(T_inds) - 1  

    if T_inds.size != 0: 
        T_inds = max(bw_vec.shape) - 1
    
    
    PNO_block = reconstruct_full_block(PNO, desc_bw_vec)
    
    # Check stopping condition -- Are there overlaps?
    while np.sum(PNO_block[ : T_inds, :]) > 0:
        
        # Find all overlaps by comparing the rows of repeats pairwise
        overlaps_PNO_block = check_overlaps(PNO_block)
        
        # Remove the rows with bandwidth T or less from consideration
        overlaps_PNO_block[T_inds:, ] = 0
        overlaps_PNO_block[:,T_inds:] = 0
        
        # Find the first two groups of repeats that overlap, calling one group
        # RED and the other group BLUE
        [ri,bi] = overlaps_PNO_block.nonzero() 
        [ri,bi] = np.where(overlaps_PNO_block !=0)
        
        red = PNO[ri,:]
        RL = bw_vec[ri,:]
        
        blue = PNO[bi,:]
        BL = bw_vec[bi,:]
        
        # Compare the repeats in RED and BLUE, cutting the repeats in those
        # groups into non-overlapping pieces
        union_mat, union_length = compare_and_cut(red, RL, blue, BL)
        
        PNO = np.delete(PNO, ri, axis = 0)
        PNO = np.delete(PNO, bi, axis = 0)

        bw_vec = np.delete(bw_vec, ri, axis = 0)
        bw_vec = np.delete(bw_vec, bi, axis = 0)
        
        PNO = np.vstack(PNO, union_mat) 
        bw_vec = np.vstack(bw_vec, union_length)
        
        # Check there are any repeats of length 1 that should be merged into
        # other groups of repeats of length 1 and merge them if necessary
        if sum(union_length == 1) > 0:
            PNO, bw_vec = merge_based_on_length(PNO, bw_vec, 1)
        
        #AGAIN, Sort the bw_vec and the PNO so that we process the biggest 
        #pieces first
        #Part 1: Sort the lengths in bw_vec in descending order 
        sort_bw_vec = np.sort(bw_vec)
        desc_bw_vec = sort_bw_vec[::-1]
        #Part 2: Sort the indices of bw_vec in descending order 
        bw_inds = np.argsort(desc_bw_vec, axis = 0)
        row_bw_inds = np.transpose(bw_inds)
        
        PNO = PNO[(row_bw_inds),:]
        
        # Find the first row that contains repeats of length less than T and
        # remove these rows from consideration during the next check of the
        # stopping condition
        #T_inds = np.nonzeros(bw_vec == T, 1) 
        T_inds = np.amin(desc_bw_vec == T) - 1
        T_inds = np.array(T_inds) # Bends is converted into an array

        if T_inds.size != 0:  
            T_inds = max(desc_bw_vec.shape) - 1

        PNO_block = reconstruct_full_block(PNO, desc_bw_vec)
    
    #Sort the lengths in bw_vec in ascending order 
    bw_vec = np.sort(desc_bw_vec)
    #Sort the indices of bw_vec in ascending order     
    bw_inds = np.argsort(desc_bw_vec)
   
    pattern_no_overlaps = PNO[bw_inds,:]
    pattern_no_overlaps_key = bw_vec
        
    output = (pattern_no_overlaps, pattern_no_overlaps_key)
    
    return output 
