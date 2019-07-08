# -*- coding: utf-8 -*-

import numpy as np

def merge_rows(input_mat, input_width):
    """
    Merges rows with at least one common repeat from the same repeated structure
    
    Args
    ----
    input_mat: np.array
        binary matrix with ones where repeats start and zeroes otherwise
        
    input_width: int
        length of repeats encoded in input_mat
        
    Returns
    -------
    WARNING: returns as tuple as (merge_mat, merge_key)
    
    merge_mat: np.array
        binary matrix with ones where repeats start and zeroes otherwise
        
    merge_key:
        vector containing lengths of repeats encoded in merge_mat
    """
    # step 0: initialize temporary variables
    not_merge = input_mat    # everything must be checked
    merge_mat = []           # nothing has been merged yet
    merge_key = []
    rs = input_mat.shape[0]  # how many rows to merge?
    
    # step 1: has every row been checked?
    while rs > 0:
        # step 2: start merge process
        # step 2a: choose first unmerged row
        row2check = not_merge[0,:]
        r2c_mat = np.kron(np.ones((rs,1)), row2check) # create a comparison matrix
                                                      # with copies of row2check stacked
                                                      # so that r2c_mat is the same
                                                      # size as the set of rows waiting
                                                      # to be merged
        
        # step 2b: find indices of unmerged overlapping rows
        merge_inds = np.sum(((r2c_mat + not_merge) == 2), axis = 1) > 0
        
        # step 2c: union rows with starting indices in common with row2check and
        # remove those rows from input_mat
        union_merge = np.sum(not_merge[merge_inds,:], axis = 0) > 0
        not_merge[merge_inds,:] = []
        # possibility: not_merge = not_merge[1:,:]
          
        # step 2d: check that newly merged rows do not cause overlaps within row
        # if there are conflicts, rerun compare_and_cut
        merge_block = reconstruct_full_block(union_merge, input_width)
        if np.max(merge_block) > 1:
            (union_merge, union_merge_key) = compare_and_cut(union_merge, input_width,
            union_merge, input_width)
            
        else:
            union_merge_key = input_width
        
        
        # step 2e: add unions to merge_mat and merge_key
        merge_mat = np.array([[merge_mat], [union_merge]])
        merge_key = np.array([[merge_key], [union_merge_key]])
        
        
        # step 3: reinitialize rs for stopping condition
        rs = not_merge.shape[0]
        
    merge_tup = tuple(merge_mat, merge_key)
    
    return merge_tup
        
    
    

