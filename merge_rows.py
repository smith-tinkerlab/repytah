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
    merge_mat: np.array
        binary matrix with ones where repeats start and zeroes otherwise
    """
    # Step 0: initialize temporary variables
    not_merge = input_mat    # Everything must be checked
    merge_mat = []           # Nothing has been merged yet
    merge_key = []
    rows = input_mat.shape[0]  # How many rows to merge?
    
    # Step 1: has every row been checked?
    while rows > 0:
        # Step 2: start merge process
        # Step 2a: choose first unmerged row
        row2check = not_merge[0,:]
        r2c_mat = np.kron(np.ones((rows,1)), row2check) # Create a comparison matrix
                                                        # with copies of row2check stacked
                                                        # so that r2c_mat is the same
                                                        # size as the set of rows waiting
                                                        # to be merged
        
        # Step 2b: find indices of unmerged overlapping rows
        merge_inds = np.sum(((r2c_mat + not_merge) == 2), axis = 1) > 0
        
        # Step 2c: union rows with starting indices in common with row2check and
        # remove those rows from input_mat
        union_merge = np.sum(not_merge[merge_inds,:], axis = 0) > 0
        np.delete(not_merge, not_merge[merge_inds,:])
          
        # Step 2d: check that newly merged rows do not cause overlaps within row
        # If there are conflicts, rerun compare_and_cut
        merge_block = reconstruct_full_block(union_merge, input_width)
        if np.max(merge_block) > 1:
            (union_merge, union_merge_key) = compare_and_cut(union_merge, input_width,
            union_merge, input_width)
        else:
            union_merge_key = input_width
        
        # Step 2e: add unions to merge_mat and merge_key
        merge_mat = np.array([[merge_mat], [union_merge]])
        merge_key = np.array([[merge_key], [union_merge_key]])
        
        # Step 3: reinitialize rs for stopping condition
        rows = not_merge.shape[0]
    
    return merge_mat