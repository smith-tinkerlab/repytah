#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

def compare_and_cut(red, RL, blue, BL):
    """
    Compares two rows of repeats labeled RED and BLUE, and determines if there 
    are any overlaps in time between them. If there is, then we cut the repeats
    in RED and BLUE into up to 3 pieces. 
    
    Args
    ----
        red: np.array 
            binary row vector encoding a set of repeats with 1's where each
            repeat starts and 0's otherwise 
            
        red_len: number 
            length of repeats encoded in red 
            
        blue: np.array 
            binary row vector encoding a set of repeats with 1's where each
            repeat starts and 0's otherwise 
            
        blue_len: number 
            length of repeats encoded in blue 

    Returns
    -------
        union_mat: np.array 
            binary matrix representation of up to three rows encoding
            non-overlapping repeats cut from red and blue

        union_length: np.array 
            vector containing the lengths of the repeats encoded in union_mat
    """
    sn = red.shape[0]
    assert sn == blue.shape[0]
    
    start_red = np.flatnonzero(red)
    start_red = start_red[None, :] 

    start_blue = np.flatnonzero(blue)
    start_blue = start_blue[None, :] 
    
    # Determine if the rows have any intersections
    red_block = reconstruct_full_block(red, RL)
    blue_block = reconstruct_full_block(blue, BL)

    red_block = red_block > 0
    blue_block = blue_block > 0 
    purple_block = np.logical_and(red_block, blue_block)
    
    # If there is any intersection between the rows, then start comparing one
    # repeat in RED to one repeat in BLUE
    if purple_block.sum() > 0:  
        # Find number of blocks in red and in blue
        LSR = max(start_red.shape)
        LSB = max(start_blue.shape) 
        
        # Build the pairs of starting indices to search, where each pair
        #contains a starting index in RED and a starting index in BLUE
        red_inds = np.tile(start_red.transpose(), (LSB, 1))
        blue_inds = np.tile(start_blue, (LSR,1))

        
        compare_inds = np.concatenate((blue_inds.transpose(),  red_inds), axis = None)
        compare_inds = np.reshape(compare_inds, (4,2), order='F')
    
        
        # Initialize the output variables union_mat and union_length
        union_mat = np.array([])
        union_length = np.array([]) 
    
        # Loop over all pairs of starting indices
        for start_ind in range(0, LSR*LSB):
            # Isolate one repeat in RED and one repeat in BLUE
            ri = compare_inds[start_ind, 1]
            bi = compare_inds[start_ind, 0]
            
            red_ri = np.arange(ri, ri+RL)
            blue_bi = np.arange(bi, bi+BL)
            
            # Determine if the blocks intersect and call the intersection
            # PURPLE
            purple = np.intersect1d(red_ri,blue_bi)
            
            if purple.size != 0: 
            
                # Remove PURPLE from RED_RI, call it RED_MINUS_PURPLE
                red_minus_purple = np.setdiff1d(red_ri,purple)
                
                # If RED_MINUS_PURPLE is not empty, then see if there are one
                # or two parts in RED_MINUS_PURPLE. Then cut PURPLE out of ALL
                # of the repeats in RED. If there are two parts left in
                # RED_MINUS_PURPLE, then the new variable NEW_RED, which holds
                # the part(s) of RED_MINUS_PURPLE, should have two rows with
                # 1's for the starting indices of the resulting pieces and 0's
                # elsewhere. Also RED_LENGTH_VEC will have the length(s) of the
                # parts in NEW_RED.
                if red_minus_purple.size != 0:
                    red_start_mat, red_length_vec = num_of_parts(red_minus_purple, ri, start_red)
                    new_red = inds_to_rows(red_start_mat,sn)
                else:
                    # If RED_MINUS_PURPLE is empty, then set NEW_RED and
                    # RED_LENGTH_VEC to empty
                    new_red = np.array([])
                    red_length_vec = np.array([])
           
                # Noting that PURPLE is only one part and in both RED_RI and
                # BLUE_BI, then we need to find where the purple starting
                # indices are in all the RED_RI
                purple_in_red_mat, purple_length = num_of_parts(purple, ri, start_red)
                
                # If BLUE_MINUS_PURPLE is not empty, then see if there are one
                # or two parts in BLUE_MINUS_PURPLE. Then cut PURPLE out of ALL
                # of the repeats in BLUE. If there are two parts left in
                # BLUE_MINUS_PURPLE, then the new variable NEW_BLUE, which
                # holds the part(s) of BLUE_MINUS_PURPLE, should have two rows
                # with 1's for the starting indices of the resulting pieces and
                # 0's elsewhere. Also BLUE_LENGTH_VEC will have the length(s)
                # of the parts in NEW_BLUE.
                blue_minus_purple = np.setdiff1d(blue_bi,purple)
                
                if blue_minus_purple.size != 0: 
                    blue_start_mat, blue_length_vec = num_of_parts(blue_minus_purple, bi, start_blue)
                    new_blue = inds_to_rows(blue_start_mat, sn)
                else:
                    # If BLUE_MINUS_PURPLE is empty, then set NEW_BLUE and
                    # BLUE_LENGTH_VEC to empty
                    new_blue = np.array([])
                    blue_length_vec = np.array([])
                    
                # Recalling that PURPLE is only one part and in both RED_RI and
                # BLUE_BI, then we need to find where the purple starting
                # indices are in all the BLUE_RI
                purple_in_blue_mat, x = num_of_parts(purple, bi, start_blue)
                
                # Union PURPLE_IN_RED_MAT and PURPLE_IN_BLUE_MAT to get
                # PURPLE_START, which stores all the purple indices
                purple_start = np.union1d(purple_in_red_mat, purple_in_blue_mat)
                
                # Use PURPLE_START to get NEW_PURPLE with 1's where the repeats
                # in the purple rows start and 0 otherwise. 
                new_purple = inds_to_rows(purple_start, sn);
                
                if new_red.size != 0 | new_blue.size != 0:
                    # Form the outputs
                    union_mat = np.vstack((new_red, new_blue, new_purple))
                    union_length = np.vstack((red_length_vec, blue_length_vec, purple_length))

                    union_mat, union_length = merge_based_on_length(union_mat, union_length, union_length)
                    break
                elif new_red.size == 0 & new_blue.size == 0:
                    new_purple_block = reconstruct_full_block(new_purple, purple_length)
                    if max(new_purple_block.shape) < 2:
                        union_mat = new_purple
                        union_length = purple_length
                        break
            
    # Check that there are no overlaps in each row of union_mat
    union_mat_add = np.array([])
    union_mat_add_length = np.array([])
    union_mat_rminds = np.array([])
    
    # Isolate one row at a time, call it union_row
    for i in range(0, union_mat.shape[0] + 1):
        union_row = union_mat[i,:]
        union_row_width = union_length[i];
        union_row_block = reconstruct_full_block(union_row, union_row_width)
        
        # If there are at least one overlap, then compare and cut that row
        # until there are no overlaps
        if (union_row_block.sum(axis = 0) > 1) > 0:
            union_mat_rminds = np.vstack(union_mat_rminds, i)
            
            union_row_new, union_row_new_length = compare_and_cut(union_row, union_row_width, union_row, union_row_width)
            
            # Add UNION_ROW_NEW and UNION_ROW_NEW_LENGTH to UNION_MAT_ADD and
            # UNION_MAT_ADD_LENGTH, respectively
            union_mat_add = np.vstack(union_mat_add, union_row_new)
            union_mat_add_length = np.vstack(union_mat_add_length, union_row_new_length)

    # Remove the old rows from UNION_MAT (as well as the old lengths from
    # UNION_LENGTH)
    
    union_mat = np.delete(union_mat, union_mat_rminds, axis = 0)
    union_length = np.delete(union_length, union_mat_rminds)

    
    # Add UNION_ROW_NEW and UNION_ROW_NEW_LENGTH to UNION_MAT and
    # UNION_LENGTH, respectively, such that UNION_MAT is in order by
    # lengths in UNION_LENGTH
    union_mat = np.vstack(union_mat, union_mat_add)
    union_length = np.vstack(union_length, union_mat_add_length)
    
    union_length, UM_inds = np.sort(union_length)
    union_mat = union_mat[UM_inds,:]
    
    output = union_mat, union_length 
    
    return output 





