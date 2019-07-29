#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np 

def compare_and_cut(red, red_len, blue, blue_len):
    """
    Compares two rows of repeats labeled RED and BLUE, and determines if there 
    are any overlaps in time between them. If there is, then we cut the 
    repeats in RED and BLUE into up to 3 pieces. 
    
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
    
    #Determine if the rows have any intersections
    red_block = reconstruct_full_block(red, RL)
    blue_block = reconstruct_full_block(blue, BL)

    red_block = red_block > 0
    blue_block = blue_block > 0 
    purple_block = np.logical_and(red_block, blue_block)
    
    #If there is any intersection between the rows, then start comparing one
    #repeat in red to one repeat in blue
    if purple_block.sum() > 0:  
        # Find number of blocks in red and in blue
        LSR = max(start_red.shape)
        LSB = max(start_blue.shape) 
        
        #Build the pairs of starting indices to search, where each pair
        #contains a starting index in red and a starting index in blue
        red_inds = np.tile(start_red.transpose(), (LSB, 1))
        blue_inds = np.tile(start_blue, (LSR,1))

        
        compare_inds = np.concatenate((blue_inds.transpose(),  red_inds), \
                                      axis = None)
        compare_inds = np.reshape(compare_inds, (4,2), order='F')
    
        
        #Initialize the output variables union_mat and union_length
        union_mat = np.array([])
        union_length = np.array([]) 
    
        #Loop over all pairs of starting indices
        for start_ind in range(0, LSR*LSB):
            #Isolate one repeat in red and one repeat in blue
            ri = compare_inds[start_ind, 1]
            bi = compare_inds[start_ind, 0]
            
            red_ri = np.arange(ri, ri+RL)
            blue_bi = np.arange(bi, bi+BL)
            
            #Determine if the blocks intersect and call the intersection
            #purple
            purple = np.intersect1d(red_ri,blue_bi)
            
            if purple.size != 0: 
            
                #Remove pruple from red_ri, call it red_minus_purple
                red_minus_purple = np.setdiff1d(red_ri,purple)
                
                #If red_minus_purple is not empty, then see if there are one
                #or two parts in red_minus_purple. Then cut purple out of all
                #of the repeats in red. If there are two parts left in
                #red_minus_purple, then the new variable new_red, which holds
                #the part(s) of red_minus_purple, should have two rows with
                #1's for the starting indices of the resulting pieces and 0's
                #elsewhere. Also red_length_vec will have the length(s) of the
                #parts in new_red.
                if red_minus_purple.size != 0:
                    red_start_mat, red_length_vec = __num_of_parts(\
                                              red_minus_purple, ri, start_red)
                    new_red = __inds_to_rows(red_start_mat,sn)
                else:
                    # If red_minus_purple is empty, then set new_red and
                    # red_length_vec to empty
                    new_red = np.array([])
                    red_length_vec = np.array([])
           
                #Noting that purple is only one part and in both red_ri and
                #blue_bi, then we need to find where the purple starting
                #indices are in all the red_ri
                purple_in_red_mat, purple_length = __num_of_parts(purple, ri, \
                                                                start_red)
                
                #If blue_minus_purple is not empty, then see if there are one
                #or two parts in blue_minus_purple. Then cut purple out of all
                #of the repeats in blue. If there are two parts left in
                #blue_minus_purple, then the new variable new_blue, which
                #holds the part(s) of blue_minus_purple, should have two rows
                #with 1's for the starting indices of the resulting pieces and
                #0's elsewhere. Also blue_length_vec will have the length(s)
                #of the parts in new_blue.
                blue_minus_purple = np.setdiff1d(blue_bi,purple)
                
                if blue_minus_purple.size != 0: 
                    blue_start_mat, blue_length_vec = __num_of_parts(\
                                            blue_minus_purple, bi, start_blue)
                    new_blue = __inds_to_rows(blue_start_mat, sn)
                else:
                    #If blue_minus_purple is empty, then set new_blue and
                    #blue_length_vec to empty
                    new_blue = np.array([])
                    blue_length_vec = np.array([])
                    
                #Recalling that purple is only one part and in both red_rd 
                #and blue_bi, then we need to find where the purple starting
                #indices are in all the blue_ri
                purple_in_blue_mat, x = __num_of_parts(purple, bi, start_blue)
                
                #Union purple_in_red_mat and purple_in_blue_mat to get
                #purple_start, which stores all the purple indices
                purple_start = np.union1d(purple_in_red_mat, \
                                          purple_in_blue_mat)
                
                #Use purple_start to get new_purple with 1's where the repeats
                #in the purple rows start and 0 otherwise. 
                new_purple = __inds_to_rows(purple_start, sn);
                
                if new_red.size != 0 | new_blue.size != 0:
                    #Form the outputs
                    union_mat = np.vstack((new_red, new_blue, new_purple))
                    union_length = np.vstack((red_length_vec, \
                                              blue_length_vec, purple_length))

                    union_mat, union_length = _merge_based_on_length(\
                                        union_mat, union_length, union_length)
                    break
                elif new_red.size == 0 & new_blue.size == 0:
                    new_purple_block = reconstruct_full_block(new_purple,\
                                                              purple_length)
                    if max(new_purple_block.shape) < 2:
                        union_mat = new_purple
                        union_length = purple_length
                        break
            
    #Check that there are no overlaps in each row of union_mat
    union_mat_add = np.array([])
    union_mat_add_length = np.array([])
    union_mat_rminds = np.array([])
    
    #Isolate one row at a time, call it union_row
    for i in range(0, union_mat.shape[0] + 1):
        union_row = union_mat[i,:]
        union_row_width = union_length[i];
        union_row_block = reconstruct_full_block(union_row, union_row_width)
        
        #If there are at least one overlap, then compare and cut that row
        #until there are no overlaps
        if (union_row_block.sum(axis = 0) > 1) > 0:
            union_mat_rminds = np.vstack(union_mat_rminds, i)
            
            union_row_new, union_row_new_length = compare_and_cut(union_row,\
                                union_row_width, union_row, union_row_width)
            
            #Add union_row_new and union_row_new_length to union_mat_add and
            #union_mat_add_length, respectively
            union_mat_add = np.vstack(union_mat_add, union_row_new)
            union_mat_add_length = np.vstack(union_mat_add_length,\
                                             union_row_new_length)

    #Remove the old rows from union_mat (as well as the old lengths from
    #union_length)
    
    union_mat = np.delete(union_mat, union_mat_rminds, axis = 0)
    union_length = np.delete(union_length, union_mat_rminds)

    
    #Add union_row_new and union_row_new_length to union_mat and
    #union_length, respectively, such that union_mat is in order by
    #lengths in union_length
    union_mat = np.vstack(union_mat, union_mat_add)
    union_length = np.vstack(union_length, union_mat_add_length)
    
    union_length, UM_inds = np.sort(union_length)
    union_mat = union_mat[UM_inds,:]
    
    output = (union_mat, union_length) 
    
    return output 





