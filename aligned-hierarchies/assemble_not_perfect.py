# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 16:40:35 2020

@author: Administrator
"""
import numpy as np
from inspect import signature 
from search import find_all_repeats
from utilities import reconstruct_full_block


def check_each_row(union_mat,union_length):
    """    
    This function checks if there are overlaps within each row.
    If there is an overlap, return True and an int which tells which 
    row has the overlap.
    If there is no overlap, return false(To make the output in the same
    function, I added a 1 after False, but we will not use that number).
    
    """
    for i in range(0, union_mat.shape[0]):
        union_row = union_mat[i,:]
        union_row_width = np.array([union_length[i]])
        union_row_block = reconstruct_full_block(union_row, union_row_width)
        if (np.sum(union_row_block[0]>1)) > 0:
            return (True,i)
    return (False,1)
        

def __num_of_parts(input_vec, input_start, input_all_starts):
    """    
    This function is used to determine the number of blocks of consecutive 
    time steps in a list of time steps. A block of consecutive time steps
    represent a distilled section of a repeat. This distilled section will be 
    replicated and the starting indices of the repeats within it will be 
    returned. 
    

    Args
    ----
        input_vec: np.array 
            contains one or two parts of a repeat that are overlap(s) in time 
            that may need to be replicated 
            
        input_start: np.array index 
            starting index for the part to be replicated 
        
        input_all_starts: np.array indices 
            starting indices for replication 
    
    Returns
    -------
        start_mat: np.array 
            array of one or two rows, containing the starting indices of the 
            replicated repeats 
            
        length_vec: np.array 
            column vector containing the lengths of the replicated parts 
    """
    
    diff_vec = np.subtract(input_vec[1:], input_vec[:-1])
    break_mark = diff_vec > 1
    if sum(break_mark) == 0: 
        start_vec = input_vec[0]
        end_vec = input_vec[-1]
        add_vec = start_vec - input_start
        start_mat = input_all_starts + add_vec

    else:
        start_vec = np.zeros((2,1))
        end_vec =  np.zeros((2,1))
    
        start_vec[0] = input_vec[0]
        end_vec[0] = input_vec[break_mark - 2]
    
        start_vec[1] = input_vec[break_mark - 1]
        end_vec[1] = input_vec[-1]
    
        add_vec = np.array(start_vec - input_start).astype(int)
        input_all_starts = np.array(input_all_starts).astype(int)
        start_mat = np.vstack((input_all_starts + add_vec[0], input_all_starts + add_vec[1]))

    length_vec = end_vec - start_vec + 1
    output = (start_mat, length_vec)
    
    return output 

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
    if (start_mat.ndim == 1): 
        #Convert a 1D array into 2D array 
        #From:
        #https://stackoverflow.com/questions/3061761/numpy-array-dimensions
        start_mat = start_mat[None, : ]
    mat_rows = start_mat.shape[0]
    new_mat = np.zeros((mat_rows,row_length))
    for i in range(0, mat_rows):
        inds = start_mat[i,:]
        new_mat[i,inds] = 1;

    return new_mat


def cut(red, red_len, blue, blue_len):
    if_merge = False
    sn = red.shape[0]
    assert sn == blue.shape[0]
    start_red = np.flatnonzero(red)
    start_red = start_red[None, :] 

    start_blue = np.flatnonzero(blue)
    start_blue = start_blue[None, :] 
    #Determine if the rows have any intersections
    red_block = reconstruct_full_block(red, red_len)
    blue_block = reconstruct_full_block(blue, blue_len)
    red_block = red_block > 0
    blue_block = blue_block > 0 
    purple_block = np.logical_and(red_block, blue_block)
    
    # If there is any intersection between the rows, then start comparing one
    # repeat in red to one repeat in blue
    if purple_block.sum() > 0:
        
        # Find number of blocks in red and in blue
        LSR = max(start_red.shape)
        LSB = max(start_blue.shape) 
        
        # Build the pairs of starting indices to search, where each pair
        # contains a starting index in red and a starting index in blue
        red_inds = np.tile(start_red.transpose(), (LSB, 1))
        blue_inds = np.tile(start_blue, (LSR,1))
        tem_blue = blue_inds[0][0]
        for i in range (0,blue_inds.shape[1]):
            for j in range (0,blue_inds.shape[0]):
                tem_blue = np.vstack((tem_blue,blue_inds[j][i]))
        tem_blue = np.delete(tem_blue,1,0) 
        compare_inds = np.concatenate((tem_blue,  red_inds), \
                                      axis = 1)
       
        
        # Initialize the output variables union_mat and union_length
        union_mat = np.array([])
        union_length = np.array([]) 
    
        # Loop over all pairs of starting indices
        for start_ind in range(0, LSR*LSB):
            # Isolate one repeat in red and one repeat in blue
            ri = compare_inds[start_ind, 1]
            bi = compare_inds[start_ind, 0]
            red_ri = np.arange(ri, ri+red_len)
            blue_bi = np.arange(bi, bi+blue_len)
            
            # Determine if the blocks intersect and call the intersection
            # purple
            purple = np.intersect1d(red_ri, blue_bi)
            
            if purple.size != 0: 
            
                # Remove purple from red_ri, call it red_minus_purple
                red_minus_purple = np.setdiff1d(red_ri, purple)
                
                # If red_minus_purple is not empty, then see if there are one
                # or two parts in red_minus_purple.
                # Then cut purple out of all of the repeats in red. 
                if red_minus_purple.size != 0:
                    # red_length_vec will have the length(s) of the parts in 
                    # new_red 
                    red_start_mat, red_length_vec = __num_of_parts(\
                                              red_minus_purple, ri, start_red)
                    
                    # If there are two parts left in red_minus_purple, then 
                    # the new variable new_red, which holds the part(s) of 
                    # red_minus_purple, should have two rows with 1's for the 
                    # starting indices of the resulting pieces and 0's 
                    # elsewhere.
                    new_red = __inds_to_rows(red_start_mat, sn)
                
                else:
                    # If red_minus_purple is empty, then set new_red and
                    # red_length_vec to empty
                    new_red = np.array([]) 
                    red_length_vec = np.array([])
           
                # Noting that purple is only one part and in both red_ri and
                # blue_bi, then we need to find where the purple starting
                # indices are in all the red_ri
                purple_in_red_mat,purple_length_vec = __num_of_parts(purple, ri, \
                                                                start_red)
                blue_minus_purple = np.setdiff1d(blue_bi,purple)
                
                # If blue_minus_purple is not empty, then see if there are one
                # or two parts in blue_minus_purple. Then cut purple out of 
                # all of the repeats in blue. 
                if blue_minus_purple.size != 0: 
                    blue_start_mat, blue_length_vec = __num_of_parts(\
                                            blue_minus_purple, bi, start_blue)
                    new_blue = __inds_to_rows(blue_start_mat, sn)

                # If there are two parts left in blue_minus_purple, then the 
                # new variable new_blue, which holds the part(s) of 
                # blue_minus_purple, should have two rows with 1's for the 
                # starting indices of the resulting pieces and 0's elsewhere. 
                else:
                    # If blue_minus_purple is empty, then set new_blue and
                    # blue_length_vec to empty
                    new_blue = np.array([])
                     # Also blue_length_vec will have the length(s) of the 
                     # parts in new_blue.
                    blue_length_vec = np.array([])
                   
                # Recalling that purple is only one part and in both red_rd 
                # and blue_bi, then we need to find where the purple starting
                # indices are in all the blue_ri
                purple_in_blue_mat, purple_length = __num_of_parts(purple, bi, start_blue)
                # Union purple_in_red_mat and purple_in_blue_mat to get
                # purple_start, which stores all the purple indices
                
                purple_start = np.union1d(purple_in_red_mat[0], \
                                          purple_in_blue_mat[0])
                    
                # Use purple_start to get new_purple with 1's where the repeats
                # in the purple rows start and 0 otherwise. 
                
                new_purple = __inds_to_rows(purple_start, sn);
                if new_red.size != 0 or new_blue.size != 0:
                    # Form the outputs
                    if new_red.size != 0 and new_blue.size == 0 :
                        union_mat = np.vstack((new_red, new_purple))
                        union_length = np.vstack((red_length_vec, purple_length))
                    elif new_red.size == 0 and new_blue.size != 0 :
                        union_mat = np.vstack((new_blue, new_purple))
                        union_length = np.vstack((\
                                              blue_length_vec, purple_length))
                    else:
                        union_mat = np.vstack((new_red, new_blue,new_purple))
                        union_length = np.vstack((red_length_vec,\
                                              blue_length_vec, purple_length))
                     
                    if_merge = True
                    #union_mat, union_length = try2._merge_based_on_length(\
                                       # union_mat, union_length, union_length)
                    
                elif new_red.size == 0 and new_blue.size == 0:
                    new_purple_block = reconstruct_full_block(new_purple,\
                                                              np.array([purple_length]))
                        
                    if max(new_purple_block[0]) < 2:
                        union_mat = new_purple
                        union_length = np.array([purple_length])
    
    ifstop = check_each_row(union_mat, union_length)[0]
    
    while (ifstop==True):
        where = check_each_row(union_mat, union_length)[1]
        a,b,if_merge= cut(union_mat[where],union_length[where],union_mat[where],union_length[where])
        union_mat = np.delete(union_mat, where, axis = 0)
        union_length = np.delete(union_length,where,axis = 0)
        union_mat = np.vstack((a,union_mat))
        union_length = np.vstack((b,union_length))
        ifstop = check_each_row(union_mat, union_length)[0]
        
    output = (union_mat,union_length,if_merge)
    return output
                        
    
def _merge_based_on_length(full_mat,full_bw,target_bw):
    
    """
    Merges repeats that are the same length, as set 
    by full_bandwidth, and are repeats of the same piece of structure
        
    Args
    ----
    full_mat: np.array
        binary matrix with ones where repeats start and zeroes otherwise
        
    full_bw: np.array
        length of repeats encoded in input_mat
    
    target_bw: np.array
        lengths of repeats that we seek to merge
        
    Returns
    -------    
    out_mat: np.array
        binary matrix with ones where repeats start and zeros otherwise
        with rows of full_mat merged if appropriate
        
    one_length_vec: np.array
        length of the repeats encoded in out_mat
    """
    temp_bandwidth = np.sort(full_bw,axis=0)
    # Return the indices that would sort full_bandwidth
    bnds = np.argsort(full_bw,axis=None) 
    temp_mat = full_mat[bnds,:] 
    # Find the unique elements of target_bandwidth
    target_bandwidth = np.array(np.unique(target_bw))

    # Number of columns 
    target_size = target_bandwidth.shape[0] 
    for i in range(1,target_size+1):
        test_bandwidth = np.array([[target_bandwidth[i-1]]])
        # Check if temp_bandwidth is equal to test_bandwidth
        inds = np.array(np.where(temp_bandwidth == test_bandwidth))[0]
        # If the sum of all inds elements is greater than 1, then execute this 
        # if statement
        if inds.sum() > 1:
            # Isolate rows that correspond to test_bandwidth and merge them
            merge_bw = temp_mat[inds]
            merge_mat = _merge_rows(merge_bw,test_bandwidth)
                   
            # Number of columns
            bandwidth_add_size = merge_mat.shape[0] 
            bandwidth_add = test_bandwidth * \
            np.ones((bandwidth_add_size,1)).astype(int)
            if np.any(inds == True):
                # Convert the boolean array inds into an array of integers
                inds = np.array(inds).astype(int)
                remove_inds = np.where(inds == 1)
                # Delete the rows that meet the condition set by remove_inds
                temp_mat = np.delete(temp_mat,remove_inds,axis=0)
                temp_bandwidth = np.delete(temp_bandwidth,remove_inds,axis=0)
            # Combine rows into a single matrix
            
            if temp_mat.size!=0:                             
                bind_rows = [temp_mat,merge_mat]
                temp_mat = np.concatenate(bind_rows)                                
            else:
                temp_mat = merge_mat
            
            
            # Indicates temp_bandwidth is an empty array
            if temp_bandwidth.size == 0: 
                temp_bandwidth = np.concatenate(bandwidth_add)
            # Indicates temp_bandwidth is not an empty array
            elif temp_bandwidth.size > 0: 
                
                bind_bw = [temp_bandwidth,bandwidth_add]
                temp_bandwidth = np.vstack((bind_bw))

            temp_bandwidth = np.sort(temp_bandwidth,axis=0)
            # Return the indices that would sort temp_bandwidth
            bnds = np.argsort(temp_bandwidth,axis=None) 
            temp_mat = temp_mat[bnds,:] 
                       
            out_mat = temp_mat
            out_length_vec = temp_bandwidth
            
            output = (out_mat,out_length_vec)
    
    return output

def _merge_rows(input_mat, input_width):
    
    """
    Merges rows that have at least one common repeat; said common repeat(s)
    must occur at the same time step and be of common length
    
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
    not_merge = input_mat    # Everything must be checked
    merge_mat = np.array([])          # Nothing has been merged yet
    merge_key = np.array([])
    rows = input_mat.shape[0]  # How many rows to merge?
    # Step 1: has every row been checked?
    while rows > 0:
        # Step 2: start merge process
        # Step 2a: choose first unmerged row
        row2check = not_merge[0,:]
        # Create a comparison matrix
        # with copies of row2check stacked
        # so that r2c_mat is the same
        # size as the set of rows waiting
        # to be merged
        r2c_mat = np.kron(np.ones((rows,1)), row2check) 
        
        
        # Step 2b: find indices of unmerged overlapping rows
        merge_inds = np.sum(((r2c_mat + not_merge) == 2), axis = 1) > 0
        
       
        # Step 2c: union rows with starting indices in common with row2check 
        # and remove those rows from input_mat
        union_merge = np.sum(not_merge[merge_inds,:], axis = 0) > 0
        not_merge = np.delete(not_merge,np.where(merge_inds==1),0)
        #return not_merge
        # Step 2d: check that newly merged rows do not cause overlaps within
        # row 
        # If there are conflicts, rerun compare_and_cut

        merge_block = reconstruct_full_block(union_merge, np.array([input_width]))
        
        
        if np.max(merge_block) > 1:
            (union_merge, union_merge_key,if_merge) = cut(union_merge,\
            input_width,
            union_merge, input_width)
        else:
            union_merge_key = input_width
        # Step 2e: add unions to merge_mat and merge_key
        if merge_mat.size!=0:
            merge_mat = np.vstack((merge_mat,np.array([union_merge]))).astype(int)
        else:
            merge_mat = np.array([union_merge]).astype(int)
        if merge_mat.ndim ==3:
            merge_mat = merge_mat[0]
        if merge_key.size!= 0:
            merge_key = np.vstack((merge_key,np.array([union_merge_key])))
        else:
            merge_key = np.array([union_merge_key])
        if merge_key.ndim ==3:
            merge_key = merge_key[0]
        # Step 3: reinitialize rs for stopping condition
        rows = not_merge.shape[0]
    
    return merge_mat 

def merge(union_mat,union_length,if_merge):
    # Maybe need a while loop
    
    if if_merge:
        union_mat, union_length = _merge_based_on_length(\
                                        union_mat, union_length, union_length)
    
    # # Check that there are no overlaps in each row of union_mat 
    # union_mat_add = np.array([])
    # union_mat_add_length = np.array([])
    # union_mat_rminds = np.array([])
    
    # # Isolate one row at a time, call it union_row
    # for i in range(0, union_mat.shape[0]):
    #     union_row = union_mat[i,:]
    #     union_row_width = np.array([union_length[i]])
    #     union_row_block = reconstruct_full_block(union_row, union_row_width)
    #     # If there are at least one overlap, then compare and cut that row
    #     # until there are no overlaps
        
    #     if (np.sum(union_row_block[0]>1)) > 0:
    #         print(1)
    #         if union_mat_rminds.size!=0:
    #             union_mat_rminds = np.vstack((union_mat_rminds, i))
    #         else:
    #             union_mat_rminds = np.array([i])
    #         union_row_new, union_row_new_length,if_merge = cut(union_row,\
    #                             union_row_width, union_row, union_row_width)
            
    #         # Add union_row_new and union_row_new_length to union_mat_add and
    #         # union_mat_add_length, respectively
    #         if union_mat_add.size!=0:
                
    #             union_mat_add = np.vstack((union_mat_add, union_row_new))
    #             union_mat_add_length = np.vstack((union_mat_add_length,\
    #                                          union_row_new_length))
    #         else:
    #             union_mat_add = union_row_new
    #             union_mat_add_length = union_row_new_length
    
    #     # Remove the old rows from union_mat (as well as the old lengths from
    # # union_length)
    # if union_mat_rminds.size!=0:
    #     union_mat = np.delete(union_mat, union_mat_rminds, axis = 0)
    #     union_length = np.delete(union_length, union_mat_rminds)
     
    # #Add union_row_new and union_row_new_length to union_mat and
    # #union_length, respectively, such that union_mat is in order by
    # #lengths in union_length
    # if union_mat_add.size!=0:
    #     union_mat = np.vstack((union_mat, union_mat_add))
    # if union_mat_add_length.size!=0:
    #     union_length = np.vstack((union_length, union_mat_add_length))
    
    UM_inds = np.argsort(union_length.flatten())
    union_length = np.sort(union_length)
    union_mat = union_mat[UM_inds,:].astype(int)
    if union_mat.ndim ==3:
        union_mat = union_mat[0]
    
    return union_mat,union_length
        


# red = np.array([1,0,1,1,0,0,1,0,0,1,0,0,0])
red_len = np.array([4])
# blue = np.array([0,1,0,0,1,0,1,1,1,0,0,0,0])
blue_len = np.array([2])
red = np.array([1,1,0,0,0,0,0,1,0,0,0,0,0])
blue = np.array([0,1,0,1,0,0,0,0,1,0,1,0,0])
output = cut(red, red_len, blue, blue_len)
output1 = merge(output[0],output[1],output[2])
print(output1)
