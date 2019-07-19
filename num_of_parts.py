#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def num_of_parts(input_vec, input_start, input_all_starts):
    """
    Determines the number of blocks of consecutive time steps
    in a given list of time steps 
    
    Args
    ----
        input_vec: np.array 
            one or two parts to replicate 
            
        input_start: np.array index 
            starting index for part to be replicated 
        
        input_all_starts: np.array indices 
            starting indices for replication 
    
    Returns
    -------
        start_mat: np.array 
            matrix of one or two rows, containing 
            the starting indices 
            
        length_vec: np.array 
            column vector of the lengths 
    """
    #Initialize Inputs 
    #Example 1 
    input_vec = np.array([1, 2, 3, 4])
    input_start = 1 
    input_all_starts = np.array([1, 5, 9])
    
    #diff_vec = input_vec(2:end) - input_vec(1:end-1);
    print(input_vec[1:])
    print(input_vec[:-1])
    #diff_vec = input_vec[1:-1] - input_vec[:-2]
    diff_vec = np.subtract(input_vec[1:], input_vec[:-1])
    print(diff_vec)
    #diff_vec = [1,diff_vec];
    #diff_vec = 
    break_mark = diff_vec > 1
    print(break_mark)
    print(sum(break_mark))
    
    if sum(break_mark) == 0: 
        start_vec = input_vec[0]
        print("start_vec\n", start_vec)
        end_vec = input_vec[-1]
        print("end_vec\n", end_vec)
        add_vec = start_vec - input_start
        print(add_vec)
        start_mat = input_all_starts + add_vec
        print(start_mat)
    else:
        start_vec = np.zeros((2,1))
        end_vec =  np.zeros((2,1))
    
        start_vec[0] = input_vec[0]
        end_vec[0] = input_vec[break_mark - 2]
    
        start_vec[1] = input_vec[break_mark - 1]
        end_vec[1] = input_vec[-1]
    
        add_vec = start_vec - input_start
        start_mat = np.concatenate((input_all_starts + add_vec[0]), (input_all_starts + add_vec[1]))

    length_vec = end_vec - start_vec + 2
        
    output = (start_mat, length_vec)
    
    return output 