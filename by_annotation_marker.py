#!/usr/bin/env python
# coding: utf-8

import numpy as np

def separate_anno_markers(k_mat, sn, band_width, pattern_row): 
    """
    Expands pattern_row into a matrix, so that there is one group of repeats per row.

    Attributes
    ----
        k_mat: np.array
            List of pairs of repeats of length BAND_WIDTH with 
            annotations marked. The first two columns refer to the
            first repeat of the pair, the second two refer to the
            second repeat of the pair, the fifth column refers to the
            length of the repeats, and the sixth column contains the
            annotation markers.
        
        sn: number
            song length, which is the number of audio shingles
        
        band_width: number 
            the length of repeats encoded in k_mat
        
        pattern_row: np.array
            row vector that marks where non-overlapping repeats
            occur, marking the annotation markers for the start 
            indices and 0's otherwise

    Returns
    -------
        pattern_mat: np.array
            matrix representation of k_mat with 
            one row for each group of repeats
        
        patter_key: np.array
            row vector containing the lengths of the repeats 
            encoded in each row of pattern_mat
        
        anno_id_lst: np.array 
            row vector containing the annotation markers of 
            the repeats encoded in each row of pattern_mat
    """

    #Example 1: 
    k_mat = np.array([[1, 1, 3, 3, 1, 1],
                      [4, 4, 6, 6, 1, 2]]) 
    sn = 6
    band_width = 1
    pattern_row = np.array([1, 0, 1, 2, 0, 2])

    #Example 2: 
    #k_mat = np.array([[1, 1, 3, 3, 1, 1]])
    #sn = 6
    #band_width = 1
    #pattern_row = np.array([1, 0, 1, 0, 0, 0])

    #List of annotation markers 
    anno_lst = k_mat[:,5] 

    #Initialize pattern_mat: Start with a matrix of all 0's that has
    #the same number of rows as there are annotations and sn columns 
    pattern_mat = np.zeros((anno_lst.size, sn), dtype = np.intp)

    #Separate the annotions into individual rows 

    if anno_lst.size > 1: #If there are two or more annotations 
        #Loops through the list of annotation markers 
        for a in anno_lst: 
            #Index the first starting index of repeat (row a - 1, column 0) 
            a_one = k_mat[a-1, 0] - 1
            #Index the second starting index of repeat(row a - 1, column 2)  
            a_two = k_mat[a-1, 2]-1
            #Appends both starting indices 
            s_inds = np.append(a_one, a_two)
            #Replace entries in pattern_mat with a "1" at the repeats' starting indices 
            pattern_mat[a - 1, s_inds] = 1
            
            #confucius 
            pattern_key = band_width * np.ones((anno_lst.size, 1)).astype(int)
            print(pattern_key)
    else: #When there is one annotation  
        pattern_mat = pattern_row 
        print(pattern_mat)
        pattern_key = band_width
        #anno_id_lst = [:anno_max] isnt this just anno_lst? 

    return(pattern_mat)








