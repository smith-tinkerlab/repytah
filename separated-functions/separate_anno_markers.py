#!/usr/bin/env python
# coding: utf-8

import numpy as np

def separate_anno_markers(k_mat, sn, band_width, pattern_row): 
    """
    Expands pattern_row, a row vector that marks where non-overlapping
    repeats occur, into a matrix representation or np.array. The dimension of 
    this array is twice the pairs of repeats by the length of the song (sn). 
    k_mat provides a list of annotation markers that is used in separating the 
    repeats of length band_width into individual rows. Each row will mark the 
    start and end time steps of a repeat with 1's and 0's otherwise. The array 
    is a visual record of where all of the repeats in a song start and end.

    Args
    ----
        k_mat: np.array
            List of pairs of repeats of length BAND_WIDTH with annotations 
            marked. The first two columns refer to the start and end time
            steps of the first repeat of the pair, the second two refer to 
            the start and end time steps of second repeat of the pair, the 
            fifth column refers to the length of the repeats, and the sixth 
            column contains the annotation markers. We will be indexing into 
            the sixth column to obtain a list of annotation markers. 
        
        sn: number
            song length, which is the number of audio shingles
        
        band_width: number 
            the length of repeats encoded in k_mat
        
        pattern_row: np.array
            row vector of the length of the song that marks where 
            non-overlapping repeats occur with the repeats' corresponding 
            annotation markers and 0's otherwise

    Returns
    -------
        pattern_mat: np.array
            
            matrix representation where each row contains a group of repeats
            marked 
        
        patter_key: np.array
            column vector containing the lengths of the repeats encoded in 
            each row of pattern_mat
        
        anno_id_lst: np.array 
            column vector containing the annotation markers of the repeats 
            encoded in each row of pattern_mat
    """
    
    #List of annotation markers 
    anno_lst = k_mat[:,5] 

    #Initialize pattern_mat: Start with a matrix of all 0's that has
    #the same number of rows as there are annotations and sn columns 
    pattern_mat = np.zeros((anno_lst.size, sn), dtype = np.intp)

    #Separate the annotions into individual rows 
    if anno_lst.size > 1: #If there are two or more annotations 
        #Loops through the list of annotation markers 
        for a in anno_lst: 
        #Find starting indices:  
            #Start index of first repeat a 
            a_one = k_mat[a-1, 0] - 1

            #Start index of second repeat a
            a_two = k_mat[a-1, 2] - 1

            #Start indices of repeat a 
            s_inds = np.append(a_one, a_two)

            #Replace entries at each repeats' start time with "1"
            pattern_mat[a - 1, s_inds] = 1

        #Creates row vector with the same dimensions of anno_lst   
        pattern_key = band_width * np.ones((anno_lst.size, 1)).astype(int)

    else: #When there is one annotation  
        pattern_mat = pattern_row 
        pattern_key = band_width
        
    #Transpose anno_lst from a row vector into a column vector 
    anno_id_lst = anno_lst.reshape((1,2)).transpose()
    
    output = (pattern_mat, pattern_key, anno_id_lst)
    
    return output 







