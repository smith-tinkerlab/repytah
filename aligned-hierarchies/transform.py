# -*- coding: utf-8 -*-
"""
transform.py
This script contains functions that take inputs and transform them to be of use in 
bigger functions where they are called. They focus mainly on overlapping
repeated structures and annotation markers.
This file contains the following functions:
    
    * __create_anno_remove_overlaps - Turns rows of repeats into marked rows with 
    annotation markers for the start indices and zeroes otherwise. After 
    removing the annotations that have overlaps, creates separate arrays
    for annotations with overlaps and annotations without overlaps. Finally,
    the annotation markers are checked and fixed if necessary.
    
    * __create_anno_rows - Turns rows of repeats into marked rows with annotation
    markers for start indices and zeroes otherwise. Then checks if the correct 
    annotation markers were given and fixes the markers if necessary.
    
    * remove_overlaps - Removes any pairs of repeats with the same length and 
    annotation marker where at least one pair of repeats overlap in time
    
    * __separate_anno_markers - Expands vector of non-overlapping repeats into
    a matrix representation. The matrix representation is a visual recored of
    where all of the repeats in a song start and end.
"""

import numpy as np
from utilities import reconstruct_full_block
from utilities import add_annotations

def __create_anno_remove_overlaps(k_mat,song_length,band_width):
    """
    Turn k_mat into marked rows with annotation markers for the start indices 
    and zeroes otherwise. After removing the annotations that have overlaps, 
    output k_lst_out which only contains rows that have no overlaps. Then 
    take the annotations that have overlaps from k_lst_out and put them in
    overlap_lst. Lastly, check if the proper sequence of annotation markers 
    was given and fix them if necessary.
    
    Args
    ----
    k_mat: np.array
        List of pairs of repeats of length 1 with annotations 
        marked. The first two columns refer to the first repeat
        of the pair, the second two refer to the second repeat of
        the pair, the fifth column refers to the length of the
        repeats, and the sixth column contains the annotation markers.
    
    song_length: int
        number of audio shingles
    
    band_width: int
        the length of repeats encoded in k_mat
    
    Returns
    -------
    pattern_row: np.array
        row that marks where non-overlapping repeats occur, 
        marking the annotation markers for the start indices 
        and 0's otherwise
    
    k_lst_out: np.array
        list of pairs of repeats of length band_width that 
        contain no overlapping repeats with annotations
        marked
    
    overlap_lst: np.array
        list of pairs of repeats of length band_width that
        contain overlapping repeats with annotations marked
    """
    
    # Step 0: Initialize outputs: Start with a vector of all 0's for
    #         pattern_row and assume that the row has no overlaps
    pattern_row = np.zeros((1,song_length)).astype(int)
    overlap_lst = np.array([])
    bw = band_width
    
    # Step 0a: Find the number of distinct annotations
    anno_lst = k_mat[:,5] # Get the elements of k_mat's fifth column
    anno_max = anno_lst.max(0) # Max in each column
    
    # print("anno_lst:",anno_lst)
    # print("anno_max:",anno_max)
    
    # Step 1: Loop over the annotations
    for a in range (1,anno_max+1):
        # Step 1a: Add 1's to pattern_row to the time steps where repeats with
        # annotation a begin
        ands = (anno_lst == a) # Check if anno_lst is equal to a
        bind_rows = [k_mat[ands,0],k_mat[ands,2]]
        start_inds = np.concatenate(bind_rows)
        pattern_row[0,(start_inds-1)] = a

        # Step 2: check annotation by annotation
        # Start with row of 0's
        good_check = np.zeros((1,song_length)).astype(int) 
        good_check[0,start_inds-1] = 1 # Add 1 to all time steps where repeats 
                                       # with annotation a begin
        
        bw = np.array(bw).flatten()
        # Using reconstruct_full_block to check for overlaps
        block_check = reconstruct_full_block(good_check,bw)
        
        #print(block_check)
        
        # If there are any overlaps, remove the bad annotations from both
        # the pattern_row and from the k_lst_out
        if block_check.max() > 1:
            # Remove the bad annotations from pattern_row
            pattern_row[0,start_inds-1] = 0
    
            # Remove the bad annotations from k_lst_out and add them to 
            # overlap_lst
            remove_inds = ands

            temp_add = k_mat[remove_inds,:]
            
            if (overlap_lst.size == 0):
                overlap_lst = temp_add
            else:
                overlap_lst = np.vstack((overlap_lst,temp_add))
            
            if np.any(remove_inds == True):
                # Convert the boolean array rm_inds into an array of integers
                remove_inds = np.array(remove_inds).astype(int)
                remove = np.where(remove_inds==1)
                
                # Delete the row that meets the condition set by remove_inds
                k_mat = np.delete(k_mat,remove,axis=0)
                
            anno_lst = k_mat[:,5]
           
    inds_markers = np.unique(pattern_row)
    
    # If any of inds_markers[i] is equal to zero, then remove this index
    if np.any(inds_markers == 0):
        inds_markers = np.delete(inds_markers,0)

    # If inds_markers is not empty, then execute this if statement
    if inds_markers.size > 0:
        for na in range(1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in pattern_row
                temp_anno = (pattern_row == IM)
                pattern_row = pattern_row - (IM * temp_anno) + (na * temp_anno)

    # If k_mat is not empty, then execute this if statement
    if k_mat.size > 0:
        k_lst_out = np.unique(k_mat,axis=0)
        for na in range(1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in k_lst_out
                kmat_temp_anno = (k_lst_out[:,5] == IM)
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + \
                (na * kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    # Edit the annotations in the overlap_lst so that the annotations start
    # with 1 and increase one each time
    if overlap_lst.size > 0:
        overlap_lst = np.unique(overlap_lst,axis=0)
        overlap_lst = add_annotations(overlap_lst, song_length)
    
    output = (pattern_row.flatten(),k_lst_out,overlap_lst)
    
    return output


def __create_anno_rows(k_mat,song_length):
    """
    Turn the k_mat into marked rows with annotation markers for the start 
    indices and zeroes otherwise. Check if the proper sequence of annotation 
    markers was given and fix them if necessary.
    Args
    ----
    k_mat: np.array
        List of pairs of repeats of length 1 with annotations 
        marked. The first two columns refer to the first repeat
        of the pair, the second two refer to the second repeat of
        the pair, the fifth column refers to the length of the
        repeats, and the sixth column contains the annotation markers.
                 
   song_length: int
        song length, which is the number of audio shingles
   
    Returns
    ------- 
    pattern_row: np.array
        row that marks where non-overlapping repeats
        occur, marking the annotation markers for the
        start indices and zeroes otherwise.
    k_lst_out: np.array
        list of pairs of repeats of length BAND_WIDTH that
        contain no overlapping repeats with annotations marked.
    """
    # Step 0 Initialize outputs: Start with a vector of all 0's for 
    #       pattern_row and assume that the row has no overlaps 
    pattern_row = np.zeros((1,song_length)).astype(int)
    
    # Step 0a: Find the number of distinct annotations
    anno_lst = k_mat[:,5] # Get the elements of k_mat's fifth column
    anno_max = anno_lst.max(0) # Set the number of max elements in each column
    
    # Step 1: Loop over the annotations
    for a in range(1,anno_max+1):
        ands = (anno_lst == a) # Check if anno_lst is equal to a 
        
        # Combine rows into a single matrix
        bind_rows = [k_mat[ands,0],k_mat[ands,2]]
        start_inds = np.concatenate(bind_rows)
        pattern_row[0,start_inds-1] = a
    
    # Step 2: Check that in fact each annotation has a repeat associated to it
    inds_markers = np.unique(pattern_row)

    # If any of inds_markers[i] == 0, then delete this index
    if np.any(inds_markers == 0):
        inds_markers = np.delete(inds_markers,0)

    if inds_markers.size > 0:
        for na in range (1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in pattern_row
                temp_anno = (pattern_row == IM)
                pattern_row = pattern_row - (IM * temp_anno) + (na * temp_anno)
    
    # Edit the annotations to match the annotations in pattern_row
    if k_mat.size > 0:
        k_lst_out = np.unique(k_mat, axis=0)
        for na in range (1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotaions in k_lst_out
                kmat_temp_anno = (k_lst_out[:,5] == IM)
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + \
                (na*kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    output = (pattern_row,k_lst_out)
    
    return output


def remove_overlaps(input_mat, song_length):  
    """
    Removes any pairs of repeat length and specific annotation marker 
        where there exists at least one pair of repeats that do
        overlap in time.

    Args
    ----
    input_mat: np.array(int)
         List of pairs of repeats with annotations marked. The first 
         two columns refer to the first repeat or the pair, the second 
         two refer to the second repeat of the pair, the fifth column 
         refers to the length of the repeats, and the sixth column 
         contains the annotation markers.
         
    song_length: int
         the number of audio shingles
 
    Returns
    -------
    lst_no_overlaps: np.array(int)
        List of pairs of repeats with annotations marked. All the 
        repeats of a given length and with a specific annotation 
        marker do not overlap in time.
        
    matrix_no_overlaps: np.array(int)
        Matrix representation of lst_no_overlaps with one row for 
        each group of repeats
        
    key_no_overlaps: np.array(int)
        Vector containing the lengths of the repeats encoded in 
        each row of matrix_no_overlaps
        
    annotations_no_overlaps: np.array(int)
        Vector containing the annotation markers of the repeats 
        encoded in each row of matrix_no_overlaps
        
    all_overlap_lst: np.array(int)
        List of pairs of repeats with annotations marked removed 
        from input_mat. For each pair of repeat length and specific 
        annotation marker, there exist at least one pair of repeats 
        that do overlap in time.
    """
    # Same list with repetitions removed
    bw_vec = np.unique(input_mat[:,4])
    
    L = input_mat

    # Sort list ascending, then reverse it
    bw_vec = np.sort(bw_vec)
    bw_vec = bw_vec[::-1]
    

    mat_NO = np.empty([0, song_length])
    key_NO = np.empty([0, 1])
    
    anno_NO = np.empty([0, 1])
    all_overlap_lst = np.array([[0,0,0,0,0,0]])
    
# While bw_vec still has entries
    while np.size(bw_vec) != 0:

        bw_lst = np.array([])
        #set bandwidth value
        bw = bw_vec[0]
        
        deleteArray = []
        
        for i in range(len(L)):
            #print(L[i][4])
            linebw = L[i][4]
            if linebw == bw:
                if bw_lst.size == 0:
                    bw_lst = np.array([L[i]])
                    
                else:
                    bw_lst=np.vstack((bw_lst,L[i]))
                    
                deleteArray.append(i)

        L = np.delete(L, deleteArray, 0)
        
        if bw > 1:
    #         Use LIGHTUP_PATTERN_ROW_GB to do the following three things:
    #         ONE: Turn the BW_LST into marked rows with annotation markers for 
    #             the start indices and 0's otherwise 
    #         TWO: After removing the annotations that have overlaps, output
    #              BW_LST_OUT which only contains rows that have no overlaps
    #         THREE: The annotations that have overlaps get removed from 
    #                BW_LST_OUT and gets added to ALL_OVERLAP_LST
            
            tuple_of_outputs = __create_anno_remove_overlaps(bw_lst, song_length, bw)
            
            pattern_row = tuple_of_outputs[0]
            bw_lst_out = tuple_of_outputs[1]
            overlap_lst = tuple_of_outputs[2]
        
            
            if overlap_lst.size>0:
                all_overlap_lst = np.vstack((all_overlap_lst, overlap_lst))
        
        else:            
            # Similar to the IF case -- 
            # Use LIGHTUP_PATTERN_ROW_BW_1 to do the following two things:
            # ONE: Turn the BW_LST into marked rows with annotation markers for 
            #      the start indices and 0's otherwise 
            # TWO: In this case, there are no overlaps. Then BW_LST_OUT is just
            #      BW_LST. Also in this case, THREE from above does not exist
            tuple_of_outputs = __create_anno_rows(bw_lst, song_length)
            pattern_row =  tuple_of_outputs[0]
            bw_lst_out =  tuple_of_outputs[1]
            
        
        if np.max(np.max(pattern_row)) > 0:
            # Separate ALL annotations. In this step, we expand a row into a
            # matrix, so that there is one group of repeats per row.
            
            tuple_of_outputs = __separate_anno_markers(bw_lst_out, 
                                                        song_length, bw, 
                                                        pattern_row)
            pattern_mat = tuple_of_outputs[0]
            pattern_key = np.array(tuple_of_outputs[1])
            anno_temp_lst = tuple_of_outputs[2]
            
        
        else:
            pattern_mat = []
            pattern_key = []
        
        if np.sum(np.sum(pattern_mat)) > 0:
            
            # If there are lines to add, add them
            mat_NO = np.vstack((mat_NO,pattern_mat))
            
            key_NO = np.vstack((key_NO,pattern_key))

            anno_NO = np.vstack((anno_NO,anno_temp_lst))

        
        if (bw_lst_out.size > 0):
            L = np.vstack((L,bw_lst_out))
        
        # create a new, sorted array
        ind = np.lexsort((L[:,0], L[:,4]))
        L = L[ind]
        
        bw_vec = np.unique(L[:,4])
        bw_vec = np.sort(bw_vec)
        bw_vec = bw_vec[::-1]
        
        # Remove entries that fall below the bandwidth threshold
        cut_index = 0

        for value in bw_vec:
        # If the value is above the bandwidth 
            if value >= bw:
                cut_index = cut_index+1

        bw_vec = bw_vec[cut_index:np.shape(bw_vec)[0]]

    if mat_NO.size>0:
    
        
        masterArray = np.hstack((mat_NO,key_NO,anno_NO))
        cNum = masterArray.shape[1]
    
        ind = np.lexsort((masterArray[:,cNum-1], masterArray[:,cNum-2]))
        masterArray = masterArray[ind]
        
        matrix_no_overlaps = masterArray[:,:(cNum-2)]

        key_no_overlaps = masterArray[:,(cNum-2)]
        
        annotations_no_overlaps = masterArray[:,(cNum-1)]
    else:
        matrix_no_overlaps = mat_NO
        key_no_overlaps = key_NO
        annotations_no_overlaps = anno_NO
        
    #setting the outputs
    lst_no_overlaps = L
    
    all_overlap_lst = np.delete(all_overlap_lst,0,0)
    
    output = (lst_no_overlaps,matrix_no_overlaps.astype(int),\
              key_no_overlaps.astype(int), annotations_no_overlaps.astype(int),\
                  all_overlap_lst)
    
    return output


def __separate_anno_markers(k_mat, sn, band_width, pattern_row): 
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
    anno_max = max(anno_lst)
    
    #Initialize pattern_mat: Start with a matrix of all 0's that has
    #the same number of rows as there are annotations and sn columns 
    pattern_mat = np.zeros((anno_max, sn), dtype = np.intp)

    #Separate the annotions into individual rows 
    if anno_max > 1: #If there are two or more annotations 
        #Loops through the list of annotation markers 
        for a in range (1,anno_max+1): 
           
        #Find starting indices:  
            ands = (anno_lst == a)
            a_starts = np.concatenate((k_mat[ands,0], k_mat[ands,2]), axis=None)
            #Replace entries at each repeats' start time with "1"
            pattern_mat[a-1, a_starts-1] = 1;
        
        #Creates row vector with the same dimensions of anno_lst   
        pattern_key = band_width * np.ones((anno_max, 1)).astype(int)


    else: #When there is one annotation  
        pattern_mat = pattern_row 
        pattern_key = band_width
    
    anno_id_lst = np.arange(anno_max).reshape(anno_max,1)+1
    output = (pattern_mat, pattern_key, anno_id_lst)
    
    return output


