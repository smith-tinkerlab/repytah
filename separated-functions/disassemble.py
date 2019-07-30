# -*- coding: utf-8 -*-
"""
disassemble.py

This script contains functions that separate and disassemble inputs into 
smaller or more specific pieces. They focus mainly on overlaps and annotation
markers.

This file contains the following functions:
    
    * create_anno_remove_overlaps - Turns rows of repeats into marked rows with 
    annotation markers for the start indices and zeroes otherwise. After 
    removing the annotations that have overlaps, creates separate arrays
    for annotations with overlaps and annotations without overlaps. Finally,
    the annotation markers are checked and fixed if necessary.
    
    * create_anno_rows - Turns rows of repeats into marked rows with annotation
    markers for start indices and zeroes otherwise. Then checks if the correct 
    annotation markers were given and fixes the markers if necessary.
    
    * remove_overlaps - Removes any pairs of repeats with the same length and 
    annotation marker where at least one pair of repeats overlap in time
    
    * separate_anno_markers - Expands vector of non-overlapping repeats into
    a matrix representation. The matrix representation is a visual recored of
    where all of the repeats in a song start and end.
"""

import numpy as np

def create_anno_remove_overlaps(k_mat,song_length,band_width):
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
        list of pair of repeats with annotations marked
    
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
    overlap_lst = []
    bw = band_width
    
    # Step 0a: Find the number of distinct annotations
    anno_lst = k_mat[:,5] # Get the elements of k_mat's fifth column
    anno_max = anno_lst.max(0) # Max in each column
    
    # Step 1: Loop over the annotations
    for a in range (1,anno_max+1):
        # Step 1a: Add 1's to pattern_row to the time steps where repeats with
        # annotation a begin
        ands = (anno_lst == a) # Check if anno_lst is equal to a
        bind_rows = [k_mat[ands,0],k_mat[ands,2]]
        start_inds = np.concatenate(bind_rows)
        pattern_row[0,start_inds-1] = a

        # Step 2: check annotation by annotation
        # Start with row of 0's
        good_check = np.zeros((1,song_length)).astype(int) 
        good_check[0,start_inds-1] = 1 # Add 1 to all time steps where repeats 
                                       # with annotation a begin
    
        # Using reconstruct_full_block to check for overlaps
        block_check = reconstruct_full_block(good_check,bw)

        # If there are any overlaps, remove the bad annotations from both
        # the pattern_row and from the k_lst_out
        if block_check.max() > 1:
            # Remove the bad annotations from pattern_row
            pattern_row[0,start_inds-1] = 0
    
            # Remove the bad annotations from k_lst_out and add them to 
            # overlap_lst
            remove_inds = ands

            temp_add = k_mat[remove_inds,:]
            overlap_lst.append(temp_add)
            
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

    output = (pattern_row,k_lst_out,overlap_lst)
    
    return output


def create_anno_rows(k_mat,song_length):
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
    
    # Convert L to python list of np arrays
    L = []
    for i in range(0,(np.shape(input_mat)[0])-1):
        L.append(np.array(input_mat[i,:]))

    # Sort list ascending, then reverse it
    bw_vec = np.sort(bw_vec)
    bw_vec = bw_vec[::-1]

    mat_NO = []
    key_NO = []
    anno_NO = []
    all_overlap_lst = []
    
    # While bw_vec still has entries
    while np.size(bw_vec) != 0:
        bw_lst = []
        bw = bw_vec[0]
        # Extract pairs of repeats of length BW from the list of pairs of
        # repeats with annotation markers
        # Create bw_lst
        i = 0              
        while i < len(L):
            line = L[i][4]
            if line == bw:
                bw_lst.append(line)
                L[i] = np.array([])
            i=i+1
        #endWhile
        
    # Remove blanked entries from L (appended to bw_lst)

        # Doesn't like elem wise comparison when right operand numpy array
        L = list(filter(lambda L: L.tolist() != [], L))
        if bw > 1:
    #         Use LIGHTUP_PATTERN_ROW_GB to do the following three things:
    #         ONE: Turn the BW_LST into marked rows with annotation markers for 
    #             the start indices and 0's otherwise 
    #         TWO: After removing the annotations that have overlaps, output
    #              BW_LST_OUT which only contains rows that have no overlaps
    #         THREE: The annotations that have overlaps get removed from 
    #                BW_LST_OUT and gets added to ALL_OVERLAP_LST
    
            tuple_of_outputs = create_anno_remove_overlaps(bw_lst, 
                                                           song_length, bw)
            
            pattern_row = tuple_of_outputs[0]
            bw_lst_out = tuple_of_outputs[1]
            overlap_lst = tuple_of_outputs[2]


            # Convert the numpy arrays to lists of 1d numpy arrays
            bw_lst_out_py = []
            for i in range(0,(np.shape(bw_lst_out)[0])-1):
                bw_lst_out_py.append(np.array(input_mat[i,:]))

            overlap_lst_py = []
            for i in range(0,(np.shape(overlap_lst)[0])-1):
                overlap_lst_py.append(np.array(input_mat[i,:]))

            # If there are lines to add
            if len(overlap_lst_py) != 0:
                # Add them               
                all_overlap_lst.extend(overlap_lst_py)
        else:
            # Similar to the IF case -- 
            # Use LIGHTUP_PATTERN_ROW_BW_1 to do the following two things:
            # ONE: Turn the BW_LST into marked rows with annotation markers for 
            #      the start indices and 0's otherwise 
            # TWO: In this case, there are no overlaps. Then BW_LST_OUT is just
            #      BW_LST. Also in this case, THREE from above does not exist
            tuple_of_outputs = create_anno_rows(bw_lst, song_length)
            pattern_row =  tuple_of_outputs[0]
            bw_lst_out_orig =  tuple_of_outputs[1]
            
            # Convert the numpy arrays to lists of 1d numpy arrays
            bw_lst_out_py = []
            for i in range(0,(np.shape(bw_lst_out)[0])-1):
                bw_lst_out_py.append(np.array(input_mat[i,:]))

            overlap_lst_py = []
            for i in range(0,(np.shape(overlap_lst)[0])-1):
                overlap_lst_py.append(np.array(input_mat[i,:]))

        if np.max(np.max(pattern_row)) > 0:
            # Separate ALL annotations. In this step, we expand a row into a
            # matrix, so that there is one group of repeats per row.
            
            tuple_of_outputs = separate_anno_markers(bw_lst_out, 
                                                        song_length, bw, 
                                                        pattern_row)
            pattern_mat = tuple_of_outputs[0]
            pattern_key = tuple_of_outputs[1]
            anno_temp_lst = tuple_of_outputs[2]
 
    
            # Convert the numpy arrays to lists of 1d numpy arrays
            pattern_mat_py = []
            for i in range(0,(np.shape(pattern_mat)[0])-1):
                pattern_mat_py.append(np.array(pattern_mat[i,:]))

            pattern_key_py = []
            for i in range(0,(np.shape(pattern_key)[0])-1):
                pattern_key_py.append(np.array(pattern_key[i,:]))


            anno_temp_lst_py = []
            for i in range(0,(np.shape(anno_temp_lst)[0])-1):
                anno_temp_lst_py.append(np.array(anno_temp_lst[i,:]))


        else:
            pattern_mat = []
            pattern_key = []

        
        if np.sum(np.sum(pattern_mat)) > 0:
            # If there are lines to add, add them
            if np.shape(mat_NO)[0] != 0:
                mat_NO.append(pattern_mat)
            if np.shape(key_NO)[0] != 0:
                key_NO.append(pattern_key)
            if np.shape(anno_NO)[0] != 0:
                anno_NO.append(anno_temp_lst)


        # Add to L
        L.append(bw_lst_out_py)
        # Sort list by 5th column
        # Create dict to re-sort L
        re_sort_L = {}
        for i in range(0, len(L)-1):
            # Get 5th column values into list of tuples
            # Key = index, value = value
            re_sort_L[i] = (L[i])[4]
        # Convert to list of tuples and sort
        re_sort_L = re_sort_L.items()
        # Sort that dict by values  
        re_sort_L = sorted(re_sort_L, key=lambda re_sort_L: re_sort_L[1])

        
        sorted_inds = [x[0] for x in re_sort_L]
        # Sort L according to sorted indexes
        L = [L for sorted_inds, L in sorted(zip(sorted_inds, L))]

        # Will just use a np array here
        np_mat_L = np.array(L)
        bw_vec = np.unique(np_mat_L[:,4])
        
        # Sort list ascending, then reverse it
        bw_vec = np.sort(bw_vec)
        bw_vec = bw_vec[::-1]
        # Remove entries that fall below the bandwidth threshold
        cut_index = 0

        for value in bw_vec:
        # If the value is above the bandwidth 
            if value < bw:
                cut_index = cut_index+1
        #endfor
        bw_vec = bw_vec[cut_index:np.shape(bw_vec)[0]]

    #endWhile

    # Set the outputs
    lst_no_overlaps = np.array(L)
    
    # Turn key_NO, mat_NO, and KEY_NO to numpy lists
    key_NO = list(filter(lambda key_NO: key_NO.tolist() != [], key_NO))
    mat_NO = list(filter(lambda mat_NO: mat_NO.tolist() != [], mat_NO))
    anno_NO = list(filter(lambda anno_NO: anno_NO.tolist() != [], anno_NO))

    if len(key_NO) !=0:
        key_NO = np.concatenate(key_NO)
    else:
        key_NO = np.array([])
        
    if len(mat_NO) !=0:
        mat_NO = np.concatenate(mat_NO)
    else:
        mat_NO = np.array([])
        
    if len(anno_NO) !=0:
        anno_NO = np.concatenate(anno_NO)
    else:
        anno_NO = np.array([])

    # Convert to np.array
    all_overlap_lst = np.array(all_overlap_lst)
    if np.shape(all_overlap_lst)[0] != 0:
        overlap_inds = np.argsort(all_overlap_lst[:,4])
        all_overlap_lst = all_overlap_lst[overlap_inds, :]
    #endif
    
    key_NO = np.sort(key_NO)
    mat_inds = np.argsort(key_NO)
    if np.shape(mat_NO)[0] != 0:
        matrix_no_overlaps = mat_NO[mat_inds,:]
    else:
        matrix_no_overlaps = mat_NO
        
    key_no_overlaps = key_NO
    if np.shape(anno_NO)[0] != 0:
        annotations_no_overlaps = mat_NO[mat_inds,:]
    else:
        annotations_no_overlaps = mat_NO
        
    # Compile final outputs to a tuple
    output_tuple = (lst_no_overlaps, matrix_no_overlaps, key_no_overlaps, 
                    annotations_no_overlaps, all_overlap_lst)
   
    return output_tuple

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

