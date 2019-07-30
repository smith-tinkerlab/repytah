#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
utilities.py 

This script when imported as a module allows search.py, disassemble.py and 
assemble.py in the ah package to run smoothly. 

This file contains the following functions:
    
    * reconstruct_full_block - Creates a record of when pairs of repeated
    structures occur, from the first beat in the song to the end. Pairs of 
    repeated structures are marked with 1's. 

    * find_initial_repeats - Finds all diagonals present in thresh_mat, 
    removing each diagonal as it is found.
    
    * add_annotations - Adds annotations to each pair of repeated structures 
    according to their order of occurence. 
    
    * create_sdm - Creates a self-dissimilarity matrix; this matrix is found 
    by creating audio shingles from feature vectors, and finding cosine 
    distance between shingles. 
    
    * reformat - Transforms a binary matrix representation of when repeats 
    occur in a song into a list of repeated structures detailing the length
    and occurence of each repeat. 
     
"""

import numpy as np

from scipy import signal

import scipy.sparse as sps

import scipy.spatial.distance as spd

def reconstruct_full_block(pattern_mat, pattern_key): 
    """
    Creates a record of when pairs of repeated structures occur, from the 
    first beat in the song to the end. This record is a binary matrix with a 
    block of 1's for each repeat encoded in pattern_mat whose length 
    is encoded in pattern_key

    Args
    ----
    pattern_mat: np.array
        binary matrix with 1's where repeats begin 
        and 0's otherwise
     
    pattern_key: np.array
        vector containing the lengths of the repeats 
        encoded in each row of pattern_mat

    Returns
    -------
    pattern_block: np.array
        binary matrix representation for pattern_mat 
        with blocks of 1's equal to the length's 
        prescribed in pattern_key
    """
    #First, find number of beats (columns) in pattern_mat: 
    #Check size of pattern_mat (in cases where there is only 1 pair of
    #repeated structures)
    if (pattern_mat.ndim == 1): 
        #Convert a 1D array into 2D array 
        #From:
        #https://stackoverflow.com/questions/3061761/numpy-array-dimensions
        pattern_mat = pattern_mat[None, : ]
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
    else: 
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
        
    #Assign number of repeated structures (rows) in pattern_mat to sb 
    sb = pattern_mat.shape[0]
    
    #Pre-allocating a sn by sb array of zeros 
    pattern_block = np.zeros((sb,sn)).astype(int)  
    
    #Check if pattern_key is in vector row 
    if pattern_key.ndim != 1: 
        #Convert pattern_key into a vector row 
        length_vec = np.array([])
        for i in pattern_key:
            length_vec = np.append(length_vec, i).astype(int)
    else: 
        length_vec = pattern_key 
    
    for i in range(sb):
        #Retrieve all of row i of pattern_mat 
        repeated_struct = pattern_mat[i,:]
    
        #Retrieve the length of the repeats encoded in row i of pattern_mat 
        length = length_vec[i]
    
        #Pre-allocate a section of size length x sn for pattern_block
        sub_section = np.zeros((length, sn))
    
        #Replace first row in block_zeros with repeated_structure 
        sub_section[0,:] = repeated_struct
        
        #Creates pattern_block: Sums up each column after sliding repeated 
        #sastructure i to the right bw - 1 times 
        for b in range(2, length + 1): 
    
            #Retrieve repeated structure i up to its (1 - b) position 
            sub_struct_a = repeated_struct[0:(1 - b)]
    
            #Row vector with number of entries not included in sub_struct_a  
            sub_struct_b = np.zeros((1,( b  - 1)))
    
            #Append sub_struct_b in front of sub_struct_a 
            new_struct = np.append(sub_struct_b, sub_struct_a)
            
            #Replace part of sub_section with new_struct 
            sub_section[b - 1,:] = new_struct
    
    #Replaces part of pattern_block with the sums of each column in 
    #sub_section 
    pattern_block[i,:] = np.sum(sub_section, axis = 0)
    
    return pattern_block
    
#line 217: 
#https://stackoverflow.com/questions/2828059/sorting-arrays-in-np-by-column

def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    """
    Identifies all repeated structures in a sequential data stream which are 
    represented as diagonals in thresh_mat and then stores the pairs of
    repeats that correspond to each repeated structure in a list. 
    
    Args
    ----
        thresh_mat: np.array[int]:
            thresholded matrix that we extract diagonals from
        
        bandwidth_vec: np.array[1D,int]:
            vector of lengths of diagonals to be found
        
        thresh_bw int:
            smallest allowed diagonal length
    
    Returns
    -------
        all_lst: np.array[int]:
            list of pairs of repeats that correspond to 
            diagonals in thresh_mat
    """

    b = np.size(bandwidth_vec)

    #Create empty lists to store arrays
    int_all =  []
    sint_all = []
    eint_all = []
    mint_all = []

    #Loop over all bandwidths
    for bw in bandwidth_vec:
        if bw > thresh_bw:
            #Search for diagonals of length bw
            thresh_mat_size = np.size(thresh_mat)
            
            DDM_rename = signal.convolve2d(thresh_mat[0:thresh_mat_size, \
                                                      0:thresh_mat_size],\
    np.eye(bw), 'valid')
            #Mark where diagonals of length bw start
            thresh_DDM_rename = (DDM_rename == bw) 
            if np.sum(np.sum(thresh_DDM_rename)) > 0:
                full_bw = bw
                #1) Non-Overlaps: Search outside the overlapping shingles

                #Find the starts that are paired together
                #returns tuple of lists (python) https://docs.scipy.org/doc/np/reference/generated/np.nonzero.html
                #need to add 1 to return correct number of nonzero ints matlab vs python
                overlaps = np.nonzero(np.triu(thresh_DDM_rename, (full_bw)))

                startI = np.array(overlaps[0])
                num_nonoverlaps = np.size(startI)
                startJ = np.array(overlaps[1])
                #Find the matching ends EI for SI and EJ for SJ
                matchI = (startI + full_bw-1);
                matchJ = (startJ + full_bw-1);

                 #List pairs of starts with their ends and the widths of the
                #non-overlapping interval

                int_lst = np.column_stack([startI, matchI, startJ, matchJ, full_bw])
                #Add the new non-overlapping intervals to the full list of
                #non-overlapping intervals
                int_all.append(int_lst)
                # 2) Overlaps: Search only the overlaps in shingles
                #returns tuple (python) 
                shingle_overlaps = np.nonzero(np.tril(np.triu(thresh_DDM_rename), (full_bw-1)))
                #gets list for I and J [1,2,3,4] turn those to np, transpose them vertically
                startI_inShingle = np.array(shingle_overlaps[0]) 
                startJ_inShingle = np.array(shingle_overlaps[1]) 
                #find number of overlaps
                num_overlaps = np.size(startI_inShingle)
                if (num_overlaps == 1 and startI_inShingle == startJ_inShingle):
                    sint_lst = np.column_stack([startI_inShingle, startI_inShingle,(startI_inShingle + (full_bw - 1)),startJ_inShingle,(startJ_inShingle + (full_bw - 1)), full_bw])
                    sint_all.append(sint_lst)
                elif num_overlaps>0:
                        #Since you are checking the overlaps you need to cut these
                        #intervals into pieces: left, right, and middle. NOTE: the
                        #middle interval may NOT exist
                    # Vector of 1's that is the length of the number of
                    # overlapping intervals. This is used a lot. 
                    ones_no = np.ones(num_overlaps);

                    #2a) Left Overlap
                    #remain consistent with being matlab -1
                    K = startJ_inShingle - startI_inShingle
                    sint_lst = np.column_stack([startI_inShingle, (startJ_inShingle - ones_no), startJ_inShingle, (startJ_inShingle + K - ones_no), K])
                    #returns list of indexes of sorted list
                    Is = np.argsort(K)
                    #turn array vertical
                    Is.reshape(np.size(Is), 1)
                    #extract all columns from row Is
                    sint_lst = sint_lst[Is, :]
                                    #grab only length column
                    i = 0
                    for length in np.transpose(sint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                        #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                sint_lst = np.delete(sint_lst, (i-1), axis=0)
                            sint_all.append(sint_lst)
                                #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1
                    #endfor
                    #2b right overlap
                    endI_right = startI_inShingle + (full_bw)
                    endJ_right = startJ_inShingle + (full_bw)
                    eint_lst = np.column_stack([(endI_right + ones_no - K), endI_right, (endI_right + ones_no), endJ_right, K])
                    indexes = np.argsort(K)
                    #turn result to column
                    indexes.reshape(np.size(indexes),1)
                    eint_lst = eint_lst[indexes, :]
                    
                    #grab only length column
                    i = 0
                    for length in np.transpose(eint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                            #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                eint_lst = np.delete(eint_lst, (i-1), axis=0)
                            eint_all.append(eint_lst)
                            #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1

                    # 2) Middle Overlap
                    #returns logical 0 or 1 for true or false
                    mnds = (endI_right - startJ_inShingle - K + ones_no) > 0
                    #for each logical operator convert to 0 or 1
                    for operator in mnds:
                        if operator is True:
                            operator = 1
                        else:
                            operator = 0
                    startI_middle = startJ_inShingle*(mnds)
                    endI_middle = (endI_right*(mnds) - K*(mnds))
                    startJ_middle = (startJ_inShingle*(mnds) + K*(mnds))
                    endJ_middle = endI_right*(mnds)
                    #fixes indexing here because length starts at 1 and indexes start at 0
                    Km = (endI_right*(mnds) - startJ_inShingle*(mnds) - K*(mnds) +ones_no*(mnds))-1
                    if np.sum(np.sum(mnds)) > 0 : 
                        mint_lst = np.column_stack([startI_middle, endI_middle, startJ_middle, endJ_middle, Km])
                        #revert for same reason
                        Km = Km+1
                        Im = np.argsort(Km)
                        #turn array to column
                        Im.reshape(np.size(Im), 1)
                        mint_lst = mint_lst[Im, :]

                       #Remove the pairs that fall below the bandwidth threshold
                        #grab only length column
                        i = 0
                        for length in np.transpose(mint_lst[:,4]):
                            #if this length is greater than thresh_bw-- we found our index
                            if length > thresh_bw:
                            #if its not the first row
                                if(i!=0):
                                    #delete rows that fall below threshold
                                    mint_lst = np.delete(mint_lst, (i-1), axis=0)
                                mint_all.append(mint_lst)
                                #after we found the min that exceeds thresh_bw... break
                                break
                            i=i+1
                        #endfor
                    #endif line 143 np.sum(np.sum(mnds)) > 0
                #endif line 67 (num_overlaps == 1 and startI_inShingle == startJ_inShingle)

                                    #returns matrix with diags in it
                SDM = stretch_diags(DDM_rename, bw)
                thresh_mat = thresh_mat - SDM

                if np.sum(np.sum(thresh_mat)) == 0:
                    break
                #endIf line 174
            #endIf line 34 np.sum(np.sum(thresh_DDM_rename)) > 0
       #endIf line 28 bw > thresh_bw
    #endfor
     #Combine non-overlapping intervals with the left, right, and middle parts
     #of the overlapping intervals
    #remove empty lines from the lists


    out_lst = int_all + sint_all + eint_all + mint_all
    #remove empty lists from final output
    
    all_lst = filter(None, out_lst)

    if out_lst is not None:
        all_lst = np.vstack(out_lst)
    else:
        all_lst = np.array([])
    #return final list
    return all_lst


def add_annotations(input_mat, song_length):
    """
    Adds annotations to the pairs of repeats in input_mat   

    Args
    ----
    input_mat: np.array
        list of pairs of repeats. The first two columns refer to 
        the first repeat of the pair. The third and fourth columns refer
        to the second repeat of the pair. The fifth column refers to the
        repeat lengths. The sixth column contains any previous annotations,
        which will be removed.
        
    song_length: int
        number of audio shingles in the song.
    
    Returns
    -------
    anno_list: array
        list of pairs of repeats with annotations marked. 
    """
    num_rows = input_mat.shape[0]
    
    # Removes any already present annotation markers
    input_mat[:, 5] = 0
    
    # Find where repeats start
    s_one = input_mat[:,0]
    s_two = input_mat[:,2]
    
    # Creates matrix of all repeats
    s_three = np.ones((num_rows,), dtype = int)
    
    up_tri_mat = sps.coo_matrix((s_three, 
                                 (s_one, s_two)), shape = (song_length + 1, 
                                 song_length + 1)).toarray()
    
    low_tri_mat = up_tri_mat.conj().transpose()
    
    full_mat = up_tri_mat + low_tri_mat
    
    # Stitches info from input_mat into a single row
    song_pattern = __find_song_pattern(full_mat)
    
    # Restructures song_pattern
    song_pattern = song_pattern[:,:-1]
    song_pattern = np.insert(song_pattern, 0, 0, axis=1)
    
    # Adds annotation markers to pairs of repeats
    for i in song_pattern[0]:
        pinds = np.nonzero(song_pattern == i)
        
        #One if annotation not already marked, 0 if it is
        check_inds = (input_mat[:,5] == 0)
        
        for j in pinds[1]:
            
            # Finds all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j) + (s_two == j)
            mark_inds = (mark_inds > 0)
            mark_inds = check_inds * mark_inds
            
            # Adds found annotations to the relevant time steps
            input_mat[:,5] = (input_mat[:,5] + i * mark_inds)
            
            # Removes pairs of repeats with annotations from consideration
            check_inds = check_inds ^ mark_inds
     
    temp_inds = np.argsort(input_mat[:,5])
    
    # Creates list of annotations
    anno_list = input_mat[temp_inds,]
    
    return anno_list

def create_sdm(fv_mat, num_fv_per_shingle):
    """
    Creates self-dissimilarity matrix; this matrix is found by creating audio 
    shingles from feature vectors, and finding cosine distance between 
    shingles
    
    Args
    ----
    fv_mat: np.array
        matrix of feature vectors where each column is a timestep and each row
        includes feature information i.e. an array of 144 columns/beats and 12
        rows corresponding to chroma values
        
    num_fv_per_shingle: int
        number of feature vectors per audio shingle
    
    Returns
    -------
    self_dissim_mat: np.array 
        self dissimilarity matrix with paired cosine distances between 
        shingles
    """
    [num_rows, num_columns] = fv_mat.shape
    if num_fv_per_shingle == 1:
        mat_as = fv_mat
    else:
        mat_as = np.zeros(((num_rows * num_fv_per_shingle),
                           (num_columns - num_fv_per_shingle + 1)))
        for i in range(1, num_fv_per_shingle+1):
            # Use feature vectors to create an audio shingle
            # for each time step and represent these shingles
            # as vectors by stacking the relevant feature
            # vectors on top of each other
            mat_as[((i-1)*num_rows+1)-1:(i*num_rows), : ] = fv_mat[:, 
                   i-1:(num_columns- num_fv_per_shingle + i)]

    sdm_row = spd.pdist(mat_as.T, 'cosine')
    self_dissim_mat = spd.squareform(sdm_row)
    return self_dissim_mat
  

def reformat(pattern_mat, pattern_key):
    """Transforms a binary array with 1's where repeats start and 0's
    otherwise into an a list of repeated stuctures. This list consists of
    information about the repeats including length, when they occur and when
    they end. 
    
    Every row has a pair of repeated structure. The first two columns are 
    the time steps of when the first repeat of a repeated structure start and 
    end. Similarly, the second two columns are the time steps of when the 
    second repeat of a repeated structure start and end. The fourth colum is 
    the length of the repeated structure. 
    
    reformat.py may be helpful when writing example inputs for aligned 
    hiearchies.
    
    Args
    ----
        pattern_mat: np.array 
            binary array with 1's where repeats start and 0's otherwise 
        
        pattern_key: np.array 
            array with the lengths of each repeated structure in pattern_mat
            
    Returns
    -------
        info_mat: np.array 
            array with the time steps of when the pairs of repeated structures 
            start and end organized 

    """

    #Pre-allocate output array with zeros 
    info_mat = np.zeros((pattern_mat.shape[0], 5))
    
    #Retrieve the index values of the repeats in pattern_mat 
    results = np.where(pattern_mat == 1)
    
    #1. Find the starting indices of the repeated structures row by row 
    for r in range(pattern_mat.shape[0]):
        #Find where the repeats start  
        r_inds = (pattern_mat[r] == 1) 
        inds = np.where(r_inds)
        
        #Retrieve the starting indices of the repeats 
        s_ij = inds[0] 
        
        #Seperate the starting indices of the repeats 
        i_ind = s_ij[0]
        j_ind = s_ij[1]
        
        #2. Assign the time steps of the repeated structures into  info_mat
        for x in results[0]:
            #If the row equals the x-value of the repeat
            if r == x:
                info_mat[r, 0] = i_ind + 1
                info_mat[r, 1] = i_ind + pattern_key[r] 
                info_mat[r, 2] = j_ind + 1 
                info_mat[r, 3] = j_ind + pattern_key[r]
                info_mat[r, 4] = pattern_key[r]
                
    return info_mat 







