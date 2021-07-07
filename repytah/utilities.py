#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
utilities.py 

This script when imported as a module allows search.py, transform.py and 
assemble.py in the repytah package to run smoothly. 

This file contains the following functions:
    
    * create_sdm - Creates a self-dissimilarity matrix; this matrix is found 
    by creating audio shingles from feature vectors, and finding cosine 
    distance between shingles. 
    
    * find_initial_repeats - Finds all diagonals present in thresh_mat, 
    removing each diagonal as it is found.
    
    * stretch_diags - Fills out diagonals in binary self dissimilarity matrix
    from diagonal starts and lengths.

    * add_annotations - Adds annotations to each pair of repeated structures 
    according to their length and order of occurence. 
    
    * __find_song_pattern - Stitches information about repeat locations from 
    thresh_diags matrix into a single row. 
    
    * reconstruct_full_block - Creates a record of when pairs of repeated
    structures occur, from the first beat in the song to the last beat of the
    song. Pairs of repeated structures are marked with 1's.    
        
    * get_annotation_lst - Gets one annotation marker vector, given vector of
    lengths key_lst.
    
    * get_yLabels - Generates the labels for a visualization.
    
    * reformat [Only used for creating test examples] - Transforms a binary 
    matrix representation of when repeats occur in a song into a list of 
    repeated structures detailing the length and occurence of each repeat.   
    
"""

import numpy as np
from scipy import signal
import scipy.sparse as sps
import scipy.spatial.distance as spd

def create_sdm(fv_mat, num_fv_per_shingle):
    """
    Creates self-dissimilarity matrix; this matrix is found by creating audio 
    shingles from feature vectors, and finding the cosine distance between 
    shingles.
    
    Args
    ----
        fv_mat: np.array
            Matrix of feature vectors where each column is a timestep and each 
            row includes feature information i.e. an array of 144 columns/beats
            and 12 rows corresponding to chroma values.
            
        num_fv_per_shingle: int
            Number of feature vectors per audio shingle
    
    Returns
    -------
        self_dissim_mat: np.array 
            Self dissimilarity matrix with paired cosine distances between 
            shingles
        
    """
    
    [num_rows, num_columns] = fv_mat.shape
    
    if num_fv_per_shingle == 1:
        mat_as = fv_mat
    else:
        mat_as = np.zeros(((num_rows * num_fv_per_shingle),
                           (num_columns - num_fv_per_shingle + 1)))
        for i in range(1, num_fv_per_shingle + 1):
            # Use feature vectors to create an audio shingle
            # for each time step and represent these shingles
            # as vectors by stacking the relevant feature
            # vectors on top of each other
            mat_as[((i - 1)*num_rows + 1) - 1:(i*num_rows), : ] = fv_mat[:, 
                               i - 1:(num_columns- num_fv_per_shingle + i)]

    # Build the pairwise-cosine distance matrix between audio shingles
    sdm_row = spd.pdist(mat_as.T, 'cosine')
    
    # Build self dissimilarity matrix by changing the condensed 
    # pairwise-cosine distance matrix to a redundant matrix
    self_dissim_mat = spd.squareform(sdm_row)
    
    return self_dissim_mat
  
      
def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    """
    Looks for the largest repeated structures in thresh_mat. Finds all 
    repeated structures, represented as diagonals present in thresh_mat, 
    and then stores them with their start/end indices and lengths in a 
    list. As each diagonal is found, they are removed to avoid identifying
    repeated sub-structures. 
  
    Args
    ----
        thresh_mat: np.array[int]
            Thresholded matrix that we extract diagonals from

        bandwidth_vec: np.array[1D,int]
            Vector of lengths of diagonals to be found. Should be 1,2,3,..... 
            n where n = num_timesteps

        thresh_bw: int
            Smallest allowed diagonal length

    Returns
    -------
        all_lst: np.array[int]
            List of pairs of repeats that correspond to 
            diagonals in thresh_mat
            
    """

    # Initialize the input and temporary variables
    thresh_temp = thresh_mat

    # Interval list for non-overlapping pairs    
    int_all =  np.empty((0,5), int)
    
    # Interval list for the left side of the overlapping pairs
    sint_all = np.empty((0,5), int)
    
    # Interval list for the right side of the overlapping pairs
    eint_all = np.empty((0,5), int) 
    
    # Interval list for the middle of the overlapping pairs if they exist
    mint_all = np.empty((0,5), int) 

    # Loop over all bandwidths from n to 1
    for bw in np.flip(bandwidth_vec):
        if bw > thresh_bw:
            # Use convolution matrix to find diagonals of length bw 
            id_mat = np.identity(bw) 

            # Search for diagonals of length band_width
            diagonal_mat = signal.convolve2d(thresh_temp, id_mat, 'valid')
        
            # Mark where diagonals of length band_width start
            diag_markers = (diagonal_mat == bw).astype(int)
            
            if sum(diag_markers).any() > 0:
                full_bw = bw
                
                #1) Non-Overlaps: Search outside the overlapping shingles
                upper_tri = np.triu(diag_markers, full_bw)
                
                # Search for paired starts 
                (start_i, start_j) = upper_tri.nonzero() 
                start_i = start_i + 1
                start_j = start_j + 1
              
                # Find the matching ends for the previously found starts 
                match_i = start_i + (full_bw - 1)
                match_j = start_j + (full_bw - 1)

                # List pairs of starts with their ends and the widths of the
                # non-overlapping interval
                i_pairs = np.vstack((start_i[:], match_i[:])).T
                j_pairs = np.vstack((start_j[:], match_j[:])).T
                i_j_pairs = np.hstack((i_pairs, j_pairs))
                width = np.repeat(full_bw, i_j_pairs.shape[0], axis=0)
                width_col = width.T
                int_lst = np.column_stack((i_pairs, j_pairs, width_col))
        
                # Add the new non-overlapping intervals to the full list of
                # non-overlapping intervals
                int_all = np.vstack((int_lst, int_all))
                
                # 2) Overlaps: Search only the overlaps in shingles
                
                # Search for paired starts 
                shin_ovrlaps = np.nonzero((np.tril(np.triu(diag_markers, -1),
                                                  (full_bw - 1))))
                start_i_shin = np.array(shin_ovrlaps[0] + 1) # row
                start_j_shin = np.array(shin_ovrlaps[1]  +1) # column
                num_ovrlaps = len(start_i_shin)
                
                if (num_ovrlaps == 1 and start_i_shin == start_j_shin): 
                    i_sshin = np.concatenate((start_i_shin,start_i_shin+ \
                                              (full_bw - 1)),axis = None)
                    j_sshin = np.concatenate((start_j_shin,start_j_shin+ \
                                              (full_bw - 1)),axis = None)
                    i_j_pairs = np.hstack((i_sshin,j_sshin))
                    sint_lst = np.hstack((i_j_pairs,full_bw))
                    sint_all = np.vstack((sint_all, sint_lst))
                    
                elif num_ovrlaps > 0:
                    # Since you are checking the overlaps you need to cut these
                    # intervals into pieces: left, right, and middle. NOTE: the
                    # middle interval may NOT exist
                    
                    # Vector of 1's that is the length of the number of
                    # overlapping intervals. This is used a lot. 
                    ones_no = np.ones(num_ovrlaps);

                    # 2a) Left Overlap
                    K = start_j_shin - start_i_shin
                    
                    i_sshin = np.vstack((start_i_shin[:], (start_j_shin[:] -\
                                                           ones_no[:]))).T
                    j_sshin = np.vstack((start_j_shin[:], (start_j_shin[:] + \
                                                           K - ones_no[:]))).T
                    sint_lst = np.column_stack((i_sshin,j_sshin,K.T))
                    
                    i_s = np.argsort(K) # Return the indices that would sort K
                    sint_lst = sint_lst[i_s,]
                    
                    # Remove the pairs that fall below the bandwidth threshold
                    cut_s = np.argwhere((sint_lst[:,4] > thresh_bw))
                    cut_s = cut_s.T
                    sint_lst = sint_lst[cut_s][0]
    
                    # Add the new left overlapping intervals to the full list
                    # of left overlapping intervals
                    sint_all = np.vstack((sint_all,sint_lst))
                    
                    # 2b) Right Overlap
                    end_i_shin = start_i_shin + (full_bw - 1)
                    end_j_shin = start_j_shin + (full_bw - 1)
                
                    i_eshin = np.vstack((end_i_shin[:] + ones_no[:] - K, \
                                         end_i_shin[:])).T
                    j_eshin = np.vstack((end_i_shin[:] + ones_no[:], \
                                         end_j_shin[:])).T
                    eint_lst = np.column_stack((i_eshin,j_eshin,K.T))
                
                    i_e = np.lexsort(K) # Return the indices that would sort K
                    eint_lst = eint_lst[i_e:,]
                    
                    # Remove the pairs that fall below the bandwidth threshold
                    cut_e = np.argwhere((eint_lst[:,4] > thresh_bw))
                    cut_e = cut_e.T
                    eint_lst = eint_lst[cut_e][0]
    
                    # Add the new right overlapping intervals to the full list 
                    # of right overlapping intervals
                    eint_all = np.vstack((eint_all,eint_lst))

                    # 2) Middle Overlap
                    
                    mnds = (end_i_shin - start_j_shin - K + ones_no) > 0
                
                    if sum(mnds) > 0:
                        i_middle = (np.vstack((start_j_shin[:], \
                                               end_i_shin[:] - K ))) * mnds
                        i_middle = i_middle.T
                        i_middle = i_middle[np.all(i_middle != 0, axis=1)]
                        
                        
                        j_middle = (np.vstack((start_j_shin[:] + K, \
                                               end_i_shin[:])))  * mnds 
                        j_middle = j_middle.T
                        j_middle = j_middle[np.all(j_middle != 0, axis=1)]
                        
                        
                        k_middle = np.vstack((end_i_shin[mnds] - \
                                              start_j_shin[mnds] - K[mnds] \
                                              + ones_no[mnds]))
                        k_middle = k_middle.T
                        k_middle = k_middle[np.all(k_middle != 0, axis=1)]
        
                        mint_lst = np.column_stack((i_middle, j_middle, 
                                                    k_middle.T))
                                                

                        # Remove the pairs that fall below the bandwidth 
                        # threshold 
                        cut_m = np.argwhere((mint_lst[:,4] > thresh_bw))
                        cut_m = cut_m.T
                        mint_lst = mint_lst[cut_m][0]
                    
                        mint_all = np.vstack((mint_all, mint_lst))

            # Remove found diagonals of length BW from consideration
            SDM = stretch_diags(diag_markers, bw)
            thresh_temp = np.logical_xor(thresh_temp,SDM)

            if thresh_temp.sum() == 0:
                break
    
    # Combine all found pairs of repeats
    out_lst = np.vstack((sint_all, eint_all, mint_all))
    all_lst = np.vstack((int_all, out_lst))
    
    # Sort the output array first by repeat length, then by starts of i and 
    # finally by j
    inds = np.lexsort((all_lst[:,2],all_lst[:,0],all_lst[:,4]))
    all_lst = np.array(all_lst)[inds]
    
    return(all_lst.astype(int))


def stretch_diags(thresh_diags, band_width):
    """
    Creates binary matrix with full length diagonals from binary matrix of
    diagonal starts and length of diagonals
                                                                                 
    Args
    ----
        thresh_diags: np.array
            Binary matrix where entries equal to 1 signal the existence 
            of a diagonal
        
        band_width: int
            Length of encoded diagonals
    
    Returns
    -------
        stretch_diag_mat: np.array [boolean]
            Logical matrix with diagonals of length band_width starting 
            at each entry prescribed in thresh_diag
        
    """

    # Creates size of returned matrix
    n = thresh_diags.shape[0] + band_width - 1
    temp_song_marks_out = np.zeros(n)
    (jnds, inds) = thresh_diags.nonzero()
    
    subtemp = np.identity(band_width)
    
    # Expands each entry in thresh_diags into diagonal of
    # length band width
    for i in range(inds.shape[0]):
        tempmat = np.zeros((n,n))
        tempmat[inds[i]:(inds[i] + band_width), 
                jnds[i]:(jnds[i] + band_width)] = subtemp
        temp_song_marks_out = temp_song_marks_out + tempmat
                
    # Ensures that stretch_diag_mat is a binary matrix
    stretch_diag_mat = (temp_song_marks_out > 0)
    
    return stretch_diag_mat


def add_annotations(input_mat, song_length):
    """
    Adds annotations to the pairs of repeats in input_mat.

    Args
    ----
        input_mat: np.array
            List of pairs of repeats. The first two columns refer to 
            the first repeat of the pair. The third and fourth columns refer
            to the second repeat of the pair. The fifth column refers to the
            repeat lengths. The sixth column contains any previous annotations,
            which will be removed
            
        song_length: int
            Number of audio shingles in the song
    
    Returns
    -------
        anno_list: array
            List of pairs of repeats with annotations marked
        
    """

    num_rows = input_mat.shape[0]
    
    # Removes any already present annotation markers
    input_mat[:,5] = 0
    
    # Find where repeats start
    s_one = input_mat[:,0]  
    s_two = input_mat[:,2]

    # Creates matrix of all repeats
    s_three = np.ones((num_rows,), dtype = int)
    
    up_tri_mat = sps.coo_matrix((s_three, 
                                 (s_one - 1, s_two - 1)), shape = (song_length, 
                                 song_length)).toarray()
    
    low_tri_mat = up_tri_mat.conj().transpose()
    
    full_mat = up_tri_mat + low_tri_mat
    
    # Stitches info from input_mat into a single row
    song_pattern = __find_song_pattern(full_mat)
    SPmax = max(song_pattern)
    
    # Adds annotation markers to pairs of repeats
    for i in range(1, SPmax + 1):
        pinds = np.nonzero(song_pattern == i)     
      
        # One if annotation not already marked, zero if it is
        check_inds = (input_mat[:,5] == 0)
        
        for j in pinds[0]:
            # Finds all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j + 1) + (s_two == j + 1)  
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


def __find_song_pattern(thresh_diags):
    """
    Stitches information from thresh_diags matrix into a single
    row, song_pattern, that shows the timesteps containing repeats;
    From the full matrix that decodes repeat beginnings (thresh_diags),
    the locations, or beats, where these repeats start are found and
    encoded into the song_pattern array

    Args
    ----
        thresh_diags: np.array
            Binary matrix with 1 at the start of each repeat pair (SI,SJ) and 
            0 elsewhere. 
            WARNING: must be symmetric
    
    Returns
    -------
        song_pattern: np.array [shape = (1, song_length)]
            Row where each entry represents a time step and the group 
            that time step is a member of
        
    """

    song_length = thresh_diags.shape[0]
    
    # Initialize song pattern base
    pattern_base = np.zeros((1,song_length), dtype = int).flatten()
   
    # Initialize group number
    pattern_num = 1

    col_sum = thresh_diags.sum(axis = 0)
    check_inds = col_sum.nonzero() 
    check_inds = check_inds[0]
    
    # Creates vector of song length
    pattern_mask = np.ones((1, song_length))
    pattern_out = (col_sum == 0)
    pattern_mask = (pattern_mask - pattern_out).astype(int).flatten()
    
    while np.size(check_inds) != 0:
        # Takes first entry in check_inds
        i = check_inds[0] 
        
        # Takes the corresponding row from thresh_diags
        temp_row = thresh_diags[i,:]
        
        # Finds all time steps that i is close to
        inds = temp_row.nonzero()
        
        if np.size(inds) != 0:
            while np.size(inds) != 0:
                # Takes sum of rows corresponding to inds and
                # multiplies the sums against p_mask
                c_mat = np.sum(thresh_diags[inds,:], axis = 1).flatten()
                c_mat = c_mat*pattern_mask
                
                # Finds nonzero entries of c_mat
                c_inds = c_mat.nonzero()
                
                # Gives all elements of c_inds the same grouping 
                # number as i
                pattern_base[c_inds] = pattern_num
               
                # Removes all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                pattern_mask[c_inds] = 0
               
                # Resets inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                inds = np.array([inds])
                
            # Updates grouping number to prepare for next group
            pattern_num = pattern_num + 1
            
        # Removes i from check_inds
        check_inds = np.setdiff1d(check_inds, i)
       
    song_pattern = pattern_base
    
    return song_pattern


def reconstruct_full_block(pattern_mat, pattern_key): 
    """
    Creates a record of when pairs of repeated structures occur, from the 
    first beat in the song to the end. This record is a binary matrix with a 
    block of 1's for each repeat encoded in pattern_mat whose length is 
    encoded in pattern_key.
    
    Args
    ----
        pattern_mat: np.array
            Binary matrix with 1's where repeats begin 
            and 0's otherwise
        
        pattern_key: np.array
            Vector containing the lengths of the repeats 
            encoded in each row of pattern_mat

    Returns
    -------
        pattern_block: np.array
            Binary matrix representation for pattern_mat 
            with blocks of 1's equal to the length's 
            prescribed in pattern_key
        
    """
    
    # First, find number of beats (columns) in pattern_mat: 
    # Check size of pattern_mat (in cases where there is only 1 pair of
    # repeated structures)
    if (pattern_mat.ndim == 1): 
        #Convert a 1D array into 2D array 
        pattern_mat = pattern_mat[None, : ]
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
    else: 
        #Assign number of beats to sn 
        sn = pattern_mat.shape[1]
        
    # Assign number of repeated structures (rows) in pattern_mat to sb 
    sb = pattern_mat.shape[0]
    
    # Pre-allocating a sn by sb array of zeros 
    pattern_block = np.zeros((sb,sn)).astype(int)  
    
    # Check if pattern_key is in vector row 
    if pattern_key.ndim != 1: 
        #Convert pattern_key into a vector row 
       length_vec = pattern_key.flatten()
    else: 
        length_vec = pattern_key 
    
    for i in range(sb):
        # Retrieve all of row i of pattern_mat 
        repeated_struct = pattern_mat[i,:]
    
        # Retrieve the length of the repeats encoded in row i of pattern_mat 
        length = length_vec[i]
    
        # Pre-allocate a section of size length x sn for pattern_block
        sub_section = np.zeros((length, sn))
    
        # Replace first row in block_zeros with repeated_structure 
        sub_section[0,:] = repeated_struct
        
        # Creates pattern_block: Sums up each column after sliding repeated 
        # structure i to the right bw - 1 times 
        for b in range(2, length + 1): 
            # Retrieve repeated structure i up to its (1 - b) position 
            sub_struct_a = repeated_struct[0:(1 - b)]
    
            # Row vector with number of entries not included in sub_struct_a  
            sub_struct_b = np.zeros((1,( b - 1)))
    
            # Append sub_struct_b in front of sub_struct_a 
            new_struct = np.append(sub_struct_b, sub_struct_a)

            # Replace part of sub_section with new_struct 
            sub_section[b - 1,:] = new_struct

        # Replaces part of pattern_block with the sums of each column in 
        # sub_section 
        pattern_block[i,:] = np.sum(sub_section, axis = 0)
    
    return pattern_block


def get_annotation_lst (key_lst):
    """
    Creates one annotation marker vector, given vector of lengths key_lst.
    
    Args 
    -----
        key_lst: np.array[int]
            Array of lengths in ascending order
    
    Returns 
    -----
        anno_lst_out: np.array[int] 
            Array of one possible set of annotation markers for key_lst
            
    """

    # Initialize the temporary variable
    num_rows = np.size(key_lst)
    full_anno_lst = np.zeros(num_rows)

    # Find the first instance of each length and give it 1 as an annotation
    # marker
    unique_keys = np.unique(key_lst,return_index=True)
    full_anno_lst[unique_keys[1]] = 1
        
    # Add remaining annotations to anno list  
    for i in range (0,np.size(full_anno_lst)):
        if full_anno_lst[i] == 0:
           full_anno_lst[i] =  full_anno_lst[i - 1] + 1
    
    return full_anno_lst.astype(int)


def get_yLabels(width_vec, anno_vec):   
    """
    Generates the labels for a visualization with width_vec and anno_vec.
    
    Args 
    -----
        width_vec: np.array[int]
            Vector of widths for a visualization
            
        anno_vec: np.array[int]
            Array of annotations for a visualization
    
    Returns 
    -----
        ylabels: np.array[str] 
            Labels for the y-axis of a visualization
        
    """

    # Determine number of rows to label
    num_rows = np.size(width_vec)
    # Make sure the sizes of width_vec and anno_vec are the same
    assert(num_rows == np.size(anno_vec))
    
    # Initialize the array with 0 as the origin
    ylabels = np.array([0])
    
    # Loop over the array adding labels
    for i in range(0,num_rows):
        label = ('w = '+str(width_vec[i][0].astype(int)) + 
                 ', a = '+str(anno_vec[i]))
        ylabels = np.append(ylabels, label )
    
    return ylabels


def reformat(pattern_mat, pattern_key):
    """
    Transforms a binary array with 1's where repeats start and 0's
    otherwise into a list of repeated stuctures. This list consists of
    information about the repeats including length, when they occur and when
    they end. 
    
    Every row has a pair of repeated structure. The first two columns are 
    the time steps of when the first repeat of a repeated structure start and 
    end. Similarly, the second two columns are the time steps of when the 
    second repeat of a repeated structure start and end. The fifth column is 
    the length of the repeated structure. 
    
    Reformat is not used in the main process for creating the
    aligned-hierarchies. It is helpful when writing example inputs for 
    the tests.
    
    Args
    ----
        pattern_mat: np.array 
            Binary array with 1's where repeats start and 0's otherwise 
        
        pattern_key: np.array 
            Array with the lengths of each repeated structure in pattern_mat
            
    Returns
    -------
        info_mat: np.array 
            Array with the time steps of when the pairs of repeated structures 
            start and end organized 

    """

    # Pre-allocate output array with zeros 
    info_mat = np.zeros((pattern_mat.shape[0], 5))
    
    # Retrieve the index values of the repeats in pattern_mat 
    results = np.where(pattern_mat == 1)
    
    for x,j in zip(range(pattern_mat.shape[0]),(range(0, 
                                                results[0].size - 1,2))):
            
            # Assign the time steps of the repeated structures into info_mat
            info_mat[x,0] = results[1][j] + 1
            info_mat[x,1] = info_mat[x,0]+pattern_key[x] - 1
            info_mat[x,2] = results[1][j+1] + 1
            info_mat[x,3] = info_mat[x,2]+pattern_key[x] - 1
            info_mat[x,4] = pattern_key[x] 
            
    return info_mat.astype(int)