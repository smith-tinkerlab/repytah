# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial.distance as spd
import scipy.sparse as sps
from scipy import signal
from inspect import signature 

def add_annotations(input_mat, song_length):
    """
    Adds annotations to pairs of repeats in input matrix

    Args
    ----
    input_mat: array
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
        list of pairs of repeats with annotations marked
    """
    num_rows = input_mat.shape[0]
    
    # Removes any already present annotation markers
    input_mat[:, 5] = 0
    
    # Find where repeats start
    s_one = input_mat[:,0]
    s_two = input_mat[:,2]
    
    # Creates matrix of all repeats
    s_three = np.ones((num_rows,), dtype = int)
    
    up_tri_mat = sps.coo_matrix((s_three, (s_one, s_two)),
                                shape = (song_length, song_length)).toarray()
    
    low_tri_mat = up_tri_mat.conj().transpose()
    
    full_mat = up_tri_mat + low_tri_mat
    
    # Stitches info from input_mat into a single row
    song_pattern = find_song_pattern(full_mat)
    
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


def breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, thresh_bw):
    """
    Distills repeats encoded in input_pattern_obj and bw_vec to the 
        essential structure components, the set of repeats so that 
        no time step is contained in more than one repeat.
    
    Args
    ----
    input_pattern_obj: np.array 
        binary matrix with 1's where repeats begin 
        and 0's otherwise 
        
    bw_vec: np.array 
        vector containing the lengths of the repeats
        encoded in input_pattern_obj
        
    thresh_bw: int
        the smallest allowable repeat length 
        
    Returns
    -------
    pattern_no_overlaps: np.array 
        binary matrix with 1's where repeats of 
        essential structure components begin 
        
    pattern_no_overlaps_key: np.array 
        vector containing the lengths of the repeats of 
        essential structure components in pattern_no_overlaps 
    """
    sig = signature(breakup_overlaps_by_intersect)
    params = sig.parameters 
    if len(params) < 3: 
        T = 0 
    else: 
        T = thresh_bw
    
    # Initialize input_pattern_obj 
    PNO = input_pattern_obj
    
    # Sort the bw_vec and the PNO so that we process the biggest pieces first
    
    # Part 1: Sort the lengths in bw_vec in descending order 
    sort_bw_vec = np.sort(bw_vec)
    desc_bw_vec = sort_bw_vec[::-1]
    
    # Part 2: Sort the indices of bw_vec in descending order 
    bw_inds = np.argsort(desc_bw_vec, axis = 0)
    row_bw_inds = np.transpose(bw_inds)
    
    
    PNO = PNO[(row_bw_inds),:]
    PNO = PNO.reshape(4,19)
    
    T_inds = np.nonzero(bw_vec == T) 
    T_inds = np.array(T_inds) - 1  # Bends is converted into an array

    if T_inds.size != 0: 
        T_inds = max(bw_vec.shape) - 1

    PNO_block = reconstruct_full_block(PNO, desc_bw_vec)
    
    # Check stopping condition -- Are there overlaps?
    while np.sum(PNO_block[ : T_inds, :]) > 0:
        # Find all overlaps by comparing the rows of repeats pairwise
        overlaps_PNO_block = check_overlaps(PNO_block)
        
        # Remove the rows with bandwidth T or less from consideration
        overlaps_PNO_block[T_inds:, ] = 0
        overlaps_PNO_block[:,T_inds:] = 0
        
        # Find the first two groups of repeats that overlap, calling one group
        # RED and the other group BLUE
        [ri,bi] = overlaps_PNO_block.nonzero() 
        [ri,bi] = np.where(overlaps_PNO_block !=0)
        
        red = PNO[ri,:]
        RL = bw_vec[ri,:]
        
        blue = PNO[bi,:]
        BL = bw_vec[bi,:]
        
        # Compare the repeats in RED and BLUE, cutting the repeats in those
        # groups into non-overlapping pieces
        union_mat, union_length = compare_and_cut(red, RL, blue, BL)
        
        PNO = np.delete(PNO, ri, axis = 0)
        PNO = np.delete(PNO, bi, axis = 0)

        bw_vec = np.delete(bw_vec, ri, axis = 0)
        bw_vec = np.delete(bw_vec, bi, axis = 0)
        
        PNO = np.vstack(PNO, union_mat) 
        bw_vec = np.vstack(bw_vec, union_length)
        
        # Check there are any repeats of length 1 that should be merged into
        # other groups of repeats of length 1 and merge them if necessary
        if sum(union_length == 1) > 0:
            PNO, bw_vec = merge_based_on_length(PNO, bw_vec, 1)
        
        # AGAIN, Sort the bw_vec and the PNO so that we process the biggest 
        # pieces first
        # Part 1: Sort the lengths in bw_vec in descending order 
        sort_bw_vec = np.sort(bw_vec)
        desc_bw_vec = sort_bw_vec[::-1]
        # Part 2: Sort the indices of bw_vec in descending order 
        bw_inds = np.argsort(desc_bw_vec, axis = 0)
        row_bw_inds = np.transpose(bw_inds)
        
        PNO = PNO[(row_bw_inds),:]
        
        # Find the first row that contains repeats of length less than T and
        # remove these rows from consideration during the next check of the
        # stopping condition
        #T_inds = np.nonzeros(bw_vec == T, 1) 
        T_inds = np.amin(desc_bw_vec == T) - 1
        T_inds = np.array(T_inds) # Bends is converted into an array

        if T_inds.size != 0:  
            T_inds = max(desc_bw_vec.shape) - 1

        PNO_block = reconstruct_full_block(PNO, desc_bw_vec)
    
    # Sort the lengths in bw_vec in ascending order 
    bw_vec = np.sort(desc_bw_vec)
    # Sort the indices of bw_vec in ascending order     
    bw_inds = np.argsort(desc_bw_vec)
   
    pattern_no_overlaps = PNO[bw_inds,:]
    pattern_no_overlaps_key = bw_vec
        
    output = (pattern_no_overlaps, pattern_no_overlaps_key)
    
    return output 


def check_overlaps(input_mat):
    """
    Compares every pair of rows from input matrix and checks for overlaps
        between those pairs
    
    Args
    ----
    input_mat: np.array(int)
        matrix to be checked for overlaps
    
    Returns
    -------
    overlaps_yn: np.array(bool)
        logical array where (i,j) = 1 if row i of input matrix and row j
        of input matrix overlap and (i,j) = 0 elsewhere
    """
    # Get number of rows and columns
    rs = input_mat.shape[0]
    ws = input_mat.shape[1]

    # R_LEFT -- Every row of INPUT_MAT is repeated RS times to create a 
    # submatrix. We stack these submatrices on top of each other.
    compare_left = np.zeros(((rs*rs), ws))
    for i in range(rs):
        compare_add = input_mat[i,:]
        compare_add_mat = np.tile(compare_add, (rs,1))
        a = (i)*rs
        #python is exclusive... will return start_index to end_
        #index-1
        b = ((i+1)*rs)    
        compare_left[a:b, :] = compare_add_mat
        #endfor

    # R_RIGHT -- Stack RS copies of INPUT_MAT on top of itself
    compare_right = np.tile(input_mat, (rs,1))

    # If INPUT_MAT is not binary, create binary temporary objects
    compare_left = compare_left > 0
    compare_right = compare_right > 0


    # Empty matrix to store overlaps
    compare_all = np.zeros((compare_left.shape[0], 1))
    # For each row
    for i in range(compare_left.shape[0]):
        # Create new counter
        num_overlaps = 0
        for j in range(compare_left.shape[1]):
            if compare_left[i,j] ==1  and compare_right[i,j] == 1:
                #inc count
                num_overlaps = num_overlaps+1
            #endif
        #endinnerFor and now append num_overlaps to matrix
        compare_all[i,0] = num_overlaps

    compare_all = (compare_all > 0)
    overlap_mat = np.reshape(compare_all, (rs, rs))
    #print("overla-Mst: \n", overlap_mat)
    # If OVERLAP_MAT is symmetric, only keep the upper-triangular portion. If
    # not, keep all of OVERLAP_MAT.
    check_mat = np.allclose(overlap_mat, overlap_mat.T)
    if(check_mat):
        overlap_mat = np.triu(overlap_mat,1)

    # endif
    overlaps_yn = overlap_mat
    
    return overlaps_yn


def compare_and_cut(red, RL, blue, BL):
    """
    Compares two rows of repeats labeled RED and BLUE, and determines 
        if there are any overlaps in time between them. If there is, 
        then we cut the repeats in RED and BLUE into up to 3 pieces. 
    
    Args
    ----
    red: np.array 
        binary row vector encoding a set of repeats with 1's where each
        repeat starts and 0's otherwise 
            
    red_len: int
        length of repeats encoded in red 
            
    blue: np.array 
        binary row vector encoding a set of repeats with 1's where each
        repeat starts and 0's otherwise 
            
    blue_len: int
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
    
    output = (union_mat, union_length) 
    
    return output 


def create_sdm(fv_mat, num_fv_per_shingle):
    """
    Creates audio shingles from feature vectors, finds cosine 
        distance between shingles, and returns self dissimilarity matrix
    
    Args
    ----
    fv_mat: np.array
        matrix of feature vectors where each column is a timestep
        
    num_fv_per_shingle: int
        number of feature vectors per audio shingle
    
    Returns
    -------
    self_dissim_mat: np.array 
        self dissimilarity matrix with paired cosine distances between shingles
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
            mat_as[((i-1) * num_rows):((i * num_rows)),] = fv_mat[:,
                   (i-1):((num_columns - num_fv_per_shingle + i))]
            
    sdm_row = spd.pdist(mat_as, 'cosine')
    self_dissim_mat = spd.squareform(sdm_row)
    
    return self_dissim_mat


def fcl_anno_only(pair_list, song_length):
    """
    Finds annotations for all pairs of repeats found in previous step
    
    Args
    ----
    pair_list: np.array
        list of pairs of repeats
        WARNING: bandwidths must be in ascending order
        
    song_length: int
        number of audio shingles in song
        
    Returns
    -------
    out_lst: np.array
        list of pairs of repeats with smaller repeats added and with
        annotation markers
    """
    # Find list of unique repeat lengths
    bw_found = np.unique(pair_list[:,4])
    bw_num = bw_found.shape[0]
    
    # Remove longest bandwidth row if it is the length of the full song
    if song_length == bw_found[bw_num - 1]:
        pair_list[-1,:] = []
        bw_found[-1] = []
        bw_num = (bw_num - 1)
    p = pair_list.shape[0]
    
    # Add annotation markers to each pair of repeats
    full_list = []
    for j in range(bw_num):
        band_width = bw_found[j]
        # Isolate pairs of repeats of desired length
        bsnds = np.amin(np.nonzero(pair_list[:,4] == band_width))
        bends = np.nonzero(pair_list[:,4] > band_width)
        
        if np.size(bends) > 0:
            bends = np.amin(bends)
        else:
            bends = p
        
        bw_mat = np.array((pair_list[bsnds:bends,]))
        bw_mat_length = bw_mat.shape[0]
        
        temp_anno_mat = np.concatenate((bw_mat, (np.zeros((bw_mat_length,1)))),
                                       axis = 1).astype(int)

        # Get annotations for this bandwidth
        temp_anno_list = add_annotations(temp_anno_mat, song_length)
        full_list.append(temp_anno_list)
        
    out_list = np.concatenate(full_list)
        
    return out_list



def find_add_erows(lst_no_anno, check_inds, k):
    """
    Finds diagonals of length K that end at the same time step as 
        previously found repeats of length K. 

    Args
    ----
    lst_no_anno: np.array
        list of pairs of repeats
        
    check_inds: np.array
        list of ending indices for repeats of length k that we use 
        to check lst_anno_no for more repeats of length k
        
    k: int
        length of repeats that we are looking for 

    Returns
    -------
    add_rows: np.array
        list of newly found pairs of repeats of length k that are 
        contained in larger repeats in lst_no_anno
    """
    # List of pairs of repeats 
    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 25, 41, 55, 15]])
    # Ending indices of length k (length of repeat we are looking for)
    check_inds = np.array([10, 55, 40, 55])
    # Length of repeat we are looking for 
    k = 10
    L = lst_no_anno
    # Logical, which pairs of repeats have length greater than k? (T return 1, F return 0)
    search_inds = (L[:,4] > k)

    # Multiply ending index of all repeats "I" by search_inds
    EI = np.multiply(L[:,1], search_inds)
    # Multipy ending index of all repeats "J" by search_inds
    EJ = np.multiply(L[:,3], search_inds)

    # Loop over CHECK_INDS
    for i in range(check_inds.size): 
        #print()
        ci = check_inds[i]
        #print("loop:", i, "ci:", ci)
        
    # Left Check: Check for CI on the left side of the pairs
        # Check if the end index of the repeat "I" equals CI
        lnds = (EI == ci) 
        #print("lnds:", lnds)
        
        # Find new rows 
        if lnds.sum(axis = 0) > 0: #If the sum across (row) is greater than 0 
            # Find the 3rd entry of the row (lnds) whose starting index of repeat "J" equals CI
            EJ_li = L[lnds,3]
            
            # Number of rows in EJ_li 
            l_num = EJ_li.shape[0] 
            #print("l_num:", l_num)
            
            # Found pair of repeats on the left side
            # l_add = np.concatenate((L[lnds,1] - k + 1, L[lnds,1], (EJ_li - k + 1), EJ_li, k*np.ones((l_num,1)
            one_lsi = L[lnds,1] - k + 1     #Starting index of found repeat i
            one_lei = L[lnds,1]             #Ending index of found repeat i
            one_lsj = EJ_li - k + 1         #Starting index of found repeat j
            one_lej = EJ_li                 #Ending index of found repeat j
            one_lk = k*np.ones((l_num,1))   #Length of found pair of repeats, i and j 
            l_add = np.concatenate((one_lsi, one_lei, one_lsj, one_lej, one_lk), axis = None)
            #print("l_add:", l_add)
            
            # Found pair of repeats on the right side
            # l_add_left = np.concatenate((L[lnds,0], (L[lnds,1] - k), L[lnds,2], (EJ_li - k), (L[lnds,4] - k)), axis = None)
            two_lsi = L[lnds,0]             #Starting index of found repeat i 
            two_lei = L[lnds,1] - k         #Ending index of ofund repeat i
            two_lsj = L[lnds,2]             #Starting index of found repeat j 
            two_lej = EJ_li - k             #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats, i and j 
            l_add_left = np.concatenate((two_lsi, two_lei, two_lsj, two_lej, two_lk), axis = None)
            #print("l_add_right:", l_add_right)
            
            # Stack the found rows vertically 
            add_rows = np.vstack((l_add, l_add_left))
            
            # Stack all the rows found on the left side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0)
            #print("add_rows:", add_rows)
            
    # Right Check: Check for CI on the right side of the pairs
        # Check if the end index of the right repeat of the pair equals CI
        rnds = (EJ == ci)
        #print("rnds:", rnds)
        
        # Find new rows
        if rnds.sum(axis = 0) > 0: #If the sum across (row) is greater than 0 
            # Find the 1st entry of the row (lnds) whose ending index of repeat "I" equals CI
            EI_ri = L[rnds, 1]
            # Number of rows in EJ_ri                    
            r_num = EI_ri.shape[0]
                               
            # Found pair of repeats on the left side 
            # r_add = np.concatenate(((EI_ri - k + 1), EI_ri, (L[rnds, 3] - k + 1), L[rnds,3], k*np.ones((r_rum, 1))), axis = None)
            one_rsi = EI_ri - k + 1         #Starting index of found repeat i 
            one_rei = EI_ri                 #Ending index of found repeat i 
            one_rsj = L[rnds, 3] - k + 1    #Starting index of found repeat j
            one_rej = L[rnds,3]             #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1))  #Length of found pair or repeats, i and j 
            r_add = np.concatenate((one_rsi, one_rei, one_rsj, one_rej, one_rk), axis = None)
            
            # Found pairs on the right side 
            r_add_left = np.concatenate((L[rnds, 0], (EI_ri - k), L[rnds, 2], (L[rnds, 3] - k), L[rnds, 4] - k), axis = None) 
            two_rsi = L[rnds, 0]            #Starting index of found repeat i  
            two_rei = EI_ri - k             #Ending index of found repeat i 
            two_rsj = L[rnds, 2]            #Starting index of found repeat j
            two_rej = L[rnds, 3] - k        #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats, i and j 
            r_add_right = np.concatenate((two_rsi, two_rei, two_rsj, two_rej, two_rk), axis = None) 
            
            # Stack the found rows vertically 
            add_rows = np.vstack((r_add, r_add_right))
            
            # Stack all the rows found on the right side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)
            #print(add_rows)                   
    return add_rows


def find_add_mrows(lst_no_anno, check_inds, k): 
    """
    Finds diagonals of length k that neither start nor end at the 
        same time steps as previously found repeats of length k. 
        
    Args
    ----
    lst_no_anno: np.array 
        list of pairs of repeats
        
    check_inds: np.array
        list of ending indices for repeats of length k that we use to 
        check lst_no_anno for more repeats of length k
        
    k: int
        length of repeats that we are looking for 
        
    Returns
    -------
    add_rows: np.array
        list of newly found pairs of repeats of length K that are 
        contained in larger repeats in LST_NO_ANNO 
    """
    #Initialize list of pairs 
    L = lst_no_anno 
    #Logical, which pair of repeats has a length greater than k (T returns 1, F returns 0)
    search_inds = (L[:,4] > k)
    
    #Multiply the starting index of all repeats "I" by search_inds
    SI = np.multiply(L[:,0], search_inds)

    #Multiply the starting index of all repeats "J" by search_inds
    SJ = np.multiply(L[:,2], search_inds)

    #Multiply the ending index of all repeats "I" by search_inds
    EI = np.multiply(L[:,1], search_inds)

    #Multiply the ending index of all repeats "J" by search_inds
    EJ = np.multiply(L[:,3], search_inds)
    
    #Loop over CHECK_INDS 
    for i in range(check_inds.size): 
        ci = check_inds[i]
        #Left Check: check for CI on the left side of the pairs
        lnds = ((SI < ci) + (EI > (ci + k -1)) == 2)
        #Check that SI < CI and that EI > (CI + K - 1) indicating that there
        #is a repeat of length k with starting index CI contained in a larger
        #repeat which is the left repeat of a pair
        if lnds.sum(axis = 0) > 0:
            #Find the 2nd entry of the row (lnds) whose starting index of the repeat "I" equals CI 
            SJ_li = L[lnds,2]
            EJ_li = L[lnds,3]
            l_num = SJ_li.shape[0]

            #Left side of left pair
            l_left_k = ci*np.ones(l_num,1) - L[lnds,0]
            l_add_left = np.concatenate((L[lnds,0], (ci - 1 * np.ones((l_num,1))), SJ_li, (SJ_li + l_left_k - np.ones((l_num,1))), l_left_k), axis = None)

            # Middle of left pair
            l_add_mid = np.concatenate(((ci*np.ones((l_num,1))), (ci+k-1)*np.ones((l_num,1)), SJ_li + l_left_k, SJ_li + l_left_k + (k-1)*np.ones((l_num,1)), k*np.ones((l_num,1))), axis = None) 

            # Right side of left pair
            l_right_k = np.concatenate((L[lnds, 1] - ((ci + k) - 1) * np.ones((l_num,1))), axis = None)
            l_add_right = np.concatenate((((ci + k)*np.ones((l_num,1))), L[lnds,1], (EJ_li - l_right_k + np.ones((l_num,1))), EJ_li, l_right_k), axis = None)

            # Add the found rows        
            add_rows = np.vstack((l_add_left, l_add_mid, l_add_right))
            #add_rows = np.reshape(add_rows, (3,5))

        #Right Check: Check for CI on the right side of the pairs
        rnds = ((SJ < ci) + (EJ > (ci + k - 1)) == 2); 

        #Check that SI < CI and that EI > (CI + K - 1) indicating that there
        #is a repeat of length K with starting index CI contained in a larger
        #repeat which is the right repeat of a pair
        if rnds.sum(axis = 0) > 0:
            SI_ri = L[rnds,0]
            EI_ri = L[rnds,1]
            r_num = SI_ri.shape[0]

            #Left side of right pair
            r_left_k = ci*np.ones((r_num,1)) - L[rnds,2]
            r_add_left = np.concatenate((SI_ri, (SI_ri + r_left_k - np.ones((r_num,1))), L[rnds,3], (ci - 1)*np.ones((r_num,1)), r_left_k), axis = None)

            #Middle of right pair
            r_add_mid = np.concatenate(((SI_ri + r_left_k),(SI_ri + r_left_k + (k - 1)*np.ones((r_num,1))), ci*np.ones((r_num,1)), (ci + k - 1)*np.ones((r_num,1)), k*np.ones((r_num,1))), axis = None)

            #Right side of right pair
            r_right_k = L[rnds, 3] - ((ci + k) - 1)*np.ones((r_num,1))
            r_add_right = np.concatenate((EI_ri - r_right_k + np.ones((r_num,1)),EI_ri, (ci + k)*np.ones((r_num,1)), L[rnds,3], r_right_k), axis = None)

            add_rows = np.vstack((r_add_left, r_add_mid, r_add_right))
            #add_rows = np.reshape(add_rows, (3,5))

            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)
     
    return add_rows 


def find_add_srows(lst_no_anno, check_inds, k):
    """
    Finds diagonals of length k that start at the same time step 
        as previously found repeats of length k. 
        
    Args
    ----
    lst_no_anno: np.array 
        list of pairs of repeats
        
    check_inds: np.array
        list of ending indices for repeats of length k that we 
        use to check lst_no_anno for more repeats of length k 
       
    k: int
        length of repeats that we are looking for
            
    Returns
    -------
    add_rows: np.array
        List of newly found pairs of repeats of length K that are 
        contained in larger repeats in lst_no_anno
            
    """
    # List of pairs of repeats 
    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 25, 41, 55, 15]])

    # Ending indices of length k (length of repeat we are looking for)
    check_inds = np.array([10, 55, 40, 55])

    # Length of repeat we are looking for 
    k = 10
    L = lst_no_anno

    # Logical, which pair of repeats has a length greater than k (T returns 1, F returns 0)
    search_inds = (L[:,4] > k)

    # Multipy the starting index of all repeats "I" by search_inds
    SI = np.multiply(L[:,0], search_inds)

    # Multiply the starting index of all repeats "J" by search_inds
    SJ = np.multiply(L[:,2], search_inds)

    # Loop over check_inds
    for i in range(check_inds.size):
        ci = check_inds[i] 
            
    # Left check: check for CI on the left side of the pairs 
        # Check if the starting index of repeat "I" of pair of repeats "IJ" equals CI
        lnds = (SI == ci) 
        print("lnds:", lnds)
        #print(lnds.sum(axis = 0))
        
        # If the sum across (row) is greater than 0 
        if lnds.sum(axis = 0) > 0: 
            # Find the 2nd entry of the row (lnds) whose starting index of repeat "I" equals CI 
            SJ_li = L[lnds, 2] 
            #print("SJ_li", SJ_li)
            
            # Used for the length of found pair of repeats 
            l_num = SJ_li.shape[0] #Dim 0 corresponds to rows, wouldn't l_num always be 1? 
            #print("l_num:", l_num)
        
            # Found pair of repeats on the left side 
            one_lsi = L[lnds, 0]            #Starting index of found repeat i
            one_lei = L[lnds, 0] + k - 1    #Ending index of found repeat i
            one_lsj = SJ_li                 #Starting index of found repeat j
            one_lej = SJ_li + k - 1         #Ending index of found repeat j
            one_lk = np.ones((l_num, 1))*k  #Length of found pair of repeats, i and j 
            l_add = np.concatenate((one_lsi, one_lei, one_lsj, one_lej, one_lk), axis = None)
            #print("l_add:", l_add)
            
            # Found pair of repeats on the right side 
            two_lsi = L[lnds, 0] + k        #Starting index of found repeat i 
            two_lei = L[lnds, 1]            #Ending index of ofund repeat i
            two_lsj = SJ_li + k             #Starting index of found repeat j 
            two_lej = L[lnds, 3]            #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats, i and j 
            l_add_right = np.concatenate((two_lsi, two_lei, two_lsj, two_lej, two_lk), axis = None)
            #print("l_add_right:", l_add_right)
            
            # Stack the found rows vertically 
            add_rows = np.vstack((l_add, l_add_right))
            
            # Stack all the rows found on the left side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0)
            #print("add_rows:", add_rows)
            
            #print()
    # Right Check: check for CI on the right side of the pairs 
        # Check if the the starting index of repeat "J" of the pair "IJ" equals CI
        rnds = (SJ == ci) 
        #print("rnds:", rnds)

        if rnds.sum(axis = 0) > 0:
            SJ_ri = L[rnds, 0]
            r_num = SJ_ri.shape[0] 
          
            # Found pair of repeats on the left side 
            one_rsi = SJ_ri                 #Starting index of found repeat i 
            one_rei = SJ_ri + k - 1         #Ending index of found repeat i 
            one_rsj = L[rnds, 2]            #Starting index of found repeat j
            one_rej = L[rnds, 2] + k - 1    #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1))  #Length of found pair or repeats, i and j 
            r_add = np.concatenate((one_rsi, one_rei, one_rsj, one_rej, one_rk), axis = None)
            
            # Found pairs on the right side 
            two_rsi = SJ_ri + k             #Starting index of found repeat i  
            two_rei = L[rnds, 1]            #Ending index of found repeat i 
            two_rsj = L[rnds, 2] + k        #Starting index of found repeat j
            two_rej = L[rnds,3]             #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats, i and j 
            r_add_right = np.concatenate((two_rsi, two_rei, two_rsj, two_rej, two_rk), axis = None) 
            
            # Stack the found rows vertically 
            add_rows = np.vstack((r_add, r_add_right))
            
            # Stack all the rows found on the right side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)
            #print(add_rows)
            
    return add_rows 


def find_complete_list(pair_list,song_length):
    """
    Finds all smaller diagonals (and the associated pairs of repeats) 
        that are contained in larger diagonals found previously.
        
    Args
    ----
    pair_lst: np.array
        list of pairs of repeats found in earlier step
        (bandwidths MUST be in ascending order). If you have
        run find_initial_repeats before this script,
        then pair_lst will be ordered correctly. 
           
    song_length: int
        song length, which is the number of audio shingles.
   
    Returns
    -------  
    lst_out: np.array 
        list of pairs of repeats with smaller repeats added
    """
    # Find the list of unique repeat lengths
    bw_found = np.unique(pair_list[:,4])
    bw_num = np.size(bw_found, axis=0)
    
    # If the longest bandwidth is the length of the song, then remove that row
    if song_length == bw_found[bw_num-1]: 
        pair_list[-1,:] = []
        bw_found[-1] = []
        bw_num = (bw_num - 1)
        
    # Initalize temp variables
    p = np.size(pair_list,axis=0)
    add_mat = []

    # Step 1: For each found bandwidth, search upwards (i.e. search the larger 
    #        bandwidths) and add all found diagonals to the variable add_mat        
    for j in range (1,bw_num+1):
        band_width = bw_found[j-1]
        
        # Isolate pairs of repeats that are length bandwidth
        bsnds = np.amin((pair_list[:,4] == band_width).nonzero()) # Return the minimum of the array
        bends = (pair_list[:,4] > band_width).nonzero()
    
        # Convert bends into an array
        bend = np.array(bends)
    
        if bend.size > 0:
            bend = np.amin(bend)
        else:
            bend = p
    
        # Part A1: Isolate all starting time steps of the repeats of length bandwidth
        SI = pair_list[bsnds:bend, 0]
        SJ = pair_list[bsnds:bend, 2]
        all_vec_snds = np.concatenate((SI, SJ))
        int_snds = np.unique(all_vec_snds)
    
        # Part A2: Isolate all ending time steps of the repeats of length bandwidth
        EI = pair_list[bsnds:bend, 1] # Similar to definition for SI
        EJ = pair_list[bsnds:bend, 3] # Similar to definition for SJ
        all_vec_ends = np.concatenate((EI,EJ))
        int_ends = np.unique(all_vec_ends)
    
        # Part B: Use the current diagonal information to search for diagonals 
        #       of length BW contained in larger diagonals and thus were not
        #       detected because they were contained in larger diagonals that
        #       were removed by our method of eliminating diagonals in
        #       descending order by size
        add_srows = find_add_srows_both_check_no_anno(pair_lst, int_snds, band_width)
        add_erows = find_add_mrows_both_check_no_anno(pair_lst, int_snds, band_width)
        add_mrows = find_add_erows_both_check_no_anno(pair_lst, int_ends, band_width)
        
        # Check if any of the arrays are empty, if so, reshape them
        if add_mrows.size == 0:
            column = add_srows.shape[1]
            add_mrows = np.array([],dtype=np.int64).reshape(0,column) 
        elif add_srows.size == 0:
            column = add_erows.shape[1]
            add_mrows = np.array([],dtype=np.int64).reshape(0,column) 
        elif add_erows.size == 0:
            column = add_srows.shape[1]
            add_mrows = np.array([],dtype=np.int64).reshape(0,column) 
       
        # Add the new pairs of repeats to the temporary list add_mat
        add_mat.extend((add_srows,add_erows,add_mrows))
        add = np.concatenate(add_mat)
      
    # Step 2: Combine pair_lst and add_mat. Make sure that you don't have any
    #         double rows in add_mat. Then find the new list of found 
    #         bandwidths in combine_mat.
    combo = [pair_list,add]
    combine_mat = np.concatenate(combo)

    combine_mat = np.unique(combine_mat,axis=0)
    combine_inds = np.argsort(combine_mat[:,4]) # Return the indices that would sort combine_mat's fourth column
    combine_mat = combine_mat[combine_inds,:]
    c = np.size(combine_mat,axis=0)
    
    # Again, find the list of unique repeat lengths
    new_bw_found = np.unique(combine_mat[:,4])
    new_bw_num = np.size(new_bfound,axis=0)
    full_lst = []
    
    # Step 3: Loop over the new list of found bandwidths to add the annotation
    #         markers to each found pair of repeats
    for j in range(1, new_bw_num+1):
        new_band_width = new_bw_found[j-1]
        # Isolate pairs of repeats in combine_mat that are length bandwidth
        new_bsnds = np.amin((combine_mat[:,4] == new_band_width).nonzero()) # Return the minimum of the array
        new_bends = (combine_mat[:,4] > new_band_width).nonzero() 

        # Convert new_bends into an array
        new_bend = np.array(new_bends)
    
        if new_bend.size > 0:
            new_bend = np.amin(new_bend)
        else:
            new_bend = c
        
        band_width_mat = np.array((combine_mat[new_bsnds:new_bend,]))
        length_band_width_mat = np.size(band_width_mat,axis=0)

        temp_anno_lst = np.concatenate((band_width_mat,(np.zeros((length_band_width_mat,1)))),axis=1).astype(int)

        # Part C: Get annotation markers for this bandwidth
        temp_anno_lst = add_annotations(temp_anno_lst,song_length)
        full_lst.append(temp_anno_lst)
        full = np.vstack(full_lst)
    
    lst_out = full
    
    return lst_out


def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    """

    Finds all diagonals present in THRESH_MAT, removing each diagonal 
    as it is found.
    
    Args
    ----
    thresh_mat: np.array(int)
        Thresholded matrix that we extract diagonals from
        
    bandwidth_vec: np.array(1D, int)
        Vector of lengths of diagonals to be found
        
    thresh_bw: int
        Smallest allowed diagonal length
        
    Returns
    -------
    all_lst: np.array(int)
        list of pairs of repeats that correspond to diagonals in thresh_mat
    """

    b = np.size(bandwidth_vec)

    #create empty lists to store arrays
    int_all =  []
    sint_all = []
    eint_all = []
    mint_all = []

    #loop over all bandwidths
    for bw in bandwidth_vec:
        if bw > thresh_bw:
        #search for diagonals of length BW
            thresh_mat_size = np.size(thresh_mat)
            DDM_rename = signal.convolve2d(thresh_mat[0:thresh_mat_size, 0:thresh_mat_size],np.eye(bw),'valid')
            #mark where diagonals of length BW start
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


def find_song_pattern(thresh_diags):
    """
    Stitches information from thresholded diagonal matrix into a single
        row

    Args
    ----
    thresh_diags: array
        binary matrix with 1 at each pair (SI,SJ) and 0 elsewhere. 
        WARNING: must be symmetric
    
    Returns
    -------
    song_pattern: array
        row where each entry represents a time step and the group 
        that time step is a member of
    """
    song_length = thresh_diags.shape[0]
    
    # Initialize song pattern base
    pattern_base = np.zeros((1,song_length), dtype = int)

    # Initialize group number
    pattern_num = 1
    
    col_sum = thresh_diags.sum(axis = 0)

    check_inds = col_sum.nonzero()
    check_inds = check_inds[0]
    
    # Creates vector of song length
    pattern_mask = np.ones((1, song_length))
    pattern_out = (col_sum == 0)
    pattern_mask = pattern_mask - pattern_out
    
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
                c_mat = np.sum(thresh_diags[inds,:], axis = 0)
                c_mat = c_mat*pattern_mask
                
                # Finds nonzero entries of c_mat
                c_inds = c_mat.nonzero()
                c_inds = c_inds[1]
                
                # Gives all elements of c_inds the same grouping 
                # number as i
                pattern_base[0,c_inds] = pattern_num
                
                # Removes all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                pattern_mask[0,c_inds] = 0
                
                # Resets inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                inds = np.delete(inds,0)
                
            # Updates grouping number to prepare for next group
            pattern_num = pattern_num + 1
            
        # Removes i from check_inds
        check_inds = np.setdiff1d(check_inds, i)
        
    song_pattern = pattern_base
    
    return song_pattern


def hierarchical_structure(matrix_no,key_no,sn):
    """
    Distills the repeats encoded in matrix_no (and key_no) to the 
        essential structure components and then builds the hierarchical 
        representation
        
    Args 
    ----
    matrix_NO: np.array(int)
        binary matrix with 1's where repeats begin and 0's otherwise
        
    key_NO: np.array(int)
        vector containing the lengths of the repeats encoded in matrix_no
        
    sn: int
        song length, which is the number of audio shingles
        
    Returns 
    -------
    full_visualization: np.array(int) 
        binary matrix representation for full_matrix_no with blocks of 
        1's equal to the length's prescribed in full_key
            
    full_key: np.array(int)
        vector containing the lengths of the hierarchical structure 
        encoded in full_matrix_no
            
    full_matrix_no: np.array(int)
        binary matrix with 1's where hierarchical structure begins 
        and 0's otherwise
            
    full_anno_lst: np.array[int]
        vector containing the annotation markers of the hierarchical 
        structure encoded in each row of full_matrix_no
    """
    breakup_tuple = breakup_overlaps_by_intersect(matrix_no, key_no, 0)
    PNO = breakup_tuple[0]
    PNO_key = breakup_tuple[1]
    
    # Using PNO and PNO_KEY, we build a vector that tells us the order of the
    # repeats of the essential structure components.
    # Get the block representation for PNO, called PNO_BLOCK
    PNO_block = reconstruct_full_block(PNO, PNO_key)


    # Assign a unique (nonzero) number for each row in PNO. We refer these unique numbers
    # COLORS. 
    num_colors = PNO.shape[0]
    num_timesteps = PNO.shape[1]
    
    # Create unique color identifier for num_colors
    color_lst = np.arange(1, num_colors+1)
    
    # Turn it into a column
    color_lst = color_lst.reshape(np.size(color_lst),1)
    color_mat = np.tile(color_lst, (1, num_timesteps))

    # For each time step in row i that equals 1, change the value at that time
    # step to i
    PNO_color = color_mat * PNO
    PNO_color_vec = PNO_color.sum(axis=0)
    
    # Find where repeats exist in time, paying special attention to the starts
    # and ends of each repeat of an essential structure component
    #take sums down columns --- conv to logical
    PNO_block_vec = ( np.sum(PNO_block, axis = 0) ) > 0
    PNO_block_vec = PNO_block_vec.astype(np.float32)

    one_vec = (PNO_block_vec[0:sn-1] - PNO_block_vec[1:sn])
    
    #  Find all the blocks of consecutive time steps that are not contained in
    #  any of the essential structure components. We call these blocks zero
    #  blocks. 
    #  Shift PNO_BLOCK_VEC so that the zero blocks are marked at the correct
    #  time steps with 1's
    if PNO_block_vec[0] == 0 :
        one_vec = np.insert(one_vec, 1, 1)
    elif PNO_block_vec[0] == 1:
        one_vec = np.insert(one_vec, 1, 0)
        
    # Assign one new unique number to all the zero blocks
    PNO_color_vec[one_vec == 1] = (num_colors + 1)
    
    #  We are only concerned with the order that repeats of the essential
    #  structure components occur in. So we create a vector that only contains
    #  the starting indices for each repeat of the essential structure
    #  components.
    #  We isolate the starting index of each repeat of the essential structure
    #  components and save a binary vector with 1 at a time step if a repeat of
    #  any essential structure component occurs there
    #     non_zero_inds = PNO_color_vec > 0
    num_NZI = non_zero_inds.sum(axis=0)

    PNO_color_inds_only = PNO_color_vec[non_zero_inds-1]
    
    # For indices that signals the start of a zero block, turn those indices
    # back to 0
    zero_inds_short = (PNO_color_inds_only == (num_colors + 1))
    PNO_color_inds_only[zero_inds_short-1] = 0

    # Create a binary matrix SYMM_PNO_INDS_ONLY such that the (i,j) entry is 1
    # if the following three conditions are true: 
    #     1) a repeat of an essential structure component is the i-th thing in
    #        the ordering
    #     2) a repeat of an essential structure component is the j-th thing in 
    #        the ordering 
    #     3) the repeat occurring in the i-th place of the ordering and the one
    #        occuring in the j-th place of the ordering are repeats of the same
    #        essential structure component. 
    # If any of the above conditions are not true, then the (i,j) entry of
    # SYMM_PNO_INDS_ONLY is 0.

    # Turn our pattern row into a square matrix by stacking that row the
    # number of times equal to the columns in that row    
    PNO_IO_mat = np.tile(PNO_color_inds_only,(num_NZI, 1))
    PNO_IO_mat = PNO_IO_mat.astype(np.float32)

    PNO_IO_mask = ((PNO_IO_mat > 0).astype(np.float32) + (PNO_IO_mat.transpose() > 0).astype(np.float32)) == 2
    symm_PNO_inds_only = (PNO_IO_mat.astype(np.float32) == PNO_IO_mat.transpose().astype(np.float32))*PNO_IO_mask

    #  Extract all the diagonals in SYMM_PNO_INDS_ONLY and get pairs of repeated
    #  sublists in the order that repeats of essential structure components.
    #  These pairs of repeated sublists are the basis of our hierarchical
    #  representation.
    NZI_lst = lightup_lst_with_thresh_bw_no_remove(symm_PNO_inds_only, [0:num_NZI])                 
    remove_inds = (NZI_lst[:,0] == NZI_lst[:,2])
    
    #  Remove any pairs of repeats that are two copies of the same repeat (i.e.
    #  a pair (A,B) where A == B)
    if np.any(remove_inds == True):
        remove_inds = np.array(remove_inds).astype(int)
        remove = np.where(remove_inds == 1)
        NZI_lst = np.delete(NZI_lst,remove,axis=0)
        
    #Add the annotation markers to the pairs in NZI_LST
    NZI_lst_anno = find_complete_list_anno_only(NZI_lst, num_NZI)


    output_tuple = remove_overlaps(NZI_lst_anno, num_NZI)
    (NZI_matrix_no,NZI_key_no) = output_tuple[1:3]
                          
    NZI_pattern_block = reconstruct_full_block(NZI_matrix_no, NZI_key_no)

    nzi_rows = NZI_pattern_block.shape[0]
    
    #Find where all blocks start and end
    pattern_starts = np.nonzero(non_zero_inds)[0]

    pattern_ends = np.array([pattern_starts[1: ] - 1]) 
    pattern_ends = np.insert(pattern_ends,np.shape(pattern_ends)[1], sn-1)
    pattern_lengths = np.array(pattern_ends - pattern_starts+1) # is this suppose to be 0 instead of -1?

    full_visualization = np.zeros((nzi_rows, sn))
    full_matrix_no = np.zeros((nzi_rows, sn))       

    
    for i in range(0,num_NZI):
        repeated_sect = NZI_pattern_block[:,i].reshape(np.shape(NZI_pattern_block)[0],1)
        full_visualization[:,pattern_starts[i]:pattern_ends[i]+1] = np.tile(repeated_sect,(1,pattern_lengths[i]))
        full_matrix_no[:,pattern_starts[i]] = NZI_matrix_no[:,i]
        
    # Get FULL_KEY, the matching bandwidth key for FULL_MATRIX_NO
    full_key = np.zeros((nzi_rows,1))
    find_key_mat = full_visualization + full_matrix_no
    
    for i in range(0,nzi_rows):
        one_start = np.where(find_key_mat[i,:] == 2)[0][0]
        temp_row = find_key_mat[i,:]
        temp_row[0:one_start+1] = 1
        find_zero = np.where(temp_row == 0)[0][0]

        if np.size(find_zero) == 0:
            find_zero = sn

        find_two = np.where(temp_row == 2)[0][0]
        if np.size(find_two) == 0:
            find_two = sn

        one_end = np.minimum(find_zero,find_two);
        full_key[i] = one_end - one_start;
      
    full_key_inds = np.argsort(full_key, axis = 0)
    
    #switch to row
    full_key_inds = full_key_inds[:,0]
    full_key = np.sort(full_key, axis = 0)
    full_visualization = full_visualization[full_key_inds,:]
    full_matrix_no = full_matrix_no[full_key_inds,:]
                        
    #Remove rows of our hierarchical representation that contain only one
    # repeat        
    inds_remove = np.where(np.sum(full_matrix_no,1) <= 1)
    inds_remove = np.array([1])
    full_key = np.delete(full_key, inds_remove, axis = 0)

    full_matrix_no = np.delete(full_matrix_no, inds_remove, axis = 0)
    full_visualization = np.delete(full_visualization, inds_remove, axis = 0)

    full_anno_lst = get_annotation_lst(full_key)
                          
    output = (full_visualization,full_key,full_matrix_no,full_anno_lst)
    
    return output


def inds_to_rows(input_inds_mat, row_length):
    """
    Converts a list of indices to row(s) with 1's where an 
        index occurs and 0's otherwise
    
    Args
    ----
    input_inds_mat: np.array 
        matrix of one or two rows, containing the starting indices 
            
    row_length: int 
        length of the rows 
            
    Returns
    -------
    new_mat: np.array 
        matrix of one or two rows, with 1's where the starting indices 
        and 0's otherwise 
    """
    mat_rows = input_inds_mat.shape[0]
    new_mat = np.zeros((mat_rows,row_length))

    for i in range(0, mat_rows + 1):
        inds = input_inds_mat[i,:]
        new_mat[i,inds] = 1;

    return new_mat


def lightup_lst_with_thresh_band_width_no_remove(thresh_mat,band_width_vec):
    """
    Finds all the diagonals present in thresh_mat.
    
    Args
    ----
    thresh_mat: np.array
        thresholded matrix that we extract diagonals from
    
    band_width_vec: np.array
        vector of lengths of diagonals to be found
    
    Returns
    -------
    all_lst: np.array
        list of pairs of repeats that correspond to diagonals
        in thresh_mat
    """
    # Initialize the input and temporary variables
    thresh_temp = thresh_mat
    band_width_vec = band_width_vec
    b = np.size(band_width_vec,axis=0)
    
    int_all = []  # Interval list for non-overlapping pairs
    sint_all = [] # Interval list for the left side of the overlapping pairs
    eint_all = [] # Interval list for the right side of the overlapping pairs
    mint_all = [] # Interval list for the middle of the overlapping pairs if they exist
    
    for i in range(1,b+1): # Loop over all possible band_widths
        # Set current band_width
        j = b-i+1
        band_width = band_width_vec[j-1]

        # Search for diagonals of length band_width
        DDM = signal.convolve2d(thresh_temp[0:,0:],np.eye(band_width),'valid').astype(int)

        # Mark where diagonals of length band_width start
        thresh_DDM = (DDM == band_width)

        if thresh_DDM.sum() > 0:
            full_band_width = band_width
            # 1) Non-Overlaps: Search outside the overlapping shingles
            # Find the starts that are paired together
            find_starts = np.nonzero(np.triu(thresh_DDM,full_band_width))
            start_I = np.array(find_starts[0])
            start_J = np.array(find_starts[1])
            num_nonoverlaps = start_I.shape[0]

            # Find the matching ends EI for SI and EJ for SJ
            end_I = start_I + (full_band_width - 1)
            end_J = start_J + (full_band_width - 1)

            # List pairs of starts with their ends and the widths of the
            # non-overlapping intervals
            int_lst = np.column_stack([start_I,end_I,start_J,end_J,full_band_width * np.ones((num_nonoverlaps,1))]).astype(int)

            # Add the new non-overlapping intervals to the full list of
            # non-overlapping intervals
            int_all.append(int_lst)

            # 2) Overlaps: Search only the overlaps in shingles
            overlap_shingles = np.nonzero(np.tril(np.triu(thresh_DDM,1), (full_band_width - 1)))
            start_I_overlap = np.array(overlap_shingles[0])
            start_J_overlap = np.array(overlap_shingles[1])
            num_overlaps = start_I_overlap.shape[0]

            if num_overlaps > 0:
                # Since you are checking the overlaps you need to cut these
                # intervals into pieces: left, right, and middle. NOTE: the
                # middle interval may NOT exist

                # Vector of 1's that is the length of the number of
                # overlapping intervals. This is used a lot.
                ones_no = np.ones((num_overlaps,1)).astype(int)

                # 2a) Left Overlap
                K = start_J_overlap - start_I_overlap  # NOTE: end_J_overlap - end_I_overlap will also equal this,
                               # since the intervals that are overlapping are
                               # the same length. Therefore the "left"
                               # non-overlapping section is the same length as
                               # the "right" non-overlapping section. It does
                               # NOT follow that the "middle" section is equal
                               # to either the "left" or "right" piece. It is
                               # possible, but unlikely.
                sint_lst = np.column_stack([start_I_overlap,(start_J_overlap - ones_no),start_J_overlap,(start_J_overlap + K - ones_no),K]).astype(int)
                Is = np.argsort(K) # Return the indices that would sort K
                Is.reshape(np.size(Is), 1)
                sint_lst = sint_lst[Is,]

                # Add the new left overlapping intervals to the full list
                # of left overlapping intervals
                sint_all.append(sint_lst)

                # 2b) Right Overlap
                end_I_overlap = start_I_overlap + (full_band_width-1)
                end_J_overlap = start_J_overlap + (full_band_width-1)
                eint_lst = np.column_stack([(end_I_overlap + ones_no - K), end_I_overlap,(end_I_overlap + ones_no), end_J_overlap, K]).astype(int)
                Ie = np.argsort(K) # Return the indices that would sort K
                Ie.reshape(np.size(Ie),1)
                eint_lst = eint_lst[Ie,:]

                # Add the new right overlapping intervals to the full list of
                # right overlapping intervals
                eint_all.append(eint_lst)
                
                # 2) Middle Overlap
                mnds = (end_I_overlap - start_J_overlap - K + ones_no) > 0
                start_I_middle = start_J_overlap * (mnds)
                end_I_middle = (end_I_overlap*(mnds) - K*(mnds))
                start_J_middle = (start_J_overlap*(mnds) + K*(mnds))
                end_J_middle = end_I_overlap*(mnds)
                k_middle = (end_I_overlap*(mnds) - start_J_overlap*(mnds) - K*(mnds) + ones_no*(mnds))
            
                if mnds.sum() > 0:
                    mint_lst = np.column_stack([start_I_middle,end_I_middle,start_J_middle,end_J_middle,k_middle])
                    Im = np.argsort(k_middle)
                    Im.reshape(np.size(Im),1)
                    mint_lst = mint_lst[Im,:]

                    # Add the new middle overlapping intervals to the full list
                    # of middle overlapping intervals
                    mint_all.append(mint_lst)

        if thresh_temp.sum() == 0:
            break 
        
    out_lst = sint_all + eint_all + mint_all + int_all
    all_lst = filter(None,out_lst)

    if out_lst is not None:
        all_lst = np.vstack(out_lst)
    else:
        all_lst = np.array([])

    return all_lst


def lightup_pattern_row_bw_1(k_mat,song_length):
    """
    Turn the k_mat into marked rows with annotation markers for
        the start indices and zeroes otherwise

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
    for a in range(1, anno_max+1):
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
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + (na*kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    output = (pattern_row,k_lst_out)
    
    return output


def lightup_pattern_row_gb(k_mat,song_length,band_width):
    """
    Turn k_mat into marked rows with annotation markers for the start indices 
        and zeroes otherwise, after removing the annotations that have overlaps, 
        output k_lst_out which only contains rows that have no overlaps,
        the annotations that have overlaps get removed from k_lst_out
        gets added to overlap_lst.
    
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
        good_check = np.zeros((1,song_length)).astype(int) # Start with row of 0's
        good_check[0,start_inds-1] = 1 # Add 1 to all time steps where repeats with annotation a begin
    
        # Using reconstruct_full_block to check for overlaps
        block_check = reconstruct_full_block(good_check,bw)

        # If there are any overlaps, remove the bad annotations from both
        # the pattern_row and from the k_lst_out
        if block_check.max() > 1:
            # Remove the bad annotations from pattern_row
            pattern_row[0,start_inds-1] = 0
    
            # Remove the bad annotations from k_lst_out and add them to overlap_lst
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
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + (na * kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    # Edit the annotations in the overlap_lst so that the annotations start
    # with 1 and increase one each time
    if overlap_lst.size > 0:
        overlap_lst = np.unique(overlap_lst,axis=0)
        overlap_lst = add_annotations(overlap_lst,song_length)

    return pattern_row, k_lst_out, overlap_lst


def merge_based_on_length(full_mat,full_bandwidth,target_bandwidth):
    """
    Merges rows of full_mat that contain repeats that are the same 
        length and are repeats of the same piece of structure.
        
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
    temp_bandwidth = np.sort(full_bandwidth,axis=None) # Sort the elements of full_bandwidth
    bnds = np.argsort(full_bandwidth,axis=None) # Return the indices that would sort full_bandwidth
    temp_mat = full_mat[bnds,:] 
    
    target_bandwidth = np.unique(target_bandwidth) # Find the unique elements of target_bandwidth
    T = target_bandwidth.shape[0] # Number of columns 
    
    for i in range(1,T+1):
        test_bandwidth = target_bandwidth[i-1]
        inds = (temp_bandwidth == test_bandwidth) # Check if temp_bandwidth is equal to test_bandwidth
        
        # If the sum of all inds elements is greater than 1, then execute this if statement
        if inds.sum() > 1:
            # Isolate rows that correspond to test_bandwidth and merge them
            toBmerged = temp_mat[inds,:]
            merged_mat = merge_rows(toBmerged, test_bandwidth)
        
            bandwidth_add_size = merged_mat.shape[0] # Number of columns
            bandwidth_add = test_bandwidth * np.ones((bandwidth_add_size,1)).astype(int)
         
            if np.any(inds == True):
                # Convert the boolean array inds into an array of integers
                inds = np.array(inds).astype(int)
                remove_inds = np.where(inds == 1)
                # Delete the rows that meet the condition set by remove_inds
                temp_mat = np.delete(temp_mat,remove_inds,axis=0)
                temp_bandwidth = np.delete(temp_bandwidth,remove_inds,axis=0)
    
            # Combine rows into a single matrix
            bind_rows = [temp_mat,merged_mat]
            temp_mat = np.concatenate(bind_rows)

            if temp_bandwidth.size == 0: # Indicates temp_bandwidth is an empty array
                temp_bandwidth = np.concatenate(bandwidth_add)
            elif temp_bandwidth.size > 0: # Indicates temp_bandwidth is not an empty array
                bind_bw = [temp_bandwidth,bandwidth_add]
                temp_bandwidth = np.concatenate(bind_bw)

            temp_bandwidth = np.sort(temp_bandwidth) # Sort the elements of temp_bandwidth
            bnds = np.argsort(temp_bandwidth) # Return the indices that would sort temp_bandwidth
            temp_mat = temp_mat[bnds,]

    out_mat = temp_mat
    out_length_vec = temp_bandwidth
    
    output = (out_mat,out_length_vec)
    
    return output


def merge_rows(input_mat, input_width):
    """
    Merges rows with at least one common repeat from the same repeated structure
    
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
    # Step 0: initialize temporary variables
    not_merge = input_mat    # Everything must be checked
    merge_mat = []           # Nothing has been merged yet
    merge_key = []
    rows = input_mat.shape[0]  # How many rows to merge?
    
    # Step 1: has every row been checked?
    while rows > 0:
        # Step 2: start merge process
        # Step 2a: choose first unmerged row
        row2check = not_merge[0,:]
        r2c_mat = np.kron(np.ones((rows,1)), row2check) # Create a comparison matrix
                                                        # with copies of row2check stacked
                                                        # so that r2c_mat is the same
                                                        # size as the set of rows waiting
                                                        # to be merged
        
        # Step 2b: find indices of unmerged overlapping rows
        merge_inds = np.sum(((r2c_mat + not_merge) == 2), axis = 1) > 0
        
        # Step 2c: union rows with starting indices in common with row2check and
        # remove those rows from input_mat
        union_merge = np.sum(not_merge[merge_inds,:], axis = 0) > 0
        np.delete(not_merge, not_merge[merge_inds,:])
          
        # Step 2d: check that newly merged rows do not cause overlaps within row
        # If there are conflicts, rerun compare_and_cut
        merge_block = reconstruct_full_block(union_merge, input_width)
        if np.max(merge_block) > 1:
            (union_merge, union_merge_key) = compare_and_cut(union_merge, input_width,
            union_merge, input_width)
        else:
            union_merge_key = input_width
        
        # Step 2e: add unions to merge_mat and merge_key
        merge_mat = np.array([[merge_mat], [union_merge]])
        merge_key = np.array([[merge_key], [union_merge_key]])
        
        # Step 3: reinitialize rs for stopping condition
        rows = not_merge.shape[0]
    
    return merge_mat

def num_of_parts(input_vec, input_start, input_all_starts):
    """
    Determines the number of blocks of consecutive time steps
        in a given list of time steps 
    
    Args
    ----
    input_vec: np.array 
        one or two parts to replicate 
            
    input_start: np.array  
        starting index for part to be replicated 
        
    input_all_starts: np.array
        starting indices for replication 
    
    Returns
    -------
    start_mat: np.array 
        matrix of one or two rows, containing the starting indices 
            
    length_vec: np.array 
        column vector of the lengths 
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
    
        add_vec = start_vec - input_start
        start_mat = np.concatenate((input_all_starts + add_vec[0]), (input_all_starts + add_vec[1]))

    length_vec = end_vec - start_vec + 2
        
    output = (start_mat, length_vec)
    
    return output 


def reconstruct_full_block(pattern_mat, pattern_key): 
    """
    Creates a binary matrix with a block of 1's for 
        each repeat encoded in pattern_mat whose length 
        is encoded in patern_key

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
    #Find number of beats (columns) in pattern_mat
    
    #Check size of pattern_mat (in cases where there is only 1 pair of
    #repeated structures)
    if (pattern_mat.ndim == 1): 
        #Convert a 1D array into 2D array 
        #From https://stackoverflow.com/questions/3061761/numpy-array-dimensions
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
    
    #momentary commmends!!!!
            tuple_of_outputs = lightup_pattern_row_gb(bw_lst, song_length, bw)
            
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
            tuple_of_outputs = lightup_pattern_row_gb_1(bw_lst, song_length)
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
            
            tuple_of_outputs = separate_all_annotations(bw_lst_out, song_length, bw, pattern_row)
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
    output_tuple = (lst_no_overlaps, matrix_no_overlaps, key_no_overlaps, annotations_no_overlaps, all_overlap_lst)
   
    return output_tuple


def separate_anno_markers(k_mat, sn, band_width, pattern_row): 
    """
    Expands pattern_row into a matrix, so that there is one group of 
        repeats per row.
        
    Args
    ----
    k_mat: np.array
        list of pairs of repeats of length band_width with annotations 
        marked. The first two columns refer to the first repeat of the pair, 
        the second two refer to the second repeat of the pair, the fifth 
        column refers to the length of the repeats, and the sixth column 
        contains the annotation markers.
        
    sn: int
        song length, which is the number of audio shingles
        
    band_width: int 
        the length of repeats encoded in k_mat
        
    pattern_row: np.array
        row vector that marks where non-overlapping repeats occur, marking 
        the annotation markers for the start indices and 0's otherwise
        
    Returns
    -------
    pattern_mat: np.array
        matrix representation of k_mat with one row for each group of repeats
        
    pattern_key: np.array
        row vector containing the lengths of the repeats encoded in each row 
        of pattern_mat
        
    anno_id_lst: np.array 
        row vector containing the annotation markers of the repeats encoded 
        in each row of pattern_mat
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

    else: 
        #When there is one annotation  
        pattern_mat = pattern_row 
        pattern_key = band_width
        
    #Transpose anno_lst from a row vector into a column vector 
    anno_id_lst = anno_lst.reshape((1,2)).transpose()
    
    output = (pattern_mat, pattern_key, anno_id_lst)
    
    return output 


def stretch_diags(thresh_diags, band_width):
    """
    Stretches entries of matrix of thresholded diagonal onsets
        into diagonals of given length
    
    Args
    ----
    thresh_diags: np.array
        binary matrix where entries equal to 1 signal the existence 
        of a diagonal
    
    band_width: int
        length of encoded diagonals
    
    Returns
    -------
    stretch_diag_mat: np.array [boolean]
        logical matrix with diagonals of length band_width starting 
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