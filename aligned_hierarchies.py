# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial.distance as spd
import scipy.sparse as sps
from scipy import signal

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



def create_sdm(matrix_featurevecs, num_fv_per_shingle):
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
    # Number of beats in pattern_mat (columns)
    sn = pattern_mat.shape[1]

    # Number of repeated structures in pattern_mat (rows)
    sb = pattern_mat.shape[0]

    # Pre-allocating a sn by sb array of zeros 
    pattern_block = np.zeros((sb,sn)).astype(int)  

    for i in range(sb):

        # Retrieve all of row i of pattern_mat 
        repeated_struct = pattern_mat[i,:]

        # Retrieve the length of the repeats encoded in row i of pattern_mat 
        length = pattern_key[i]

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
            sub_struct_b = np.zeros((1,( b  - 1)))

            # Append sub_struct_b in front of sub_struct_a 
            new_struct = np.append(sub_struct_b, sub_struct_a)

            # Replace part of sub_section with new_struct 
            sub_section[b - 1,:] = new_struct

        # Replaces part of pattern_block with the sums of each column in sub_section 
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