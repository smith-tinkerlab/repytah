# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial.distance as spd
import

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







