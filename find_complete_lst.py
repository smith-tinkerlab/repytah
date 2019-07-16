import numpy as np

def find_complete_list(pair_list,song_length):
    """
    Finds all smaller diagonals (and the associated pairs
        of repeats) that are contained in larger diagonals found previously.
    Args
    ----
    pair_lst: np.array
        list of pairs of repeats found in earlier step
        (bandwidths MUST be in ascending order). If you have
        run lightup_lst_with_thresh before this script,
        then pair_lst will be ordered correctly. 
           
    song_length: int
        song length, which is the number of audio shingles.
   
    Returns
    -------  
    lst_out: np.array 
        list of pairs of repeats with smaller repeats added
    """
    # Find the list of unique repeat lengths
    bfound = np.unique(pair_list[:,4])
    b = np.size(bfound, axis=0) # Number of unique repeat lengths
    
    # If the longest bandwidth is the length of the song, then remove that row
    if song_length == bfound[b-1]: 
        pair_list[-1,:] = np.array([])
        bfound[-1] = np.array([])
        b = (b - 1)
        
    # Initalize temp variables
    p = np.size(pair_list,axis=0)
    add_mat = []

    # Step 1: For each found bandwidth, search upwards (i.e. search the larger 
    #        bandwidths) and add all found diagonals to the variable add_mat        
    for j in range (1,b+1):
        bandwidth = bfound[j-1]
        
        # Isolate pairs of repeats that are length bandwidth
        bsnds = np.amin((pair_list[:,4] == bandwidth).nonzero()) # Return the minimum of the array
        bends = (pair_list[:,4] > bandwidth).nonzero()
    
        # Convert bends into an array
        bend = np.array(bends)
    
        if bend.size > 0:
            bend = np.amin(bend)
        else:
            bend = np.amin(p)
    
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
        add_srows = find_add_srows_both_check_no_anno(pair_lst, int_snds, bandwidth)
        add_erows = find_add_mrows_both_check_no_anno(pair_lst, int_snds, bandwidth)
        add_mrows = find_add_erows_both_check_no_anno(pair_lst, int_ends, bandwidth)
        
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
    new_bfound = np.unique(combine_mat[:,4])
    new_b = np.size(new_bfound,axis=0)
    full_lst = []
    
    # Step 3: Loop over the new list of found bandwidths to add the annotation
    #         markers to each found pair of repeats
    for j in range(1, new_b + 1):
        new_bandwidth = new_bfound[j-1]
        # Isolate pairs of repeats in combine_mat that are length bandwidth
        new_bsnds = np.amin((combine_mat[:,4] == new_bandwidth).nonzero()) # Return the minimum of the array
        new_bends = (combine_mat[:,4] > new_bandwidth).nonzero() 

        # Convert new_bends into an array
        new_bend = np.array(new_bends)
    
        if new_bend.size > 0:
            new_bend = np.amin(new_bend)
        else:
            new_bend = c
        
        bandwidth_mat = np.array((combine_mat[new_bsnds:new_bend,]))
        length_bandwidth_mat = np.size(bandwidth_mat,axis=0)

        temp_anno_lst = np.concatenate((bandwidth_mat,(np.zeros((length_bandwidth_mat,1)))),axis=1).astype(int)

        # Part C: Get annotation markers for this bandwidth
        temp_anno_lst = add_annotations(temp_anno_lst,song_length)
        full_lst.append(temp_anno_lst)
        full = np.vstack(full_lst)
    
    lst_out = full
    
    return lst_out
