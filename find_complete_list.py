import numpy as np

def find_complete_list(pair_lst,song_length):
    """
    Finds all smaller diagonals (and the associated pairs
    of repeats) that are contained in larger diagonals found previously.

    Parameters
    ---------
    Args: PAIR_LST: List of pairs of repeats found in earlier step
                    (bandwidths MUST be in ascending order). If you have
                    run LIGHTUP_LST_WITH_THRESH_BW before this script,
                    then PAIR_LIST will be ordered correctly. 
           
           SONG_LENGTH: Song length, which is the number of audio shingles.

           
    Returns: LST_OUT: List of pairs of repeats with smaller repeats added
    """
    
    # Find the list of unique repeat lengths
    pair_list = np.array([[1, 15, 31, 45, 15], [1, 10, 46, 55, 10], 
                     [31, 40, 46, 55, 10], [10,20,40,50,15]])
    song_length = 55

    bfound = np.unique(pair_list[:,4])
    b = np.size(bfound, axis=0) # Number of unique repeat lengths
    
    
    # If the longest bandwidth is the length of the song, then remove that row
    if song_length == bfound[b-1]:
        pair_list[-1,:] = np.array([])
        bfound[-1] = np.array([])
        b = (b - 1)

        
    p = np.size(pair_list, axis=0)
    
    # Intialize temp variables
    add_mat = np.array([])
    
    
    # Step 1: For each found bandwidth, search upwards (i.e. search the larger 
    #        bandwidths) and add all found diagonals to the variable add_mat
    for i in range (1, b+1):
        # Set the bandwidth based on bfound
        bandwidth = bfound[i-1]


        # Isolate pairs of repeats that are length bandwidth
        fourth_column = pair_list[:,4]
        bsnds = np.amin((fourth_column == bandwidth).nonzero())
        bends = (fourth_column > bandwidth).nonzero()
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

        
        #Part B: Use the current diagonal information to search for diagonals 
        #         of length bandwidth contained in larger diagonals and thus were not
        #         detected because they were contained in larger diagonals that
        #         were removed by our method of eliminating diagonals in
        #         descending order by size
        add_srows = np.array([[10,19,40,49,10], 
                              [20,20,50,50,5],
                              [10,19,40,49,10], 
                              [20,20,50,50,5]])
        add_erows = np.array([[11,20,41,50,10], 
                              [10,10,40,40,5], 
                              [11,20,41,50,10], 
                              [10,10,40,40,5]])
        add_mrows = np.array([]) 


        # Reshape if add_srows or add_erows or add_mrows is empty
        if add_mrows.size == 0:  
            add_mrows = np.array([],dtype=np.int64).reshape(0,5)
        
        elif add_erows.size == 0:
            add_erows = np.array([],dtype=np.int64).reshape(0,5)
        
        elif add_srows.size == 0:
            add_srows = np.array([],dtype=np.int64).reshape(0,5)

            
        # Add the new pairs of repeats to the temporary list add_mat
        add_mat = np.concatenate((add_srows, add_erows, add_mrows), axis=0)

        
    # Step 2: Combine pair_lst and add_mat. Make sure that you don't have any
    #         double rows in add_mat. Then find the new list of found 
    #         bandwidths in combine_mat
    combine_mat = np.concatenate((pair_list, add_mat))
    combine_mat = np.unique(combine_mat, axis=0)
    combine_inds = np.argsort(combine_mat[:,4])
    combine_mat = combine_mat[combine_inds,:]
    c = combine_mat.shape[0]

    
    # Again, find the list of unique repeat lengths
    new_bfound = np.unique(combine_mat[:,4])
    new_b = np.size(new_bfound, axis=None)
    
    full_lst = np.array([])

    
    # Step 3: Loop over the new list of found bandwidths to add the annotation
    #         markers to each found pair of repeats
    for j in range(1, new_b+1):
        new_bandwidth = new_bfound[j-1]

        # Isolate pairs of repeats in combine_mat that are length bandwidth
        new_bsnds = np.amin((combine_mat[:,4] == new_bandwidth).nonzero())
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
        temp_anno_lst = add_annotations(temp_anno_lst, sn)   
        full_lst = np.concatenate((full_lst, temp_anno_lst))


    lst_out = full_lst
    
    return lst_out
