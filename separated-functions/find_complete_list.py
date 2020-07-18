import numpy as np 

def find_complete_list(pair_list,song_length):
    """
    Finds all smaller diagonals (and the associated pairs of repeats) 
    that are contained in pair_list, which is composed of larger 
    diagonals found in find_initial_repeats.
        
    Args
    ----
    pair_list: np.array
        list of pairs of repeats found in earlier step
        (bandwidths MUST be in ascending order). If you have
        run find_initial_repeats before this script,
        then pair_list will be ordered correctly. 
           
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
    longest_bw = bw_found[-1]
    
    # If the longest bandwidth is the length of the song, then remove 
    # that row or rows 
    
    # If the longest bandwidth is the length of the song, then remove that row or rows 
    if song_length == longest_bw: 
        #Find row or rows that needs to be removed 
        row = np.where(pair_list[:,4] == longest_bw)
        num_row = np.size(row, axis = 1)
        #If there are multiple rows that have a bw of the length of the song
        if num_row > 1:
            #Counter ensures indices will match up with the rows of the current pair_list being worked on 
            counter = 0
            for index in row[0]:
                #Finds the index of the row that needs to be removed 
                row_index = index - counter 
                #Removes row 
                pair_list = np.delete(pair_list, row_index, 0)
                #Increment counter since pair_list has been transformed 
                counter = counter + 1  
        else:
            pair_list = np.delete(pair_list, row[0], 0)
    
        #Remove longest bandwidth from list of repeat lengths 
        longest_unique_index = np.where(bw_found == longest_bw)
        bw_found = np.delete(bw_found, longest_unique_index, None)

        #Decrement number of unique repeat lengths
        bw_num = bw_num - np.size(longest_unique_index, axis = 0)
        

        #Tells you essentially how many unique bandwidths there are thus how many times you need to search for diagonals 
    for j in range(1, bw_num + 1):
        
        # Initalize temp variables
        p = np.size(pair_list, axis = 0)

        #Set band_width: traverse through each bw found in the list of unique bandwidth 
        band_width = bw_found[j - 1] 
        
        # Isolate pairs of repeats that are length bandwidth in two steps 
        # Step 1: Isolate the indices of the starting pairs 
        starting = np.where(pair_list[:,4] == band_width) 
        bsnds = starting[0][0] 
        # Step 2: Isolate the indices of the ending pairs 
        ending = np.where(pair_list[:,4] > band_width)
        
        if np.size(ending) == 0:
            bends = p
        else: 
            bends = ending[0][0] - 1 
        
        # Part A1: Isolate all starting time steps of the repeats of length bandwidth
        start_I = pair_list[bsnds:bends, 0] # 0 = first column
        start_J = pair_list[bsnds:bends, 2] # 2 = second column
        all_vec_snds = np.concatenate((start_I, start_J))
        int_snds = np.unique(all_vec_snds)
        
        # Part A2: Isolate all ending time steps of the repeats of length bandwidth
        end_I = pair_list[bsnds:bends, 1] # Similar to definition for SI
        end_J = pair_list[bsnds:bends, 3] # Similar to definition for SJ
        all_vec_ends = np.concatenate((end_I,end_J))
        int_ends = np.unique(all_vec_ends)
        
        # Part B: Use the current diagonal information to search for diagonals 
        #       of length BW contained in larger diagonals and thus were not
        #       detected because they were contained in larger diagonals that
        #       were removed by our method of eliminating diagonals in
        #       descending order by size
        
        add_srows = find_add_srows(pair_list, int_snds, band_width)
        add_erows = find_add_mrows(pair_list, int_snds, band_width)
        add_mrows = find_add_erows(pair_list, int_ends, band_width)
    
        #Check if add_srows is empty 
        outputs = np.array([add_srows, add_erows, add_mrows])
        
        #Assembles add_mat 
        for i in range(0, outputs.shape[0]):
            if outputs[i].all() != False: 
                col = outputs[i].shape[1]
                row = outputs[i].shape[0]
                add_mat = np.zeros((row, col))
                add_array = outputs[i]
                add_mat = np.vstack([add_mat, add_array])
            else:
                next
            
            new_mat = np.row_stack(add_mat)
        
        num_row = new_mat.shape[0] / 2
        r = int(num_row)
        
        new_mat = np.delete(new_mat, np.s_[:r], axis = 0)
    
    # Step 2: Combine pair_list and new_mat. Make sure that you don't have any
    #         double rows. Then find the new list of found bandwidths in combine_mat.
    if new_mat.size != 0:
        combo = [pair_list, new_mat]
        combine_mat = np.concatenate(combo)
        combine_mat = np.unique(combine_mat, axis=0)
    else:
        combine_mat = np.unique(pair_list, axis =0)
    
    combine_inds = np.argsort(combine_mat[:,4]) # Return the indices that would sort combine_mat's fourth column
    combine_mat = combine_mat[combine_inds,:]
    c = np.size(combine_mat,axis=0)
    
    # Again, find the list of unique repeat lengths
    new_bfound = np.unique(combine_mat[:,4])
    new_bw_num = np.size(new_bfound,axis=0)
    
    full_lst = []
    
    # Step 3: Loop over the new list of found bandwidths to add the annotation
    #         markers to each found pair of repeats
    for j in range(1, new_bw_num + 1):
        # Set band_width: traverse through each bw found in the list of unique
        # bandwidth 
        band_width = new_bfound[j - 1] 
        # Isolate pairs of repeats that are length bandwidth in two steps 
        
        # Step 1: Isolate pairs of repeats in combine_mat that are length 
        # bandwidth
        starting = np.where(combine_mat[:,4] == band_width) 
        # Select the first pair of repeats
        new_bsnds = starting[0][0] 
        
        # Step 2: Isolate the indices of the ending pairs 
        ending = np.where(combine_mat[:,4] > band_width)
        if np.size(ending) == 0:
            new_bends = c
        else: 
            new_bends = ending[0][0] - 1
            
        band_width_mat = np.array((combine_mat[new_bsnds:new_bends,]))
        length_band_width_mat = np.size(band_width_mat,axis=0)
        temp_anno_lst = np.concatenate((band_width_mat,\
                                        (np.zeros((length_band_width_mat,1)))),axis=1).astype(int)
        
        # Part C: Get annotation markers for this bandwidth
        temp_anno_lst = np.array(temp_anno_lst, ndmin=2)
        temp_anno_lst = add_annotations(temp_anno_lst, song_length)
        full_lst.append(temp_anno_lst)
        final_lst = np.vstack(full_lst)
        
    lst_out = final_lst
        
    return lst_out


    
    
    