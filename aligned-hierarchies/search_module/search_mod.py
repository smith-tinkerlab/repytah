#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: 
This find_complete_list is identical to the find_complete_list found in the
search module with the exception of add_annotations being called. Instead, 
temp_anno_lst has been hard-coded. 
"""

import numpy as np 


def find_complete_list1(pair_list,song_length):
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
    temp_anno_list1 = np.array([[1, 10, 46, 55, 10, 1], [31, 40, 46, 55, 10, 1]])
    temp_anno_list2 = np.array([[1, 15, 31, 45, 15, 1], [10, 20, 40, 50, 15, 2]])

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
        #temp_anno_lst = add_annotations(temp_anno_lst, song_length)
        #full_lst.append(temp_anno_lst)
    full_lst.append(temp_anno_list1)
    full_lst.append(temp_anno_list2)
    final_lst = np.vstack(full_lst)
        
    lst_out = final_lst
        
    return lst_out

def find_add_erows(lst_no_anno, check_inds, k):
    """
    Finds pairs of repeated structures, representated as diagonals of a 
    certain length, k, that end at the same time step as 
    previously found pairs of repeated structures of the same length. 

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

    L = lst_no_anno
    # Logical, which pairs of repeats have length greater than k?
    search_inds = (L[:,4] > k)

    #If there are no pairs of repeats that have a length greater than k
    if sum(search_inds) == 0:
        add_rows = np.full(1, False)
        return add_rows

    # Multiply ending index of all repeats "I" by search_inds
    EI = np.multiply(L[:,1], search_inds)
    # Multipy ending index of all repeats "J" by search_inds
    EJ = np.multiply(L[:,3], search_inds)

    #Loop over check_inds
    for i in range(check_inds.size): 

        ci = check_inds[i]

        # To check if the end index of the repeat "I" equals CI
        lnds = (EI == ci) 
        # To check if the end index of the right repeat of the pair equals CI
        rnds = (EJ == ci)

        #Left Check: Check for CI on the left side of the pairs
        if lnds.sum(axis = 0) > 0: #If the sum across (row) is greater than 0 
            # Find the 3rd entry of the row (lnds) whose starting index of 
            # repeat "J" equals CI
            EJ_li = L[lnds,3]
            
            # Number of rows in EJ_li 
            l_num = EJ_li.shape[0] 
            
            
            # Found pair of repeats on the left side
            one_lsi = L[lnds,1] - k + 1     #Starting index of found repeat i
            one_lei = L[lnds,1]             #Ending index of found repeat i
            one_lsj = EJ_li - k + 1         #Starting index of found repeat j
            one_lej = EJ_li                 #Ending index of found repeat j
            one_lk = k*np.ones((l_num,1))   #Length of found pair of repeats
            l_add = np.concatenate((one_lsi, one_lei, one_lsj, one_lej, one_lk), axis = None)

            
            # Found pair of repeats on the right side
            two_lsi = L[lnds,0]             #Starting index of found repeat i 
            two_lei = L[lnds,1] - k         #Ending index of ofund repeat i
            two_lsj = L[lnds,2]             #Starting index of found repeat j 
            two_lej = EJ_li - k             #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats 
            l_add_left = np.concatenate((two_lsi, two_lei, two_lsj, two_lej, two_lk), axis = None)
            
            # Stack the found rows vertically 
            add_rows = np.vstack((l_add, l_add_left))
            
            # Stack all the rows found on the left side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0)
            
    
        #Right Check: Check for CI on the right side of the pairs
        elif rnds.sum(axis = 0) > 0:
            # Find the 1st entry of the row whose ending index of repeat 
            # "I" equals CI
            EI_ri = L[rnds, 1]
            # Number of rows in EJ_ri                    
            r_num = EI_ri.shape[0]
                               
            # Found pair of repeats on the left side 
            one_rsi = EI_ri - k + 1         #Starting index of found repeat i 
            one_rei = EI_ri                 #Ending index of found repeat i 
            one_rsj = L[rnds, 3] - k + 1    #Starting index of found repeat j
            one_rej = L[rnds,3]             #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1))  #Length of found pair or repeats
            r_add = np.concatenate((one_rsi, one_rei, one_rsj, one_rej, one_rk), axis = None)
            
            # Found pairs on the right side 
            two_rsi = L[rnds, 0]            #Starting index of found repeat i  
            two_rei = EI_ri - k             #Ending index of found repeat i 
            two_rsj = L[rnds, 2]            #Starting index of found repeat j
            two_rej = L[rnds, 3] - k        #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats
            r_add_right = np.concatenate((two_rsi, two_rei, two_rsj, two_rej, two_rk), axis = None) 
            
            # Stack the found rows vertically 
            add_rows = np.vstack((r_add, r_add_right))
            
            # Stack all the rows found on the right side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)
            
            if add_rows.any() == None: 
                add_rows = np.full(1, False)
                return add_rows 
            else: 
                return add_rows
                
        #If there are no found pairs 
        else:
            add_rows = np.full(1, False)
            return add_rows 
            


def find_add_mrows(lst_no_anno, check_inds, k): 
    """
    Finds pairs of repeated structures, represented as diagonals of a certain
    length, k, that neither start nor end at the same time steps as previously
    found pairs of repeated structures of the same length. 

    Args
    ----
        lst_no_anno: np.array 
            list of pairs of repeats

        check_inds: np.array
            list of ending indices for repeats of length k that we use to 
            check lst_no_anno for more repeats of length k 

        k: number
            length of repeats that we are looking for 

    Returns
    -------
        add_rows: np.array
            list of newly found pairs of repeats of length K that are 
            contained in larger repeats in LST_NO_ANNO 
    """
    #Initialize list of pairs 
    L = lst_no_anno 
    
    #Logical, which pair of repeats has a length greater than k 
    search_inds = (L[:,4] > k)
    
    #If there are no pairs of repeats that have a length greater than k 
    if sum(search_inds) == 0:
        add_rows = np.full(1, False) 
        return add_rows
    
    #Multiply the starting index of all repeats "I" by search_inds
    SI = np.multiply(L[:,0], search_inds)

    #Multiply the starting index of all repeats "J" by search_inds
    SJ = np.multiply(L[:,2], search_inds)

    #Multiply the ending index of all repeats "I" by search_inds
    EI = np.multiply(L[:,1], search_inds)

    #Multiply the ending index of all repeats "J" by search_inds
    EJ = np.multiply(L[:,3], search_inds)
    
    #Loop over check_inds 
    for i in range(check_inds.size): 
        ci = check_inds[i]
        #For Left  Check: check for CI on the left side of the pairs
        lnds = ((SI < ci) + (EI > (ci + k -1)) == 2)
        
        #For Right Check: check for CI on the right side of the pairs
        rnds = ((SJ < ci) + (EJ > (ci + k - 1)) == 2);
        
        #Check that SI < CI and that EI > (CI + K - 1) indicating that there
        #is a repeat of length k with starting index CI contained in a larger
        #repeat which is the left repeat of a pair
        if lnds.sum(axis = 0) > 0:
            #Find the 2nd entry of the row (lnds) whose starting index of the
            #repeat "I" equals CI 
            SJ_li = L[lnds,2]
            EJ_li = L[lnds,3]
            l_num = SJ_li.shape[0]

            #Left side of left pair
            l_left_k = ci*np.ones(l_num,1) - L[lnds,0]
            l_add_left = np.concatenate((L[lnds,0], (ci - 1 * np.ones((l_num,1))), SJ_li, (SJ_li + l_left_k - np.ones((l_num,1))), l_left_k), axis = None)

            #Middle of left pair
            l_add_mid = np.concatenate(((ci*np.ones((l_num,1))), (ci+k-1)*np.ones((l_num,1)), SJ_li + l_left_k, SJ_li + l_left_k + (k-1)*np.ones((l_num,1)), k*np.ones((l_num,1))), axis = None) 

            #Right side of left pair
            l_right_k = np.concatenate((L[lnds, 1] - ((ci + k) - 1) * \
                                        np.ones((l_num,1))), axis = None)
            l_add_right = np.concatenate((((ci + k)*np.ones((l_num,1))), \
                                          L[lnds,1], (EJ_li - l_right_k + \
                                           np.ones((l_num,1))), EJ_li, \
                                           l_right_k), axis = None)

            # Add the found rows        
            add_rows = np.vstack((l_add_left, l_add_mid, l_add_right))
            #add_rows = np.reshape(add_rows, (3,5))
 

        #Check that SI < CI and that EI > (CI + K - 1) indicating that there
        #is a repeat of length K with starting index CI contained in a larger
        #repeat which is the right repeat of a pair
        elif rnds.sum(axis = 0) > 0:
            SI_ri = L[rnds,0]
            EI_ri = L[rnds,1]
            r_num = SI_ri.shape[0]

            #Left side of right pair
            r_left_k = ci*np.ones((r_num,1)) - L[rnds,2]
            r_add_left = np.concatenate((SI_ri, (SI_ri + r_left_k - \
                                                 np.ones((r_num,1))), \
                                              L[rnds,3], (ci - 1) * \
                                              np.ones((r_num,1)), r_left_k), \
                                              axis = None)

            #Middle of right pair
            r_add_mid = np.concatenate(((SI_ri + r_left_k),(SI_ri + r_left_k \
                                        + (k - 1)*np.ones((r_num,1))), \
                                        ci*np.ones((r_num,1)), \
                                        (ci + k - 1)*np.ones((r_num,1)), \
                                        k*np.ones((r_num,1))), axis = None)

            #Right side of right pair
            r_right_k = L[rnds, 3] - ((ci + k) - 1)*np.ones((r_num,1))
            r_add_right = np.concatenate((EI_ri - r_right_k + \
                                          np.ones((r_num,1)),EI_ri,\
                                          (ci + k)*np.ones((r_num,1)), \
                                          L[rnds,3], r_right_k), axis = None)

            add_rows = np.vstack((r_add_left, r_add_mid, r_add_right))
            #add_rows = np.reshape(add_rows, (3,5))

            add_rows = np.concatenate((add_rows, add_rows), \
                                      axis = 0).astype(int)
            if add_rows == None: 
                add_rows = np.full(1, False)
                
                return add_rows 
            else: 
                
                return add_rows
          
        #If there are no found pairs 
        else:
            add_rows = np.full(1, False)
           
            return add_rows 

def find_add_srows(lst_no_anno, check_inds, k):
    """
    Finds pairs of repeated structures, representated as diagonals of a 
    certain length, k, that start at the same time step as previously found 
    pairs of repeated structures of the same length. 
        
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
    
    L = lst_no_anno

    # Logical, which pair of repeats has a length greater than k 
    search_inds = (L[:,4] > k)
    
    #If there are no repeats greater than k 
    if sum(search_inds) == 0: 
        add_rows = np.full(1, False) 
        
        return add_rows

    # Multipy the starting index of all repeats "I" by search_inds
    SI = np.multiply(L[:,0], search_inds)

    # Multiply the starting index of all repeats "J" by search_inds
    SJ = np.multiply(L[:,2], search_inds)

    # Loop over check_inds
    for i in range(check_inds.size):
        ci = check_inds[i] 
            
    # For Left check: check for CI on the left side of the pairs 
        # Check if the starting index of repeat "I" of pair of repeats "IJ" 
        # equals CI
        lnds = (SI == ci) 
        
    # For Right Check: check for CI on the right side of the pairs 
        # Check if the the starting index of repeat "J" of the pair "IJ" 
        # equals CI
        rnds = (SJ == ci)
        
        # If the sum across (row) is greater than 0 
        if lnds.sum(axis = 0) > 0: 
            # Find the 2nd entry of the row (lnds) whose starting index of 
            # repeat "I" equals CI 
            SJ_li = L[lnds, 2] 
            
            # Used for the length of found pair of repeats 
            l_num = SJ_li.shape[0] 

            # Found pair of repeats on the left side 
            one_lsi = L[lnds, 0]            #Starting index of found repeat i
            one_lei = L[lnds, 0] + k - 1    #Ending index of found repeat i
            one_lsj = SJ_li                 #Starting index of found repeat j
            one_lej = SJ_li + k - 1         #Ending index of found repeat j
            one_lk = np.ones((l_num, 1))*k  #Length of found pair of repeats
            l_add = np.concatenate((one_lsi, one_lei, one_lsj, one_lej,\
                                    one_lk), axis = None)
            
            # Found pair of repeats on the right side 
            two_lsi = L[lnds, 0] + k        #Starting index of found repeat i 
            two_lei = L[lnds, 1]            #Ending index of ofund repeat i
            two_lsj = SJ_li + k             #Starting index of found repeat j 
            two_lej = L[lnds, 3]            #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats
            l_add_right = np.concatenate((two_lsi, two_lei, two_lsj, two_lej,\
                                          two_lk), axis = None)
            
            # Stack the found rows vertically 
            add_rows = np.vstack((l_add, l_add_right))
            
            # Stack all the rows found on the left side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0)
            
        
        elif rnds.sum(axis = 0) > 0:
            SJ_ri = L[rnds, 0]
            r_num = SJ_ri.shape[0] 
          
            # Found pair of repeats on the left side 
            one_rsi = SJ_ri                 #Starting index of found repeat i 
            one_rei = SJ_ri + k - 1         #Ending index of found repeat i 
            one_rsj = L[rnds, 2]            #Starting index of found repeat j
            one_rej = L[rnds, 2] + k - 1    #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1))  #Length of found pair or repeats
            r_add = np.concatenate((one_rsi, one_rei, one_rsj, one_rej, \
                                    one_rk), axis = None)
            
            # Found pairs on the right side 
            two_rsi = SJ_ri + k             #Starting index of found repeat i  
            two_rei = L[rnds, 1]            #Ending index of found repeat i 
            two_rsj = L[rnds, 2] + k        #Starting index of found repeat j
            two_rej = L[rnds,3]             #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats
            r_add_right = np.concatenate((two_rsi, two_rei, two_rsj, two_rej,\
                                          two_rk), axis = None) 
            
            # Stack the found rows vertically 
            add_rows = np.vstack((r_add, r_add_right))
            
            # Stack all the rows found on the right side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows),\
                                      axis = 0).astype(int)
            
            if add_rows.any() == None: 
                add_rows = np.full(1, False)
                
                return add_rows 
            else:
                
                return add_rows
            
        #If there are no found pairs 
        else:
            add_rows = np.full(1, False)
            
            return add_rows          
 
    
    
    