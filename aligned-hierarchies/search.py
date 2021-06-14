#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py 

This file contains the following functions:
    
    * find_complete_list - Finds all smaller diagonals (and the associated 
    pairs of repeats) that are contained pair_list, which is composed of 
    larger diagonals found in find_initial_repeats.
    
    * __find_add_srows - Finds pairs of repeated structures, represented as 
    diagonals of a certain length, k, that start at the same time step as 
    previously found pairs of repeated structures of the same length. 
    
    * __find_add_erows - Finds pairs of repeated structures, represented as 
    diagonals of a certain length, k, that end at the same time step as 
    previously found pairs of repeated structures of the same length.

    * __find_add_mrows - Finds pairs of repeated structures, represented as 
    diagonals of a certain length, k, that neither start nor end at the same 
    time steps as previously found pairs of repeated structures of the same 
    length. 
    
    * find_all_repeats - Finds all the diagonals present in thresh_mat. This 
    function is nearly identical to find_initial_repeats, with two crucial 
    differences. First, we do not remove diagonals after we find them. Second,
    there is no smallest bandwidth size as we are looking for all diagonals.

    * find_complete_list_anno_only - Finds annotations for all pairs of 
    repeats found in find_all_repeats. This list contains all the pairs of  
    repeated structures with their start/end indices and lengths. 
    
"""

import numpy as np
from scipy import signal
from utilities import add_annotations


def find_complete_list(pair_list,song_length):
    """
    Finds all smaller diagonals (and the associated pairs of repeats) that are
    contained pair_list, which is composed of larger diagonals found in 
    find_initial_repeats.
        
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
    
    # If the longest bandwidth is the length of the song, then remove that row
    if song_length == bw_found[-1]: 
        pair_list = np.delete(pair_list,-1,0)
        bw_found = np.delete(bw_found,-1,0)
        bw_num = (bw_num - 1)
        
    # Initalize variables
    p = np.size(pair_list,axis=0)
    add_mat = np.zeros((1,5)).astype(int)

    # Step 1: For each found bandwidth, search upwards (i.e. search the larger 
    # bandwidths) and add all found diagonals to the variable add_mat        
    for j in range (0,bw_num-1):
        band_width = bw_found[j] 
        
        # Isolate pairs of repeats that are length bandwidth
        # Return the minimum of the array
        bsnds = np.amin((pair_list[:,4] == band_width).nonzero())
        bends = (pair_list[:,4] > band_width).nonzero()
        
        # Convert bends into an array
        bend = np.array(bends)
        if bend.size > 0:
            bend = np.amin(bend)
        else:
            bend = p
        
        # Part A1: Isolate all starting time steps of the repeats of length 
        #          bandwidth
        start_I = pair_list[bsnds:bend, 0]
        start_J = pair_list[bsnds:bend, 2]
        all_vec_snds = np.concatenate((start_I,start_J), axis=None)  
        int_snds = np.unique(all_vec_snds)

        # Part A2: Isolate all ending time steps of the repeats of length 
        # bandwidth
        end_I = pair_list[bsnds:bend, 1] # Similar to definition for start_I
        end_J = pair_list[bsnds:bend, 3] # Similar to definition for start_J

        all_vec_ends = np.concatenate((end_I,end_J),axis=None)
        int_ends = np.unique(all_vec_ends)
    
        # Part B: Use the current diagonal information to search for diagonals 
        #       of length BW contained in larger diagonals and thus were not
        #       detected because they were contained in larger diagonals that
        #       were removed by our method of eliminating diagonals in
        #       descending order by size
        add_srows = __find_add_srows(pair_list, int_snds, band_width)
        add_mrows = __find_add_mrows(pair_list, int_snds, band_width)
        add_erows = __find_add_erows(pair_list, int_ends, band_width)
       
        # Check if any of the arrays are empty
        # Add the new pairs of repeats to the temporary list add_mat
        if add_mrows.size != 0:  
            add_mat=np.vstack((add_mat,add_mrows))
           
        if add_srows.size != 0:  
            add_mat=np.vstack((add_mat,add_srows))
           
        if add_erows.size != 0:  
            add_mat=np.vstack((add_mat,add_erows))
           
    #Remove the empty row
    if add_mat.size!=0:
        add_mat= np.delete(add_mat,0,0)
        
    # Step 2: Combine pair_list and new_mat. Make sure that you don't have any
    #         double rows in add_mat. Then find the new list of found 
    #         bandwidths in combine_mat.
    combine_mat = np.vstack((pair_list,add_mat))
    combine_mat = np.unique(combine_mat,axis=0)
    
    # Return the indices that would sort combine_mat's fourth column
    combine_inds = np.argsort(combine_mat[:,4]) 
    combine_mat = combine_mat[combine_inds,:]
    c = np.size(combine_mat,axis=0)
    
    # Again, find the list of unique repeat lengths
    new_bw_found = np.unique(combine_mat[:,4])
    new_bw_num = np.size(new_bw_found,axis=0)
    full_lst = []
    
    # Step 3: Loop over the new list of found bandwidths to add the annotation
    #         markers to each found pair of repeats
    for j in range(1,new_bw_num+1):
        new_bw = new_bw_found[j-1]
        
        # Isolate pairs of repeats in combine_mat that are length bandwidth
        # Return the minimum of the array
        new_bsnds = np.amin((combine_mat[:,4] == new_bw).nonzero()) 
        new_bends = (combine_mat[:,4] > new_bw).nonzero() 

        # Convert new_bends into an array
        new_bend = np.array(new_bends)
    
        if new_bend.size > 0:
            new_bend = np.amin(new_bend)
        else:
            new_bend = c
        
        band_width_mat = np.array((combine_mat[new_bsnds:new_bend,]))
        length_band_width_mat = np.size(band_width_mat,axis=0)

        temp_anno_lst = np.concatenate((band_width_mat,\
                                        (np.zeros((length_band_width_mat,1))))\
                                        ,axis=1).astype(int)

        # Part C: Get annotation markers for this bandwidth
        temp_anno_lst = add_annotations(temp_anno_lst, song_length)
        full_lst.append(temp_anno_lst)
        final_lst = np.vstack(full_lst)
        tem_final_lst = np.lexsort([final_lst[:,2], final_lst[:,0], \
                                    final_lst[:,5],final_lst[:,4]])
        final_lst = final_lst[tem_final_lst,:]
    
    lst_out = final_lst
    
    return lst_out


def __find_add_srows(lst_no_anno, check_inds, k):
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
    add_rows = np.empty((0))

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
            
    # Left check: check for CI on the left side of the pairs 
        # Check if the starting index of repeat "I" of pair of repeats "IJ" 
        # equals CI
        lnds = (SI == ci) 

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
            one_lk = k*np.ones((l_num, 1)).astype(int).flatten()  #Length of found pair of repeats
            
            l_add = np.vstack((one_lsi,one_lei, one_lsj,one_lej,one_lk))
            l_add = np.transpose(l_add)
            
            # Found pair of repeats on the right side 
            two_lsi = L[lnds, 0] + k        #Starting index of found repeat i 
            two_lei = L[lnds, 1]            #Ending index of ofund repeat i
            two_lsj = SJ_li + k             #Starting index of found repeat j 
            two_lej = L[lnds, 3]            #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats
            
            l_add_right = np.concatenate((two_lsi, two_lei, two_lsj, two_lej,\
                                          two_lk), axis = None)
            l_add_right = np.vstack((two_lsi, two_lei, two_lsj, two_lej,two_lk))
            l_add_right = np.transpose(l_add_right)
    
            # Stack the found rows vertically 
            if add_rows.size == 0:
                add_rows = np.vstack((l_add, l_add_right))
            else:
                add_rows = np.vstack((add_rows, l_add, l_add_right))
            
    # Right Check: check for CI on the right side of the pairs 
        # Check if the the starting index of repeat "J" of the pair "IJ" 
        # equals CI
        rnds = (SJ == ci) 

        if rnds.sum(axis = 0) > 0:
            SJ_ri = L[rnds, 0]
            r_num = SJ_ri.shape[0] 
          
            # Found pair of repeats on the left side 
            one_rsi = SJ_ri                 #Starting index of found repeat i 
            one_rei = SJ_ri + k - 1         #Ending index of found repeat i 
            one_rsj = L[rnds, 2]            #Starting index of found repeat j
            one_rej = L[rnds, 2] + k - 1    #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1)).astype(int).flatten()  #Length of found pair or repeats
            
            r_add = np.vstack((one_rsi, one_rei, one_rsj, one_rej,one_rk)) 
            r_add = np.transpose(r_add)
                
            # Found pairs on the right side 
            two_rsi = SJ_ri + k             #Starting index of found repeat i  
            two_rei = L[rnds, 1]            #Ending index of found repeat i 
            two_rsj = L[rnds, 2] + k        #Starting index of found repeat j
            two_rej = L[rnds,3]             #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats
            r_add_right = np.vstack((two_rsi, two_rei, two_rsj, two_rej,\
                                          two_rk)) 
            r_add_right = np.transpose(r_add_right)
            
            # Stack the found rows vertically 
            if add_rows.size == 0:
                add_rows = np.vstack((r_add,r_add_right))
            else:
                add_rows = np.vstack((add_rows, r_add, r_add_right))
            
    return add_rows


def __find_add_erows(lst_no_anno, check_inds, k):
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
    add_rows = np.empty((0))
    # Logical, which pairs of repeats have length greater than k?
    search_inds = (L[:,4] > k)
    
    #If there are no pairs of repeats that have a length greater than k
    if sum(search_inds) == 0:
        add_rows = np.full(1, False)
        return add_rows

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
            one_lk = k*np.ones((1,l_num)).astype(int).flatten()         
            l_add = np.vstack((one_lsi,one_lei, one_lsj,one_lej,one_lk))
            l_add = np.transpose(l_add)
     
            # Found pair of repeats on the right side
            two_lsi = L[lnds,0]             #Starting index of found repeat i 
            two_lei = L[lnds,1] - k         #Ending index of ofund repeat i
            two_lsj = L[lnds,2]             #Starting index of found repeat j 
            two_lej = EJ_li - k             #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats            
            l_add_left = np.vstack((two_lsi,two_lei, two_lsj,two_lej,two_lk))
            l_add_left = np.transpose(l_add_left)
           
            # Stack the found rows vertically        
            if add_rows.size == 0:
                add_rows = np.vstack((l_add, l_add_left))
            else:
                add_rows = np.vstack((add_rows, l_add, l_add_left))
                
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
            one_rk = k*np.ones((1,r_num)).astype(int).flatten()  #Length of found pair or repeats 
            r_add = np.vstack((one_rsi,one_rei, one_rsj,one_rej,one_rk))
            r_add = np.transpose(r_add)
            
            # Found pairs on the right side 
            two_rsi = L[rnds, 0]            #Starting index of found repeat i  
            two_rei = EI_ri - k             #Ending index of found repeat i 
            two_rsj = L[rnds, 2]            #Starting index of found repeat j
            two_rej = L[rnds, 3] - k        #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats
            r_add_right = np.vstack((two_rsi,two_rei, two_rsj,two_rej,two_rk))
            r_add_right = np.transpose(r_add_right)
            
            # Stack the found rows vertically  
            if add_rows.size == 0:
                add_rows = np.vstack((r_add,r_add_right))
            else:
                add_rows = np.vstack((add_rows, r_add, r_add_right))
                     
    return add_rows


def __find_add_mrows(lst_no_anno, check_inds, k): 
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
    add_rows = np.empty((0))
    
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
        #Left Check: check for CI on the left side of the pairs
        lnds = ((SI < ci) & (EI > (ci + k -1)) == True)
        
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
            l_left_k = (ci*np.ones((1,l_num))) - L[lnds,0]
            l_add_left = np.vstack((L[lnds,0]*np.ones((1,l_num)),\
                                    (ci - 1 * np.ones((1,l_num))),\
                                    SJ_li*np.ones((1,l_num)),\
                                    (SJ_li + l_left_k - np.ones((1,l_num))),\
                                    l_left_k))
            l_add_left = np.transpose(l_add_left)
        
            #Middle of left pair
            l_add_mid = np.vstack(((ci*np.ones((1,l_num))), \
                                        (ci+k-1)*np.ones((1,l_num)), \
                                        SJ_li + l_left_k, SJ_li + \
                                        l_left_k + (k-1)*np.ones((1,l_num)),\
                                        k*np.ones((1,l_num))))
            l_add_mid = np.transpose(l_add_mid)
           
            #Right side of left pair
            l_right_k = np.concatenate((L[lnds, 1] - ((ci + k) - 1) * \
                                        np.ones((1,l_num))), axis = None)
            l_add_right = np.vstack((((ci + k)*np.ones((1,l_num))), \
                                          L[lnds,1], (EJ_li - l_right_k + \
                                           np.ones((1,l_num))), EJ_li, \
                                           l_right_k))
            l_add_right = np.transpose(l_add_right)
            
            # Add the found rows        
            if add_rows.size ==0:
                add_rows = np.vstack((l_add_left, l_add_mid, l_add_right)).astype(int)
            else:
                add_rows = np.vstack((add_rows, l_add_left, l_add_mid, l_add_right)).astype(int)    
                

        #Right Check: Check for CI on the right side of the pairs
        rnds = ((SJ < ci) & (EJ > (ci + k - 1)) == True)

        #Check that SI < CI and that EI > (CI + K - 1) indicating that there
        #is a repeat of length K with starting index CI contained in a larger
        #repeat which is the right repeat of a pair
        if rnds.sum(axis = 0) > 0:
            SI_ri = L[rnds,0]
            EI_ri = L[rnds,1]
            r_num = SI_ri.shape[0]

            #Left side of right pair
            r_left_k = ci*np.ones((1,r_num)) - L[rnds,2]
            r_add_left = np.vstack((SI_ri, (SI_ri + r_left_k - \
                                                 np.ones((1,r_num))), \
                                              L[rnds,2], (ci - 1) * \
                                              np.ones((1,r_num)), r_left_k))
            r_add_left = np.transpose(r_add_left)

            #Middle of right pair
            r_add_mid = np.vstack(((SI_ri + r_left_k),(SI_ri + r_left_k \
                                        + (k - 1)*np.ones((1,r_num))), \
                                        ci*np.ones((1,r_num)), \
                                        (ci + k - 1)*np.ones((1,r_num)), \
                                        k*np.ones((1,r_num))))
            r_add_mid = np.transpose(r_add_mid)    

            #Right side of right pair
            r_right_k = L[rnds, 3] - ((ci + k) - 1)*np.ones((1,r_num))
            r_add_right = np.vstack((EI_ri - r_right_k + \
                                          np.ones((1,r_num)),EI_ri,\
                                          (ci + k)*np.ones((1,r_num)), \
                                          L[rnds,3], r_right_k))
            r_add_right = np.transpose(r_add_right)

            # Add the found rows        
            if add_rows.size ==0:
                add_rows = np.vstack((r_add_left, r_add_mid, r_add_right)).astype(int)
            else:
                add_rows = np.vstack((add_rows, r_add_left, r_add_mid, r_add_right)).astype(int)  
     
    return add_rows 


def find_all_repeats(thresh_mat, bw_vec):
    """
    Finds all the diagonals present in thresh_mat. This function is 
    nearly identical to find_initial_repeats, with two crucial 
    differences. First, we do not remove diagonals after we 
    find them. Second, there is no smallest bandwidth size 
    as we are looking for all diagonals.
        
    Args
    ----
    thresh_mat: np.array
        thresholded matrix that we extract diagonals from
    
    bw_vec: np.array
        vector of lengths of diagonals to be found
        Should be 1,2,3,..., n where n = number of timesteps. 
    """

    # Initialize the input and temporary variables
    thresh_temp = thresh_mat
    
    # Interval list for non-overlapping pairs    
    int_all = np.empty((0,5), int) 
    
    # Interval list for the left side of the overlapping pairs
    sint_all = np.empty((0,5), int) 
    
    # Interval list for the right side of the overlapping pairs
    eint_all = np.empty((0,5), int) 
    
    # Interval list for the middle of the overlapping pairs if they exist
    mint_all = np.empty((0,5), int) 
    
    # Loop over all possible band_widths
    for bw in bw_vec:  
        
        #Use convolution matrix to find diagonals of length bw 
        id_mat = np.identity(bw) 
        
        # Search for diagonals of length band_width
        diagonal_mat = signal.convolve2d(thresh_temp, id_mat, 'valid')
        
        # Mark where diagonals of length band_width start
        diag_markers = (diagonal_mat == bw).astype(int)
        
        #Constructs all_lst, contains information about the found diagonals
        if sum(diag_markers).any() > 0:
            full_bw = bw
            
            # 1) Search outside the overlapping shingles
            upper_tri = np.triu(diag_markers, full_bw)
            # Search for paired starts 
            (start_i, start_j) = upper_tri.nonzero() 
            start_i = start_i + 1
            start_j = start_j + 1 
            
            # Find the matching ends for the previously found starts 
            match_i = start_i + (full_bw - 1)
            match_j = start_j + (full_bw - 1)
                
            # List pairs of starts with their ends and the widths of the
            # non-overlapping intervals
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
            shin_ovrlaps = np.nonzero((np.tril(np.triu(diag_markers, 1),
                                                  (full_bw - 1))))
            start_i_shin = np.array(shin_ovrlaps[0]+1) # row
            start_j_shin = np.array(shin_ovrlaps[1]+1) # column
            num_ovrlaps = len(start_i_shin)
            
            if num_ovrlaps > 0:
                # Since you are checking the overlaps you need to cut these
                # intervals into pieces: left, right, and middle. NOTE: the
                # middle interval may NOT exist
    
                # Vector of 1's that is the length of the number of
                # overlapping intervals. This is used a lot.
                ones_no = np.ones((num_ovrlaps)).astype(int)
                
                # 2a) Left Overlap
                K = start_j_shin - start_i_shin  # NOTE: end_J_overlap - end_I_overlap will also equal this,
                                                # possible, but unlikely.
                i_sshin = np.vstack((start_i_shin[:], (start_j_shin[:]\
                                                       - ones_no[:]))).T
                j_sshin = np.vstack((start_j_shin[:], (start_j_shin[:]\
                                                       + K - ones_no[:]))).T
                sint_lst = np.column_stack((i_sshin,j_sshin,K.T))
        
                i_s = np.argsort(K) # Return the indices that would sort K
                sint_lst = sint_lst[i_s,]
                
                # Add the new left overlapping intervals to the full list
                # of left overlapping intervals
                sint_all = np.vstack((sint_all,sint_lst))
    
                # 2b) Right Overlap
                end_i_shin = start_i_shin + (full_bw-1)
                end_j_shin = start_j_shin + (full_bw-1)
                
                i_eshin = np.vstack((end_i_shin[:] + ones_no[:] - K, end_i_shin[:])).T
                j_eshin = np.vstack((end_i_shin[:] + ones_no[:], end_j_shin[:])).T
                eint_lst = np.column_stack((i_eshin,j_eshin,K.T))
                
                i_e = np.lexsort(K) # Return the indices that would sort K
                eint_lst = eint_lst[i_e:,]
    
                # Add the new right overlapping intervals to the full list of
                # right overlapping intervals
                eint_all = np.vstack((eint_all,eint_lst))
                    
                # 2) Middle Overlap
                mnds = (end_i_shin - start_j_shin - K + ones_no) > 0
                if sum(mnds) > 0:
                    i_middle = (np.vstack((start_j_shin[:], end_i_shin[:]\
                                           - K ))) * mnds
                    i_middle = i_middle.T
                    i_middle = i_middle[np.all(i_middle != 0, axis=1)]
                    
                    j_middle = (np.vstack((start_j_shin[:] + K, \
                                           end_i_shin[:])))  * mnds 
                    j_middle = j_middle.T
                    j_middle = j_middle[np.all(j_middle != 0, axis=1)]
                    
                    k_middle = np.vstack((end_i_shin[mnds] - start_j_shin[mnds]\
                                          - K[mnds] + ones_no[mnds]))
                    k_middle = k_middle.T
                    k_middle = k_middle[np.all(k_middle != 0, axis=1)]
 
                    mint_lst = np.column_stack((i_middle,j_middle,k_middle.T))
                    mint_all = np.vstack((mint_all, mint_lst))
        
        if thresh_temp.sum() == 0:
            break 
    
    #Combine non-overlapping intervals with the left, right, and middle parts of the nonoverlapping intervals         
    out_lst = np.vstack((sint_all, eint_all, mint_all))
    
    inds = np.argsort(out_lst[:,4])
    out_lst = np.array(out_lst)[inds]
    
    inds = np.argsort(int_all[:,4])
    int_all = np.array(int_all)[inds]
    
    all_lst = np.vstack((int_all, out_lst))
    
    return((all_lst).astype(int))


def find_complete_list_anno_only(pair_list, song_length):
    """
    Finds annotations for all pairs of repeats found in find_all_repeats. This 
    list contains all the pairs of repeated structures with their start/end 
    indices and lengths.
    
    Args
    ----
    pair_list: 
        list of pairs of repeats
        WARNING: bandwidths must be in ascending order
        
    song_length: int
        number of audio shingles in song
        
    Returns
    -------
    out_lst:
        list of pairs of repeats with smaller repeats added and with
        annotation markers
    """

    # Find list of unique repeat lengths
    bw_found = np.unique(pair_list[:,4])
    bw_num = bw_found.shape[0]
    
    # Remove longest bandwidth row if it is the length of the full song
    if song_length == bw_found[-1]:
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
        
        temp_anno_mat = np.concatenate((bw_mat,(np.zeros((bw_mat_length,1)))),\
                                       axis = 1).astype(int)

        # Get annotations for this bandwidth
        temp_anno_list = add_annotations(temp_anno_mat, song_length)
        full_list.append(temp_anno_list)
    
    #Sort the list    
    out_list = np.concatenate(full_list)
    tem_out_lst = np.lexsort([out_list[:,0], out_list[:,2], out_list[:,5]\
                              ,out_list[:,4]])
    out_list = out_list[tem_out_lst,:]
        
    return out_list

