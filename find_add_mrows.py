import numpy as np

def find_add_mrows(lst_no_anno, check_inds, k): 
    """
    Finds diagonals of length k that neither start nor end at the same time 
    steps as previously found repeats of length k. 

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
    #List of pairs of repeats 
    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 25, 41, 55, 15]])
    #Ending indices of length k (length of repeat we are looking for)
    check_inds = np.array([10, 25, 41, 55, 15])
    #Length of repeat we are looking for 
    k = 10
    
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






