# coding: utf-8

"""
FUNCTION: FIND_ADD_MROWS_BOTH_CHECK_NO_ANNO: Finds diagonals of length K
that neither start nor end at the same time steps as previously found 
repeats of length K. 

INPUT: 
    1. LST_NO_ANNO: List of pairs of repeats
    2. CHECK_INDS: List of ending indices for repeats of length K that we
    use to check LST_NO_ANNO for more repeats of length K 
    3. K: Length of repeats that we are looking for 
    
OUTPUT: 
    ADD_ROWS: List of newly found pairs of repeats of length K that are 
    contained in larger repeats in LST_NO_ANNO 
    
"""

import numpy as np

def find_add_mrows(lst_no_anno, check_start_inds, k):

    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 20, 40, 50, 15]])
    check_inds = np.array([10, 55, 40, 55, 20, 50])
    k = 10

    L = lst_no_anno 
    search_inds = (L[:,4] > k)

    SI = np.multiply(L[:,0], search_inds)
    SJ = np.multiply(L[:,2], search_inds)
    EI = np.multiply(L[:,1], search_inds)
    EJ = np.multiply(L[:,3], search_inds)

    #Loop over CHECK_INDS 
    for i in range(check_inds.size): 
        ci = check_inds[i]
        
        #Left Check: Check for CI on the left side of the pairs 
        lnds = ((SI < ci) + (EI > (ci + k -1)) == 2)
        
        if lnds.sum(axis = 0) > 0:
            SJ_li = L[lnds,2]
            EJ_li = L[lnds,3]
            l_num = SJ_li.shape[0]
            
            #Left side of left pair
            l_left_k = ci*np.ones(l_num,1) - L[lnds,0]
     
            #Note: E-S+1 = (CI - 1) - L(LNDS,1) + 1
            l_add_left = np.concatenate((L[lnds,0], (ci - 1 * np.ones((l_num,1))), SJ_li, (SJ_li + l_left_k - np.ones((l_num,1))), l_left_k), axis = None)
            
            # Middle of left pair
            # Note: L_MID_K = K
            l_add_mid = np.concatenate(((ci*np.ones((l_num,1))), (ci+k-1)*np.ones((l_num,1)), SJ_li + l_left_k, SJ_li + l_left_k + (k-1)*np.ones((l_num,1)), k*np.ones((l_num,1))), axis = None) 
            
            # Right side of left pair
            l_right_k = np.concatenate((L[lnds, 1] - ((ci + k) - 1) * np.ones((l_num,1))), axis = None)
            
            l_add_right = np.concatenate((((ci + k)*np.ones((l_num,1))), L[lnds,1], (EJ_li - l_right_k + np.ones((l_num,1))), EJ_li, l_right_k), axis = None)
            
            # Add the found rows        
            add_rows = np.concatenate((l_add_left, l_add_mid, l_add_right), axis= 0)
            add_rows = np.reshape(add_rows, (3,5))
        
        #Right Check: Check for CI on the right side of the pairs
        rnds = ((SJ < ci) + (EJ > (ci + k - 1)) == 2); 
        
        if rnds.sum(axis = 0) > 0:
            SI_ri = L[rnds,0]
            EI_ri = L[rnds,1]
            r_num = SI_ri.shape[0]
            
            #Left side of right pair
            r_left_k = ci*np.ones((r_num,1)) - L[rnds,2]
            #Note: E-S+1 = (CI - 1) - L(LNDS,3) + 1
            r_add_left = np.concatenate((SI_ri, (SI_ri + r_left_k - np.ones((r_num,1))), L[rnds,3], (ci - 1)*np.ones((r_num,1)), r_left_k), axis = None)
            
            #Middle of right pair
            # Note: R_MID_K = K
            r_add_mid = np.concatenate(((SI_ri + r_left_k),(SI_ri + r_left_k + (k - 1)*np.ones((r_num,1))), ci*np.ones((r_num,1)), (ci + k - 1)*np.ones((r_num,1)), k*np.ones((r_num,1))), axis = None)

            #Right side of right pair
            r_right_k = L[rnds, 3] - ((ci + k) - 1)*np.ones((r_num,1))
            r_add_right = np.concatenate((EI_ri - r_right_k + np.ones((r_num,1)),EI_ri, (ci + k)*np.ones((r_num,1)), L[rnds,3], r_right_k), axis = None)
            
            add_rows = np.concatenate((r_add_left, r_add_mid, r_add_right), axis = 0)
            add_rows = np.reshape(add_rows, (3,5))
            
            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)

        return(add_rows)





