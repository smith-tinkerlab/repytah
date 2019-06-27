# coding: utf-8

"""
FUNCTION: FIND_ADD_EROWS_BOTH_CHECK_NO_ANNO: Finds diagonals of length K that end at
the same time step as previously found repeats of length K. 

INPUT: 
    1. LST_NO_ANNO: List of pairs of repeats
    2. CHECK_INDS: List of ending indices for repeats of length K
    that we use to check LST_NO_ANNO for more repeats of
    length K
    3. K: Length of repeats that we are looking for 

OUTPUT: 
    ADD_ROWS: List of newly found pairs of repeats of length K
    that are contained in larger repeats in LST_NO_ANNO
"""

import numpy as np

def find_add_erows(lst_no_anno, check_inds, k):
    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 20, 40, 50, 15]])
    check_inds = np.array([10, 55, 40, 55, 20, 50])
    k = 10
    L = lst_no_anno
    search_inds = (L[:,4] > k)

    #Multiply by SEARCH_INDS to narrow search to pairs of repeats of length greater than K
    EI = np.multiply(L[:,1], search_inds)
    EJ = np.multiply(L[:,3], search_inds)

    #Loop over CHECK_INDS
    for i in range(check_inds.size): 
        ci = check_inds[i]
        
        #Left Check: Check for CI on the left side of the pairs
        # Check if the end index of the left repeat of the pair equals CI
        lnds = (EI == ci)
        
        if lnds.sum(axis = 0) > 0: 
            EJ_li = L[lnds,3]
            l_num = EJ_li.shape[0] 
            l_add = np.append(L[lnds,1] - k + 1, L[lnds,1], (EJ_li - k + 1), 
                               EJ_li, k*np.ones((l_num,1)))
            l_add_left = np.append(L[lnds,0], (L[lnds,1] - k), L[lnds,2], 
                                    (EJ_li - k), (L[lnds,4] - k)) 
            
            ##Add the found pairs of repeats
            add_rows = np.append(l_add, l_add_left)
            add_rows = np.reshape(add_rows, (2,5))
        
        # Right Check: Check for CI on the right side of the pairs
        #Check if the end index of the right repeat of the pair equals CI
        rnds = (EJ == ci) 
        
        if rnds.sum(axis = 0) > 0:
            EI_ri = L[rnds, 1]
            r_rum = EI_ri.shape[0]
            r_add = np.append((EI_ri - k + 1), EI_ri, (L[rnds, 3] - k + 1), 
                              L[rnds,3], k*np.ones((r_rum, 1)))
            r_add_left = np.append(L[rnds, 0], (EI_ri - k), L[rnds, 2], 
                                   (L[rnds, 3] - k), L[rnds, 4] - k) 

            ##Add the found rows 
            add_rows = np.append(r_add, r_add_left)
            add_rows = np.reshape(add_rows, (2,5))
            add_rows = np.append(add_rows, add_rows).astype(int)

        return(add_rows) 
        




