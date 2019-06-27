# coding: utf-8

"""
FUNCTION:
    FIND_ADD_SROWS_BOTH_CHECK_NO_ANNO: Finds diagonals of length k that start at the same time step 
    as previously found repeats of length k. 
    
    INPUT: 
        1. LST_NO_ANNO: List of pairs of repeats
        2. CHECK_INDS: List of ending indices for repeats of length k 
        that we use to check 1st_no_anno for more repeats of length k 
        3. K: Length of repeats that we are looking for
        
    OUTPUT:
        ADD_ROWS: List of newly found pairs of repeats of length K
        that are contained in larger repeats in LST_NO_ANNO
        
"""
import numpy as np

def find_add_srows(lst_no_anno, check_inds, k):
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

    #Loop over check_inds 
    for i in range(check_inds.size):
        ci = check_inds[i] 
            
        #Left check: check for CI on the left side of the pairs 
        #Check if the starting index of the left repeat of the pair equals CI
        lnds = (SI == ci) 
        
        if lnds.sum(axis = 0) > 0:
            SJ_li = L[lnds, 2] 
            l_num = SJ_li.shape[0] 
            
            l_add = np.append(L[lnds, 0], L[lnds, 0] + k - 1, SJ_li, (SJ_li + k - 1), 
                              np.ones((l_num, 1))*k)
            l_add_right = np.append(L[lnds, 0] + k , L[lnds, 1], SJ_li + k, L[lnds, 3], 
                                    L[lnds, 4] - k)

            ##Add the found rows 
            add_rows = np.append(l_add, l_add_right, axis= 0)
            add_rows = np.reshape(add_rows, (2,5))
            add_rows = np.append(add_rows, add_rows, axis = 0)
            
        #Right Check: check for CI on the right side of the pairs 
        #Check if the the starting index of the right repeat of the pair equals CI
        rnds = (SJ == ci) 

        if rnds.sum(axis = 0) > 0:
            SJ_ri = L[rnds, 0]
            r_num = SJ_ri.shape[0]
            
            r_add = np.append(SJ_ri, (SJ_ri + k - 1), L[rnds, 2], (L[rnds, 2] + k - 1), 
                              k*np.ones((r_num, 1)))
            r_add_right = np.append((SJ_ri + k), L[rnds, 1], (L[rnds, 2] + k), L[rnds,3], 
                                    (L[rnds, 4] -k)) 

            add_rows = np.append(r_add, r_add_right)
            add_rows = np.reshape(add_rows, (2,5)) 
            add_rows = np.append(add_rows, add_rows, axis = 0)

    return add_rows



