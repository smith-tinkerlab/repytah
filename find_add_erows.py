# coding: utf-8

import numpy as np

def find_add_erows(lst_no_anno, check_inds, k):
    """
    Finds diagonals of length K that end at
    the same time step as previously found repeats of length K. 

    Attributes
    ----------
        lst_no_anno: np.array
            list of pairs of repeats
        check_inds: np.array
            list of ending indices for repeats of length k that we use 
            to check lst_anno_no for more repeats of length k
        k: number 
            length of repeats that we are looking for 

    Returns
    -------
        add_rows: np.array
            list of newly found pairs of repeats of length k that are 
            contained in larger repeats in lst_no_anno
    """
    #List of pairs of repeats 
    lst_no_anno = np.array([[1, 15, 31, 45, 15], 
                            [1, 10, 46, 55, 10], 
                            [31, 40, 46, 55, 10],
                            [10, 25, 41, 55, 15]])
    #Ending indices of length k (length of repeat we are looking for)
    check_inds = np.array([10, 55, 40, 55])
    #Length of repeat we are looking for 
    k = 10
    L = lst_no_anno
    #Logical, which pairs of repeats have length greater than k? (T return 1, F return 0)
    search_inds = (L[:,4] > k)

    #Multiply ending index of all repeats "I" by search_inds
    EI = np.multiply(L[:,1], search_inds)
    #Multipy ending index of all repeats "J" by search_inds
    EJ = np.multiply(L[:,3], search_inds)

    #Loop over CHECK_INDS
    for i in range(check_inds.size): 
        #print()
        ci = check_inds[i]
        #print("loop:", i, "ci:", ci)
        
    #Left Check: Check for CI on the left side of the pairs
        #Check if the end index of the repeat "I" equals CI
        lnds = (EI == ci) 
        #print("lnds:", lnds)
        
        #Find new rows 
        if lnds.sum(axis = 0) > 0: #If the sum across (row) is greater than 0 
            #Find the 3rd entry of the row (lnds) whose starting index of repeat "J" equals CI
            EJ_li = L[lnds,3]
            
            #Number of rows in EJ_li 
            l_num = EJ_li.shape[0] 
            #print("l_num:", l_num)
            
            #Found pair of repeats on the left side
            #l_add = np.concatenate((L[lnds,1] - k + 1, L[lnds,1], (EJ_li - k + 1), EJ_li, k*np.ones((l_num,1)
            one_lsi = L[lnds,1] - k + 1     #Starting index of found repeat i
            one_lei = L[lnds,1]             #Ending index of found repeat i
            one_lsj = EJ_li - k + 1         #Starting index of found repeat j
            one_lej = EJ_li                 #Ending index of found repeat j
            one_lk = k*np.ones((l_num,1))   #Length of found pair of repeats, i and j 
            l_add = np.concatenate((one_lsi, one_lei, one_lsj, one_lej, one_lk), axis = None)
            #print("l_add:", l_add)
            
            #Found pair of repeats on the right side
            #l_add_left = np.concatenate((L[lnds,0], (L[lnds,1] - k), L[lnds,2], (EJ_li - k), (L[lnds,4] - k)), axis = None)
            two_lsi = L[lnds,0]             #Starting index of found repeat i 
            two_lei = L[lnds,1] - k         #Ending index of ofund repeat i
            two_lsj = L[lnds,2]             #Starting index of found repeat j 
            two_lej = EJ_li - k             #Ending index of found repeat j
            two_lk = L[lnds, 4] - k         #Length of found pair of repeats, i and j 
            l_add_left = np.concatenate((two_lsi, two_lei, two_lsj, two_lej, two_lk), axis = None)
            #print("l_add_right:", l_add_right)
            
            #Stack the found rows vertically 
            add_rows = np.vstack((l_add, l_add_left))
            
            #Stack all the rows found on the left side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0)
            #print("add_rows:", add_rows)
            
    # Right Check: Check for CI on the right side of the pairs
        #Check if the end index of the right repeat of the pair equals CI
        rnds = (EJ == ci)
        #print("rnds:", rnds)
        
        #Find new rows
        if rnds.sum(axis = 0) > 0: #If the sum across (row) is greater than 0 
            #Find the 1st entry of the row (lnds) whose ending index of repeat "I" equals CI
            EI_ri = L[rnds, 1]
            #Number of rows in EJ_ri                    
            r_num = EI_ri.shape[0]
                               
            #Found pair of repeats on the left side 
            #r_add = np.concatenate(((EI_ri - k + 1), EI_ri, (L[rnds, 3] - k + 1), L[rnds,3], k*np.ones((r_rum, 1))), axis = None)
            one_rsi = EI_ri - k + 1         #Starting index of found repeat i 
            one_rei = EI_ri                 #Ending index of found repeat i 
            one_rsj = L[rnds, 3] - k + 1    #Starting index of found repeat j
            one_rej = L[rnds,3]             #Ending index of found repeat j 
            one_rk = k*np.ones((r_num, 1))  #Length of found pair or repeats, i and j 
            r_add = np.concatenate((one_rsi, one_rei, one_rsj, one_rej, one_rk), axis = None)
            
            #Found pairs on the right side 
            r_add_left = np.concatenate((L[rnds, 0], (EI_ri - k), L[rnds, 2], (L[rnds, 3] - k), L[rnds, 4] - k), axis = None) 
            two_rsi = L[rnds, 0]            #Starting index of found repeat i  
            two_rei = EI_ri - k             #Ending index of found repeat i 
            two_rsj = L[rnds, 2]            #Starting index of found repeat j
            two_rej = L[rnds, 3] - k        #Ending index of found repeat j 
            two_rk = L[rnds, 4] - k         #Length of found pair or repeats, i and j 
            r_add_right = np.concatenate((two_rsi, two_rei, two_rsj, two_rej, two_rk), axis = None) 
            
            #Stack the found rows vertically 
            add_rows = np.vstack((r_add, r_add_right))
            
            #Stack all the rows found on the right side of the pairs 
            add_rows = np.concatenate((add_rows, add_rows), axis = 0).astype(int)
            #print(add_rows)                   
    return add_rows
