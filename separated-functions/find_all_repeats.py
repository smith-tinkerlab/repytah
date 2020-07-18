import numpy as np
from scipy import signal

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
    
    Returns
    -------
    all_lst: np.array
        list of pairs of repeats that correspond to diagonals
        in thresh_mat
        
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
            
            # 1) Non-Overlaps: Search outside the overlapping shingles 
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
            start_i_shin = np.array(shin_ovrlaps[0]) # row
            start_j_shin = np.array(shin_ovrlaps[1]) # column
            num_ovrlaps = len(start_i_shin)
            
            if num_ovrlaps > 0:
                # Since you are checking the overlaps you need to cut these
                # intervals into pieces: left, right, and middle. NOTE: the
                # middle interval may NOT exist
    
                # Vector of 1's that is the length of the number of
                # overlapping intervals. This is used a lot.
                ones_no = np.ones((num_ovrlaps,1))

                # 2a) Left Overlap
                K = start_j_shin - start_i_shin  # NOTE: end_J_overlap - end_I_overlap will also equal this,
                                                       # since the intervals that are overlapping are
                                                       # the same length. Therefore the "left"
                                                       # non-overlapping section is the same length as
                                                       # the "right" non-overlapping section. It does
                                                       # NOT follow that the "middle" section is equal
                                                       # to either the "left" or "right" piece. It is
                                                       # possible, but unlikely.

                
                i_sshin = np.vstack((start_i_shin[:], (start_j_shin[:] - ones_no[:]))).T
                j_sshin = np.vstack((start_j_shin[:], (start_j_shin[:] + K - ones_no[:]))).T
                sint_lst = np.column_stack((i_sshin,j_sshin,K.T))
        
 
                i_s = np.argsort(K) # Return the indices that would sort K
                #Is.reshape(np.size(Is), 1)
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
                #Ie.reshape(np.size(Ie),1)
                eint_lst = eint_lst[i_e:,]
    
                # Add the new right overlapping intervals to the full list of
                # right overlapping intervals
                eint_all = np.vstack((eint_all,eint_lst))
                    
                # 2) Middle Overlap
                mnds = (end_i_shin - start_j_shin - K + ones_no) > 0
                
                i_middle = (np.vstack((start_j_shin[:], end_i_shin[:] - K ))) * mnds
                i_middle = i_middle.T
                i_middle = i_middle[np.all(i_middle != 0, axis=1)]
                
                j_middle = (np.vstack((start_j_shin[:] + K, end_i_shin[:])))  * mnds 
                j_middle = j_middle.T
                j_middle = j_middle[np.all(j_middle != 0, axis=1)]
                
                k_middle = np.vstack((end_i_shin[:] - start_j_shin[:] - K + ones_no[:])) * mnds
                k_middle = k_middle.T
                k_middle = k_middle[np.all(k_middle != 0, axis=1)]
                
                mint_lst = np.column_stack((i_middle,j_middle,k_middle))
                
                mint_all = np.vstack((mint_all, mint_lst))
    
        if thresh_temp.sum() == 0:
            break 
            
    out_lst = np.vstack((sint_all, eint_all, mint_all))
    all_lst = np.vstack((int_all, out_lst))
    
    inds = np.argsort(all_lst[:,4])
    all_lst = np.array(all_lst)[inds]
    
    all_lst_in = np.lexsort((all_lst[:,0],all_lst[:,2],all_lst[:,4]))
    all_lst = all_lst[all_lst_in]
    
    return(all_lst.astype(int))