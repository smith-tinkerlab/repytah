#line 217: 
#https://stackoverflow.com/questions/2828059/sorting-arrays-in-np-by-column
import numpy as np
from scipy import signal
from utilities import stretch_diags

def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    """
    Looks for the largest repeated structures in thresh_mat. Finds all 
    repeated structures, represented as diagonals present in thresh_mat, 
    and then stores them with their start/end indices and lengths in a 
    list. As each diagonal is found, they are removed to avoid identifying
    repeated sub-structures. 
  
    Args
    ----
        thresh_mat: np.array[int]
            thresholded matrix that we extract diagonals from

        bandwidth_vec: np.array[1D,int]
            vector of lengths of diagonals to be found. Should be 1,2,3,..... n where n = num_timesteps

        thresh_bw: int
            smallest allowed diagonal length

    Returns
    -------
        all_lst: np.array[int]
            list of pairs of repeats that correspond to 
            diagonals in thresh_mat
    """

     # Initialize the input and temporary variables
    thresh_temp = thresh_mat
    
    #For removing already found diagonals 
    Tbw = thresh_bw; 
    # Interval list for non-overlapping pairs    
    int_all =  np.empty((0,5), int) 
    
    # Interval list for the left side of the overlapping pairs
    sint_all = np.empty((0,5), int)
    
    # Interval list for the right side of the overlapping pairs
    eint_all = np.empty((0,5), int) 
    
     # Interval list for the middle of the overlapping pairs if they exist
    mint_all = np.empty((0,5), int) 

    #Loop over all bandwidths
    for bw in np.flip((bandwidth_vec)):
        if bw > thresh_bw:
            #Use convolution matrix to find diagonals of length bw 
            id_mat = np.identity(bw) 
            # Search for diagonals of length band_width
            diagonal_mat = signal.convolve2d(thresh_temp, id_mat, 'valid')
        
            # Mark where diagonals of length band_width start
            diag_markers = (diagonal_mat == bw).astype(int)
            
            if sum(diag_markers).any() > 0:
                full_bw = bw
                
                #1) Non-Overlaps: Search outside the overlapping shingles
                upper_tri = np.triu(diag_markers, full_bw)
                
                # Search for paired starts 
                (start_i, start_j) = upper_tri.nonzero() 
                start_i = start_i + 1
                start_j = start_j + 1
                # Find the matching ends for the previously found starts 
                match_i = start_i + (full_bw - 1)
                match_j = start_j + (full_bw - 1)
                # List pairs of starts with their ends and the widths of the
                # non-overlapping interval
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
                shin_ovrlaps = np.nonzero((np.tril(np.triu(diag_markers, -1),
                                                   (full_bw-1))))
                
                #print('shin_ovrlaps:',shin_ovrlaps)
                start_i_shin = np.array(shin_ovrlaps[0]+1) # row
                start_j_shin = np.array(shin_ovrlaps[1]+1) # column
                num_ovrlaps = len(start_i_shin)
                #print('start_i_shin:',start_i_shin)
                #print('start_j_shin:',start_j_shin)
                if (num_ovrlaps == 1 and start_i_shin == start_j_shin):
                    i_sshin = np.concatenate((start_i_shin,start_i_shin+ (full_bw - 1)),axis = None)
                    j_sshin = np.concatenate((start_j_shin,start_j_shin+ (full_bw - 1)),axis = None)
                    i_j_pairs = np.hstack((i_sshin,j_sshin))
                    sint_lst = np.hstack((i_j_pairs,full_bw))
                    sint_all = np.vstack((sint_all, sint_lst))
                    
                elif num_ovrlaps > 0:
                    # Since you are checking the overlaps you need to cut these
                    # intervals into pieces: left, right, and middle. NOTE: the
                    # middle interval may NOT exist
                    
                    # Vector of 1's that is the length of the number of
                    # overlapping intervals. This is used a lot. 
                    ones_no = np.ones(num_ovrlaps);
                    
                    # 2a) Left Overlap
                    K = start_j_shin - start_i_shin
                    i_sshin = np.vstack((start_i_shin[:], (start_j_shin[:] - ones_no[:]))).T
                    j_sshin = np.vstack((start_j_shin[:], (start_j_shin[:] + K - ones_no[:]))).T
                    
                    sint_lst = np.column_stack((i_sshin,j_sshin,K.T))
                    
                    i_s = np.argsort(K) # Return the indices that would sort K
                    sint_lst = sint_lst[i_s,]
                    # Remove the pairs that fall below the bandwidth threshold
                    cut_s = np.argwhere((sint_lst[:,4] > Tbw))
                    cut_s = cut_s.T
                    sint_lst = sint_lst[cut_s][0]

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
                    
                    # Remove the pairs that fall below the bandwidth threshold
                    cut_e = np.argwhere((eint_lst[:,4] > Tbw))
                    cut_e = cut_e.T
                    eint_lst = eint_lst[cut_e][0]
    
                    # Add the new right overlapping intervals to the full list of
                    # right overlapping intervals
                    eint_all = np.vstack((eint_all,eint_lst))

                    # 2) Middle Overlap
                    mnds = (end_i_shin - start_j_shin - K + ones_no) > 0
                    if sum(mnds) > 0:
                        i_middle = (np.vstack((start_j_shin[:], end_i_shin[:] - K ))) * mnds
                        i_middle = i_middle.T
                        i_middle = i_middle[np.all(i_middle != 0, axis=1)]


                        j_middle = (np.vstack((start_j_shin[:] + K, end_i_shin[:])))  * mnds 
                        j_middle = j_middle.T
                        j_middle = j_middle[np.all(j_middle != 0, axis=1)]


                        k_middle = np.vstack((end_i_shin[mnds] - start_j_shin[mnds] - K[mnds] + ones_no[mnds]))
                        k_middle = k_middle.T
                        k_middle = k_middle[np.all(k_middle != 0, axis=1)]

                        mint_lst = np.column_stack((i_middle,j_middle,k_middle.T))

                    # Remove the pairs that fall below the bandwidth threshold 
                        cut_m = np.argwhere((mint_lst[:,4] > Tbw))
                        cut_m = cut_m.T
                        mint_lst = mint_lst[cut_m][0]
                        mint_all = np.vstack((mint_all, mint_lst))
                    
                    # Remove found diagonals of length BW from consideration
        SDM = stretch_diags(diag_markers, bw)
        thresh_temp = thresh_temp - SDM

        if thresh_temp.sum() == 0:
            break
        
    out_lst = np.vstack((sint_all, eint_all, mint_all))
    all_lst = np.vstack((int_all, out_lst))
    
    inds = np.argsort(all_lst[:,4])
    all_lst = np.array(all_lst)[inds]
    
    all_lst_in = np.lexsort((all_lst[:,0],all_lst[:,2],all_lst[:,4]))
    all_lst = all_lst[all_lst_in]
    
    return(all_lst.astype(int))