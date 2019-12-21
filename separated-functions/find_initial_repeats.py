# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
"""
find_initial_repeats

We're completely recoding it

@author: jorda

Code for find_initial_repeats (lightup_lst_with_thresh_bw in MATLAB)
up to line 63

This is the updated version recoded by Jordan and Denise. 

Update: this file defined the function as find_all_repeats instead of 
find_initial_repeats. 
"""

def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    
    #bw_len = bandwidth_vec.shape[1]
    
    int_all = []
    sint_all = []
    eint_all = []
    mint_all = []
    
    for bw in bandwidth_vec:
        if bw > thresh_bw:
            # Use convolution mx to find diagonals of length bw
            id_mat = np.identity(bw)
            diagonal_mat = signal.convolve2d(thresh_mat, id_mat, 'valid')
            
            # Mark where diagonals of length bw start
            diag_markers = (diagonal_mat == bw)
            
            if np.sum(diag_markers) > 0:
                full_bw = bw
                # 1) Search outside the overlapping shingles
                # Search for paired starts
                upper_tri = np.triu(diag_markers, full_bw)
                (start_i, start_j) = upper_tri.nonzero()
                num_nonoverlaps = start_i.shape[0]
                
                # Find the matching ends for the prevously found starts
                match_i = start_i + (full_bw - 1)
                match_j = start_j + (full_bw - 1)
                
                # List pairs of starts with their ends and the widths of the
                # non-overlapping intervals
                int_lst = np.column_stack([start_i, match_i, start_j, match_j, full_bw * np.ones(num_nonoverlaps,1)])
    
                # Add the new non-overlapping intervals to the full list of
                # non-overlapping intervals
                int_all.append(int_lst)
                
                # 2) Overlaps: Search only the overlaps in shingles
                shin_ovrlaps = np.nonzero((np.tril(np.triu(diag_markers), 
                                                   (full_bw - 1))))
                start_i_shin = np.array(shin_ovrlaps[0]) # row
                start_j_shin = np.array(shin_ovrlaps[1]) # column
                num_ovrlaps = start_i.shape[0]
                
                if(num_ovrlaps == 1 and start_i_shin == start_j_shin):
                    sint_lst = np.column_stack([start_i_shin, (start_i_shin + (full_bw - 1)), start_j_shin, (start_j_shin + (full_bw - 1)), full_bw])
                    sint_all.append(sint_lst)
                    
                elif num_ovrlaps > 0:
                        # Since you are checking the overlaps you need to 
                        # cut these intervals into pieces: left, right, and 
                        # middle. NOTE: the middle interval may NOT exist.
                        
                        # Create vector of 1's that is the length of the 
                        # number of overlapping intervals. This is used a lot. 
                        ones_no = np.ones(num_ovrlaps,1)
                        
                        # 2a) Left Overlap
                        K = start_j_shin - start_i_shin 
                        # NOTE: matchJ - matchI will also equal this, since the 
                        # intervals that are overlapping are the same length. 
                        # Therefore the "left" non-overlapping section is the 
                        # same length as the "right" non-overlapping section. 
                        # It does NOT follow that the "middle" section is 
                        # equal to either the "left" or "right" piece. It is
                        # possible, but unlikely.
                        
                        sint_lst = np.column_stack([start_i_shin, (start_j_shin - ones_no), start_j_shin, (start_j_shin + K - ones_no), K])
                        i_s = np.argsort(K)
                        sint_lst = sint_lst[i_s,]
                        
                        # Remove pairs that fall below the bandwidth threshold
                        cut_s = np.amin((sint_lst[:,4]).nonzero())
                        sint_lst = sint_lst[cut_s:,]
                        
                        # Add new left overlapping intervals to the full list
                        # of left overlapping intervals
                        sint_all.append(sint_lst)
                        
                        # 2b) Right Overlap
                        end_i_right = start_i_shin + (full_bw - 1)
                        end_j_right = start_j_shin + (full_bw - 1)
                        eint_lst = np.column_stack([(end_i_right + ones_no - K),
                                                    end_i_right,
                                                    (end_i_right + ones_no), 
                                                    end_j_right, K])
                        ie = np.argsort(K)
                        eint_lst = eint_lst[ie,]
                        
                        # Remove pairs that fall below the bandwidth threshold
                        cut_e = np.amin((eint_lst[:,4]).nonzero())
                        eint_lst = eint_lst[cut_e:,]
                        
                        # Add the new right overlapping intervals to the full list
                        # of right overlapping intervals
                        eint_all.append(eint_lst)
                        
                        # 2b) Middle Overlap
                        mnds = (end_i_right - start_j_shin - K + ones_no) > 0
                        start_i_mid = (start_j_shin * mnds)
                        end_i_mid = (end_i_right * (mnds) - K * (mnds))
                        start_j_mid = (start_j_shin * (mnds) + K * (mnds))
                        end_j_mid = (end_i_mid * mnds)
                        km = (end_i_mid * (mnds) - start_j_mid * 
                              (mnds) - K * (mnds) + ones_no * (mnds))
                        
                        if mnds.sum() > 0:
                            mint_lst = np.column_stack([start_i_mid, end_i_mid,
                                                        start_j_mid, end_j_mid,
                                                        km])
                            im = np.argsort(km)
                            mint_lst = mint_lst[im,]
                            
                            # Add the new middle overlapping intervals to 
                            # the full list of middle overlapping intervals
                            mint_all.append(mint_lst)
                            
                        # 4) Remove found diagonals of length bw from consideration
                        sdm = stretch_diags(diag_markers, bw)
                        thresh_mat = thresh_mat - sdm
                        
                if thresh_mat.sum() == 0:
                        break
                            
    # Combine non-overlapping intervals with the left, right, and middle parts
    # of the overlapping intervals
    ovrlap_lst = np.concatenate((sint_all, eint_all, mint_all), axis = 0)
    all_lst = np.concatenate((int_all, ovrlap_lst))
    all_lst = filter(None, all_lst)

    # Sort the list by bandwidth size
    I = np.argsort(all_lst[:,4])
    all_lst = all_lst[I,]                     
    
    return all_lst 

                
    
    
                
                
            
            
    