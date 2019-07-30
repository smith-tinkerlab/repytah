#line 217: 
#https://stackoverflow.com/questions/2828059/sorting-arrays-in-np-by-column
import numpy as np
from scipy import signal

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

    b = np.size(bandwidth_vec)

    #Create empty lists to store arrays
    int_all =  []
    sint_all = []
    eint_all = []
    mint_all = []

    #Loop over all bandwidths
    for bw in bandwidth_vec:
        if bw > thresh_bw:
            #Search for diagonals of length bw
            thresh_mat_size = np.size(thresh_mat)
            
            DDM_rename = signal.convolve2d(thresh_mat[0:thresh_mat_size, \
                                                      0:thresh_mat_size],\
    np.eye(bw), 'valid')
            #Mark where diagonals of length bw start
            thresh_DDM_rename = (DDM_rename == bw) 
            if np.sum(np.sum(thresh_DDM_rename)) > 0:
                full_bw = bw
                #1) Non-Overlaps: Search outside the overlapping shingles

                #Find the starts that are paired together
                #returns tuple of lists (python) https://docs.scipy.org/doc/np/reference/generated/np.nonzero.html
                #need to add 1 to return correct number of nonzero ints matlab vs python
                overlaps = np.nonzero(np.triu(thresh_DDM_rename, (full_bw)))

                startI = np.array(overlaps[0])
                num_nonoverlaps = np.size(startI)
                startJ = np.array(overlaps[1])
                #Find the matching ends EI for SI and EJ for SJ
                matchI = (startI + full_bw-1);
                matchJ = (startJ + full_bw-1);

                 #List pairs of starts with their ends and the widths of the
                #non-overlapping interval

                int_lst = np.column_stack([startI, matchI, startJ, matchJ, full_bw])
                #Add the new non-overlapping intervals to the full list of
                #non-overlapping intervals
                int_all.append(int_lst)
                # 2) Overlaps: Search only the overlaps in shingles
                #returns tuple (python) 
                shingle_overlaps = np.nonzero(np.tril(np.triu(thresh_DDM_rename), (full_bw-1)))
                #gets list for I and J [1,2,3,4] turn those to np, transpose them vertically
                startI_inShingle = np.array(shingle_overlaps[0]) 
                startJ_inShingle = np.array(shingle_overlaps[1]) 
                #find number of overlaps
                num_overlaps = np.size(startI_inShingle)
                if (num_overlaps == 1 and startI_inShingle == startJ_inShingle):
                    sint_lst = np.column_stack([startI_inShingle, startI_inShingle,(startI_inShingle + (full_bw - 1)),startJ_inShingle,(startJ_inShingle + (full_bw - 1)), full_bw])
                    sint_all.append(sint_lst)
                elif num_overlaps>0:
                        #Since you are checking the overlaps you need to cut these
                        #intervals into pieces: left, right, and middle. NOTE: the
                        #middle interval may NOT exist
                    # Vector of 1's that is the length of the number of
                    # overlapping intervals. This is used a lot. 
                    ones_no = np.ones(num_overlaps);

                    #2a) Left Overlap
                    #remain consistent with being matlab -1
                    K = startJ_inShingle - startI_inShingle
                    sint_lst = np.column_stack([startI_inShingle, (startJ_inShingle - ones_no), startJ_inShingle, (startJ_inShingle + K - ones_no), K])
                    #returns list of indexes of sorted list
                    Is = np.argsort(K)
                    #turn array vertical
                    Is.reshape(np.size(Is), 1)
                    #extract all columns from row Is
                    sint_lst = sint_lst[Is, :]
                                    #grab only length column
                    i = 0
                    for length in np.transpose(sint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                        #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                sint_lst = np.delete(sint_lst, (i-1), axis=0)
                            sint_all.append(sint_lst)
                                #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1
                    #endfor
                    #2b right overlap
                    endI_right = startI_inShingle + (full_bw)
                    endJ_right = startJ_inShingle + (full_bw)
                    eint_lst = np.column_stack([(endI_right + ones_no - K), endI_right, (endI_right + ones_no), endJ_right, K])
                    indexes = np.argsort(K)
                    #turn result to column
                    indexes.reshape(np.size(indexes),1)
                    eint_lst = eint_lst[indexes, :]
                    
                    #grab only length column
                    i = 0
                    for length in np.transpose(eint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                            #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                eint_lst = np.delete(eint_lst, (i-1), axis=0)
                            eint_all.append(eint_lst)
                            #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1

                    # 2) Middle Overlap
                    #returns logical 0 or 1 for true or false
                    mnds = (endI_right - startJ_inShingle - K + ones_no) > 0
                    #for each logical operator convert to 0 or 1
                    for operator in mnds:
                        if operator is True:
                            operator = 1
                        else:
                            operator = 0
                    startI_middle = startJ_inShingle*(mnds)
                    endI_middle = (endI_right*(mnds) - K*(mnds))
                    startJ_middle = (startJ_inShingle*(mnds) + K*(mnds))
                    endJ_middle = endI_right*(mnds)
                    #fixes indexing here because length starts at 1 and indexes start at 0
                    Km = (endI_right*(mnds) - startJ_inShingle*(mnds) - K*(mnds) +ones_no*(mnds))-1
                    if np.sum(np.sum(mnds)) > 0 : 
                        mint_lst = np.column_stack([startI_middle, endI_middle, startJ_middle, endJ_middle, Km])
                        #revert for same reason
                        Km = Km+1
                        Im = np.argsort(Km)
                        #turn array to column
                        Im.reshape(np.size(Im), 1)
                        mint_lst = mint_lst[Im, :]

                       #Remove the pairs that fall below the bandwidth threshold
                        #grab only length column
                        i = 0
                        for length in np.transpose(mint_lst[:,4]):
                            #if this length is greater than thresh_bw-- we found our index
                            if length > thresh_bw:
                            #if its not the first row
                                if(i!=0):
                                    #delete rows that fall below threshold
                                    mint_lst = np.delete(mint_lst, (i-1), axis=0)
                                mint_all.append(mint_lst)
                                #after we found the min that exceeds thresh_bw... break
                                break
                            i=i+1
                        #endfor
                    #endif line 143 np.sum(np.sum(mnds)) > 0
                #endif line 67 (num_overlaps == 1 and startI_inShingle == startJ_inShingle)

                                    #returns matrix with diags in it
                SDM = stretch_diags(DDM_rename, bw)
                thresh_mat = thresh_mat - SDM

                if np.sum(np.sum(thresh_mat)) == 0:
                    break
                #endIf line 174
            #endIf line 34 np.sum(np.sum(thresh_DDM_rename)) > 0
       #endIf line 28 bw > thresh_bw
    #endfor
     #Combine non-overlapping intervals with the left, right, and middle parts
     #of the overlapping intervals
    #remove empty lines from the lists


    out_lst = int_all + sint_all + eint_all + mint_all
    #remove empty lists from final output
    
    all_lst = filter(None, out_lst)

    if out_lst is not None:
        all_lst = np.vstack(out_lst)
    else:
        all_lst = np.array([])
    #return final list
    return all_lst
