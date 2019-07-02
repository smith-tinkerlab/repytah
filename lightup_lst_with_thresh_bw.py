#line 217: https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
import numpy
from scipy import signal
"""
Finds all diagonals present in THRESH_MAT,
removing each diagonal as it is found.

args
    thresh_mat np.array[int]:
        Thresholded matrix that we extract diagonals from
    bandwidth_vec np.array[1D,int]:
        Vector of lengths of diagonals to be found
    thresh_bw int:
        Smallest allowed diagonal length

returns
    all_lst np.array[int]:
        list of pairs of repeats that 
        correspond to diagonals in thresh_mat
"""
def lightup_lst_with_thresh_bw(thresh_mat, bandwidth_vec, thresh_bw)
    b = numpy.size(bandwidth_vec)

    #create empty arrays with correct dimensions
    int_all =  numpy.zeros(5)
    sint_all = numpy.zeros(5)
    eint_all = numpy.zeros(5)
    mint_all = numpy.zeros(5)

    #loop over all bandwidths
    for bw in bandwidth_vec:
        if bw > thresh_bw:
        #search for diagonals of length BW
            thresh_mat_size = numpy.size(thresh_mat)
            DDM_rename = signal.convolve2d(thresh_mat[0:thresh_mat_size, 0:thresh_mat_size],numpy.eye(bw),'valid')
            #mark where diagonals of length BW start
            thresh_DDM_rename = (DDM_rename == bw) 
            if numpy.sum(numpy.sum(thresh_DDM_rename)) > 0:
                full_bw = bw
                #1) Non-Overlaps: Search outside the overlapping shingles

                #Find the starts that are paired together
                #returns tuple of lists (python) https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
                #need to add 1 to return correct number of nonzero ints matlab vs python
                overlaps = numpy.nonzero(numpy.triu(thresh_DDM_rename, (full_bw)))

                startI = numpy.array(overlaps[0])
                num_nonoverlaps = numpy.size(startI)
                startJ = numpy.array(overlaps[1])
                #Find the matching ends EI for SI and EJ for SJ
                matchI = (startI + full_bw-1);
                matchJ = (startJ + full_bw-1);

                 #List pairs of starts with their ends and the widths of the
                #non-overlapping interval

                #int_lst = numpy.column_stack((startI, matchI, startJ, matchJ, full_bw*numpy.ones(num_nonoverlaps)))
                #*numpy.ones(num_nonoverlaps)
                int_lst = numpy.column_stack([startI, matchI, startJ, matchJ, full_bw])
                #Add the new non-overlapping intervals to the full list of
                #non-overlapping intervals
                #https://stackoverflow.com/questions/3881453/numpy-add-row-to-array
                int_all = numpy.vstack([int_all, int_lst])

                # 2) Overlaps: Search only the overlaps in shingles
                #returns tuple (python) 
                shingle_overlaps = numpy.nonzero(numpy.tril(numpy.triu(thresh_DDM_rename), (full_bw-1)))
                #gets list for I and J [1,2,3,4] turn those to numpy, transpose them vertically
                startI_inShingle = numpy.array(shingle_overlaps[0]) 
                startJ_inShingle = numpy.array(shingle_overlaps[1]) 
                #find number of overlaps
                num_overlaps = numpy.size(startI_inShingle)
                if (num_overlaps == 1 and startI_inShingle == startJ_inShingle):
                    sint_lst = numpy.column_stack([startI_inShingle, startI_inShingle,(startI_inShingle + (full_bw - 1)),startJ_inShingle,(startJ_inShingle + (full_bw - 1)), full_bw])
                    sint_all = numpy.vstack([sint_all, sint_lst])

                elif num_overlaps>0:
                        #Since you are checking the overlaps you need to cut these
                        #intervals into pieces: left, right, and middle. NOTE: the
                        #middle interval may NOT exist
                    # Vector of 1's that is the length of the number of
                    # overlapping intervals. This is used a lot. 
                    ones_no = numpy.ones(num_overlaps);

        #2a) Left Overlap
                    #remain consistent with being matlab -1
                    K = startJ_inShingle - startI_inShingle
                    sint_lst = numpy.column_stack([startI_inShingle, (startJ_inShingle - ones_no), startJ_inShingle, (startJ_inShingle + K - ones_no), K])
                    #returns list of indexes of sorted list
                    Is = numpy.argsort(K)
                    #turn array vertical
                    Is.reshape(numpy.size(Is), 1)
                    #extract all columns from row Is
                    sint_lst = sint_lst[Is, :]
                                    #grab only length column
                    i = 0
                    for length in numpy.transpose(sint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                        #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                sint_lst = numpy.delete(sint_lst, (i-1), axis=0)
                            sint_all = numpy.vstack([sint_all, sint_lst])
                                #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1
                    #endfor
                    #2b right overlap
                    EIo = startI_inShingle + (full_bw)
                    EJo = startJ_inShingle + (full_bw)
                    eint_lst = numpy.column_stack([(EIo + ones_no - K), EIo, (EIo + ones_no), EJo, K])
                    Ie = numpy.argsort(K)
                    #turn result to column
                    Ie.reshape(numpy.size(Ie),1)
                    eint_lst = eint_lst[Ie, :]
                    
                    #grab only length column
                    i = 0
                    for length in numpy.transpose(eint_lst[:,4]):
                        #if this length is greater than thresh_bw-- we found our index
                        if length > thresh_bw:
                            #if its not the first row
                            if(i!=0):
                                #delete rows that fall below threshold
                                eint_lst = numpy.delete(eint_lst, (i-1), axis=0)
                            eint_all = numpy.vstack([eint_all, eint_lst])
                            #after we found the min that exceeds thresh_bw... break
                            break
                        i=i+1

                    # 2) Middle Overlap
                    #returns logical 0 or 1 for true or false
                    mnds = (EIo - startJ_inShingle - K + ones_no) > 0
                    #for each logical operator convert to 0 or 1
                    for operator in mnds:
                        if operator is True:
                            operator = 1
                        else:
                            operator = 0
                    SIm = startJ_inShingle*(mnds)
                    EIm = (EIo*(mnds) - K*(mnds))
                    SJm = (startJ_inShingle*(mnds) + K*(mnds))
                    EJm = EIo*(mnds)
                    #fixes indexing here because length starts at 1 and indexes start at 0
                    Km = (EIo*(mnds) - startJ_inShingle*(mnds) - K*(mnds) +ones_no*(mnds))-1
                    if numpy.sum(numpy.sum(mnds)) > 0 : 
                        mint_lst = numpy.column_stack([SIm, EIm, SJm, EJm, Km])
                        #revert for same reason
                        Km = Km+1
                        Im = numpy.argsort(Km)
                        #turn array to column
                        Im.reshape(numpy.size(Im), 1)
                        mint_lst = mint_lst[Im, :]

                       #Remove the pairs that fall below the bandwidth threshold
                        #grab only length column
                        i = 0
                        for length in numpy.transpose(mint_lst[:,4]):
                            #if this length is greater than thresh_bw-- we found our index
                            if length > thresh_bw:
                            #if its not the first row
                                if(i!=0):
                                    #delete rows that fall below threshold
                                    mint_lst = numpy.delete(mint_lst, (i-1), axis=0)
                                mint_all = numpy.vstack([mint_all, mint_lst])
                                #after we found the min that exceeds thresh_bw... break
                                break
                            i=i+1
                        #endfor
                    #endif line 143 numpy.sum(numpy.sum(mnds)) > 0
                #endif line 67 (num_overlaps == 1 and startI_inShingle == startJ_inShingle)

                                    #returns matrix with diags in it
                SDM = stretch_diags(DDM_rename, bw)
                thresh_mat = thresh_mat - SDM

                if numpy.sum(numpy.sum(thresh_mat)) == 0:
                    break
                #endIf line 174
            #endIf line 34 numpy.sum(numpy.sum(thresh_DDM_rename)) > 0
       #endIf line 28 bw > thresh_bw
    #endfor
     #Combine non-overlapping intervals with the left, right, and middle parts
     #of the overlapping intervals
    #remove leading 0s if lists aren't empty
    #create empty final list
    all_lst = numpy.zeros(5)
    #for each list:
       #if the list isn't empty delete the first row (empty 0s) and append to final list

    if int_all.ndim != 1:
        int_all = numpy.delete(int_all, 0, axis=0)
        all_lst = numpy.vstack([all_lst,int_all])

    if sint_all.ndim != 1:
        sint_all = numpy.delete(sint_all, 0, axis=0)
        all_lst = numpy.vstack([all_lst,sint_all])

    if eint_all.ndim != 1:
        eint_all = numpy.delete(eint_all, 0, axis=0)
        all_lst = numpy.vstack([all_lst,eint_all])

    if mint_all.ndim != 1:
        mint_all = numpy.delete(mint_all, 0, axis=0)
        all_lst = numpy.vstack([all_lst,mint_all])
    #if there is anything in the final list
    if all_lst.ndim != 1:
        #remove leading empty 0s line
        all_lst = numpy.delete(all_lst, 0, axis=0)
        # Sort the list by bandwidth size
        all_lst[all_lst[:,4].argsort()]
    
    #return final list
    return all_lst
