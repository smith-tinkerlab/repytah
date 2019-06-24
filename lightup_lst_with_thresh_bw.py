#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy
from scipy import signal
thresh_mat = numpy.array([[1, 1, 0, 1, 0, 0,], [1, 1, 1, 0, 1, 0], [0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
bandwidth_vec = numpy.array([3, 2])
thresh_bw = 2
band_len = numpy.size(bandwidth_vec)

#create empty arrays with correct dimensions
int_all =  numpy.empty(5)
sint_all = numpy.empty(5)
eint_all = numpy.empty(5)
mint_all = numpy.empty(5)


# In[14]:


for i in range(1, band_len):
    #set current bandwidth
    j = band_len-i
    bw = bandwidth_vec[j]
    if bw==thresh_bw:
    #search for diagonals of length BW
        thresh_mat_size = numpy.size(thresh_mat)
        DDM_rename = signal.convolve2d(thresh_mat[1:thresh_mat_size, 1:thresh_mat_size],numpy.eye(bw),'valid')
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
            matchI = startI + full_bw;
            matchJ = startJ + full_bw;
 
             #List pairs of starts with their ends and the widths of the
            #non-overlapping interval
            
            #int_lst = numpy.column_stack((startI, matchI, startJ, matchJ, full_bw*numpy.ones(num_nonoverlaps)))
            int_lst = numpy.column_stack([startI, matchI, startJ, matchJ, full_bw*numpy.ones(num_nonoverlaps)])
            #Add the new non-overlapping intervals to the full list of
            #non-overlapping intervals
            #https://stackoverflow.com/questions/3881453/numpy-add-row-to-array
            int_all = numpy.vstack([int_all, int_lst])
            
            # 2) Overlaps: Search only the overlaps in shingles
            #returns tuple (python) 
            shingle_overlaps = numpy.nonzero(numpy.tril(numpy.triu(thresh_DDM_rename), (full_bw)))
            #gets list for I and J [1,2,3,4] turn those to numpy, transpose them vertically
            startI_inShingle = numpy.array(shingle_overlaps[0]) 
            startJ_inShingle = numpy.array(shingle_overlaps[1]) 
            #find number of overlaps
            num_overlaps = numpy.size(startI_inShingle)
            if (num_overlaps == 1 and startI_inShingle == startJ_inShingle):
                sint_lst = numpy.column_stack([startI_inShingle, startI_inShingle,(startI_inShingle + (full_bw - 1)),startJ_inShingle,(startJ_inShingle + (full_bw - 1)), full_bw])
                sint_all = numpy.vstack([sint_all, sint_lst])

        # In[ ]:


            elif num_overlaps>0:
                    #Since you are checking the overlaps you need to cut these
                    #intervals into pieces: left, right, and middle. NOTE: the
                    #middle interval may NOT exist
                # Vector of 1's that is the length of the number of
                # overlapping intervals. This is used a lot. 
                ones_no = numpy.ones(num_overlaps);
            

# In[ ]:


    #2a) Left Overlap
                #remain consistent with being matlab -1
                K = startJ_inShingle - startI_inShingle
                sint_lst = numpy.column_stack([startI_inShingle, (startJ_inShingle - ones_no), startJ_inShingle, (startJ_inShingle + K - ones_no), K])
                #https://www.mathworks.com/matlabcentral/answers/72537-what-does-a-tilde-inside-square-brackets-mean
                #returns list of indexes of sorted list
                Is = numpy.argsort(K)
                #turn array vertical
                Is.reshape(numpy.size(Is), 1)
                #extract all columns from row Is
                sint_lst = sint_lst[Is, :]
                #remove the pairs that fall belwo the bandwidth threshold
                if numpy.size(numpy.nonzero(sint_lst[:,4] > thresh_bw)) != 0:
                    #only if there are entries to find
                    cut_s = numpy.amin(numpy.nonzero(sint_lst[:,4] > thresh_bw))
                    sint_lst = sint_lst[cut_s:sint_lst.size(),:]
                    sint_all = numpy.vstack([sint_all, sint_lst])

                #2b right overlap
                EIo = startI_inShingle + (full_bw)
                EJo = startJ_inShingle + (full_bw)
                eint_lst = numpy.column_stack([(EIo + ones_no - K), EIo, (EIo + ones_no), EJo, K])
                Ie = numpy.argsort(K)
                #turn result to column
                Ie.reshape(numpy.size(Ie),1)
                eint_lst = eint_lst[Ie, :]

                if numpy.size(numpy.nonzero(eint_lst[:,4] > thresh_bw)) != 0:
                #remove the pairs that fall below the bandwidth threshold
                    cut_e = numpy.nonzero((eint_lst[:,5] > thresh_bw))
                    eint_lst = eint_lst[cut_e : end, : ]

                #add the new right overlapping intervals to the full list of right overlapping intervals
                sint_all = numpy.vstack([sint_all, sint_lst])

                # 2) Middle Overlap
                #returns logical 0 or 1 for true or false
                mnds = (EIo - startJ_inShingle - K + ones_no) > 0
                #for each logical operator convert to 0 or 1
                for operator in mnds:
                    if operator is True:
                        operator = 1
                    else:
                        operator = 0
                print(mnds)
                SIm = startJ_inShingle*(mnds)
                EIm = (EIo*(mnds) - K*(mnds))
                SJm = (startJ_inShingle*(mnds) + K*(mnds))
                EJm = EIo*(mnds)
                Km = (EIo*(mnds) - startJ_inShingle*(mnds) - K*(mnds) +ones_no*(mnds))

                if numpy.sum(numpy.sum(mnds)) > 0 : 
                    mint_lst = numpy.column_stack([SIm, EIm, SJm, EJm, Km])
                    Im = numpy.argsort(Km)
                    #turn array to column
                    Im.reshape(numpy.size(Im), 1)
                    mint_lst = mint_lst[Im, :]

                    #Remove the pairs that fall below the bandwidth 
                    #  threshold
                    if numpy.size(numpy.nonzero(mint_lst[:,4] > thresh_bw)) != 0:
                        cut_m = numpy.nonzero((mint_lst[:,4] > thresh_bw))
                        print(cut_m)
                        mint_lst = mint_lst[cut_m:mint_lst.size(), :]

                    #Add the new middle overlapping intervals to the full 
                    #list of middle overlapping intervals
                    mint_all = numpy.vstack(mint_all, mint_lst)
    #endelif
#endelif
      
            


# In[ ]:


#         SDM = stretch_diags(T_DDM, bw)
#         thresh_mat = thresh_mat - SDM

#     if numpy.sum(numpy.sum(thresh_mat)) == 0:
#         break
#     #Combine non-overlapping intervals with the left, right, and middle parts
#     #of the overlapping intervals
#     overlap_lst = numpy.concatenate((sint_all, eint_all, mint_all),axis = 0)
#     all_lst = numpy.concatenate((int_all, overlap_lst), axis = 0)


