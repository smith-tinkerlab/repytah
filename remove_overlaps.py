 """
 Removes any pairs of repeat length and specific
 annotation marker where there exists at least one pair of repeats that do
 overlap in time.

args
-----
    input_mat np.array[int]:
         List of pairs of repeats with annotations marked. The
         first two columns refer to the first repeat or the
         pair, the second two refer to the second repeat of
         the pair, the fifth column refers to the length of
         the repeats, and the sixth column contains the
         annotation markers.
    song_length int: 
         the number of audio shingles
 
returns
-----
    lst_no_overlaps np.array[int]:
        List of pairs of repeats with annotations
        marked. All the repeats of a given length and
        with a specific annotation marker do not
        overlap in time.
    matrix_no_overlaps np.array[int]:
        Matrix representation of LST_NO_OVERLAPS
        with one row for each group of repeats
    key_no_overlaps np.array[int]:
        Vector containing the lengths of the repeats
        encoded in each row of matrix_no_overlaps
    annotations_no_overlaps np.array[int]:
        Vector containing the annotation
        markers of the repeats encoded in each
        row of matrix_no_overlaps
    all_overlap_lst List of pairs of repeats with annotations
                            marked removed from INPUT_MAT. For each pair
                            of repeat length and specific annotation
                            marker, there exist at least one pair of
                            repeats that do overlap in time.
"""
import numpy as np
def remove_overlaps(input_mat, song_length):    
    #same list with repetitions removed
    bw_vec = np.unique(input_mat[:,4])
    #convert L to python list of np arrays

    L = []
    for i in range(0,(np.shape(input_mat)[0])-1):
        L.append(np.array(input_mat[i,:]))
        



    #sort list ascending, then reverse it
    bw_vec = np.sort(bw_vec)
    bw_vec = bw_vec[::-1]


    mat_NO = []
    key_NO = []
    anno_NO = []
    all_overlap_lst = []
    #while bw_vec still has entries
    while np.size(bw_vec) != 0:
        bw_lst = []
        bw = bw_vec[0]
        #Extract pairs of repeats of length BW from the list of pairs of
        #repeats with annotation markers
        #create bw_lst
        i = 0              
        while i < len(L):
            line = L[i][4]
            if line == bw:
                bw_lst.append(line)
                L[i] = np.array([])
            i=i+1
        #endWhile
        
    #remove blanked entries from L (appended to bw_lst)

        #doesn't like elem wise comparison when right operand numpy array
        L = list(filter(lambda L: L.tolist() != [], L))
        if bw > 1:
    #         Use LIGHTUP_PATTERN_ROW_GB to do the following three things:
    #         ONE: Turn the BW_LST into marked rows with annotation markers for 
    #             the start indices and 0's otherwise 
    #         TWO: After removing the annotations that have overlaps, output
    #              BW_LST_OUT which only contains rows that have no overlaps
    #         THREE: The annotations that have overlaps get removed from 
    #                BW_LST_OUT and gets added to ALL_OVERLAP_LST
    
    #momentary commmends!!!!
            tuple_of_outputs = lightup_pattern_row_gb(bw_lst, song_length, bw)
            
            pattern_row = tuple_of_outputs[0]
            bw_lst_out = tuple_of_outputs[1]
            overlap_lst = tuple_of_outputs[2]


            #convert the numpy arrays to lists of 1d numpy arrays
            bw_lst_out_py = []
            for i in range(0,(np.shape(bw_lst_out)[0])-1):
                bw_lst_out_py.append(np.array(input_mat[i,:]))

            overlap_lst_py = []
            for i in range(0,(np.shape(overlap_lst)[0])-1):
                overlap_lst_py.append(np.array(input_mat[i,:]))

            #if there are lines to add
            if len(overlap_lst_py) != 0:
                #add them               
                all_overlap_lst.extend(overlap_lst_py)
        else:
            # Similar to the IF case -- 
            # Use LIGHTUP_PATTERN_ROW_BW_1 to do the following two things:
            # ONE: Turn the BW_LST into marked rows with annotation markers for 
            #      the start indices and 0's otherwise 
            # TWO: In this case, there are no overlaps. Then BW_LST_OUT is just
            #      BW_LST. Also in this case, THREE from above does not exist
            tuple_of_outputs = lightup_pattern_row_gb_1(bw_lst, song_length)
            pattern_row =  tuple_of_outputs[0]
            bw_lst_out_orig =  tuple_of_outputs[1]
            
            #convert the numpy arrays to lists of 1d numpy arrays
            bw_lst_out_py = []
            for i in range(0,(np.shape(bw_lst_out)[0])-1):
                bw_lst_out_py.append(np.array(input_mat[i,:]))

            overlap_lst_py = []
            for i in range(0,(np.shape(overlap_lst)[0])-1):
                overlap_lst_py.append(np.array(input_mat[i,:]))

        if np.max(np.max(pattern_row)) > 0:
            # Separate ALL annotations. In this step, we expand a row into a
            # matrix, so that there is one group of repeats per row.
            
            tuple_of_outputs = separate_all_annotations(bw_lst_out, song_length, bw, pattern_row)
            pattern_mat = tuple_of_outputs[0]
            pattern_key = tuple_of_outputs[1]
            anno_temp_lst = tuple_of_outputs[2]


            #convert the numpy arrays to lists of 1d numpy arrays
            pattern_mat_py = []
            for i in range(0,(np.shape(pattern_mat)[0])-1):
                pattern_mat_py.append(np.array(pattern_mat[i,:]))

            pattern_key_py = []
            for i in range(0,(np.shape(pattern_key)[0])-1):
                pattern_key_py.append(np.array(pattern_key[i,:]))


            anno_temp_lst_py = []
            for i in range(0,(np.shape(anno_temp_lst)[0])-1):
                anno_temp_lst_py.append(np.array(anno_temp_lst[i,:]))


        else:
            pattern_mat = []
            pattern_key = []

        
        if np.sum(np.sum(pattern_mat)) > 0:
            #if there are lines to add, add them
            if np.shape(mat_NO)[0] != 0:
                mat_NO.append(pattern_mat)
            if np.shape(key_NO)[0] != 0:
                key_NO.append(pattern_key)
            if np.shape(anno_NO)[0] != 0:
                anno_NO.append(anno_temp_lst)


        #add to L
        L.append(bw_lst_out_py)
        #sort list by 5th column
        #create dict to re-sort L
        re_sort_L = {}
        for i in range(0, len(L)-1):
            #get 5th column values into list of tuples
            #key = index, value = value
            re_sort_L[i] = (L[i])[4]
        #convt to list of tuples and sort
        re_sort_L = re_sort_L.items()
        #sort that dict by values  
        re_sort_L = sorted(re_sort_L, key=lambda re_sort_L: re_sort_L[1])

        
        sorted_inds = [x[0] for x in re_sort_L]
        #sort L according to sorted indexes
        L = [L for sorted_inds, L in sorted(zip(sorted_inds, L))]

        #will just use a np array here
        np_mat_L = np.array(L)
        bw_vec = np.unique(np_mat_L[:,4])
        #sort list ascending, then reverse it
        bw_vec = np.sort(bw_vec)
        bw_vec = bw_vec[::-1]
    #     #remove entries that fall below the bandwidth threshold
        cut_index = 0

        for value in bw_vec:
        #if the value is above the bandwidth 
            if value < bw:
                cut_index = cut_index+1
        #endfor
        bw_vec = bw_vec[cut_index:np.shape(bw_vec)[0]]

    #endWhile

    #     # Set the outputs
    lst_no_overlaps = np.array(L)
    #turn key_NO, mat_NO, and KEY_NO to numpy lists

    key_NO = list(filter(lambda key_NO: key_NO.tolist() != [], key_NO))
    mat_NO = list(filter(lambda mat_NO: mat_NO.tolist() != [], mat_NO))
    annno_NO = list(filter(lambda anno_NO: anno_NO.tolist() != [], anno_NO))

    if len(key_NO) !=0:
        key_NO = np.concatenate(key_NO)
    else:
        key_NO = np.array([])
        
    if len(mat_NO) !=0:
        mat_NO = np.concatenate(mat_NO)
    else:
        mat_NO = np.array([])
        
    if len(anno_NO) !=0:
        anno_NO = np.concatenate(anno_NO)
    else:
        anno_NO = np.array([])

        #conv to np array
    all_overlap_lst = np.array(all_overlap_lst)
    if np.shape(all_overlap_lst)[0] != 0:
        overlap_inds = np.argsort(all_overlap_lst[:,4])
        all_overlap_lst = all_overlap_lst[overlap_inds, :]
    #endif
    key_NO = np.sort(key_NO)
    mat_inds = np.argsort(key_NO)
    if np.shape(mat_NO)[0] != 0:
        matrix_no_overlaps = mat_NO[mat_inds,:]
    else:
        matrix_no_overlaps = mat_NO
    key_no_overlaps = key_NO
    if np.shape(anno_NO)[0] != 0:
        annotations_no_overlaps = mat_NO[mat_inds,:]
    else:
        annotations_no_overlaps = mat_NO
        
    #compile final outputs to a tuple
    output_tuple = (lst_no_overlaps, matrix_no_overlaps, key_no_overlaps, annotations_no_overlaps, all_overlap_lst)
    return output_tuple
#end remove_overlaps
