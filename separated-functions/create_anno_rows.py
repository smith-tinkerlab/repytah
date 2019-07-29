import numpy as np

def create_anno_rows(k_mat,song_length):
    """
    Turn the k_mat into marked rows with annotation markers for
        the start indices and zeroes otherwise. Check if the 
        proper sequence of annotation markers was given and 
        fix them if necessary.

    Args
    ----
    k_mat: np.array
        List of pairs of repeats of length 1 with annotations 
        marked. The first two columns refer to the first repeat
        of the pair, the second two refer to the second repeat of
        the pair, the fifth column refers to the length of the
        repeats, and the sixth column contains the annotation markers.
                 
   song_length: int
        song length, which is the number of audio shingles
   
    Returns
    ------- 
    pattern_row: np.array
        row that marks where non-overlapping repeats
        occur, marking the annotation markers for the
        start indices and zeroes otherwise.

    k_lst_out: np.array
        list of pairs of repeats of length BAND_WIDTH that
        contain no overlapping repeats with annotations marked.
    """
    # Step 0 Initialize outputs: Start with a vector of all 0's for 
    #       pattern_row and assume that the row has no overlaps 
    pattern_row = np.zeros((1,song_length)).astype(int)
    
    # Step 0a: Find the number of distinct annotations
    anno_lst = k_mat[:,5] # Get the elements of k_mat's fifth column
    anno_max = anno_lst.max(0) # Set the number of max elements in each column
    
    # Step 1: Loop over the annotations
    for a in range(1,anno_max+1):
        ands = (anno_lst == a) # Check if anno_lst is equal to a 
        
        # Combine rows into a single matrix
        bind_rows = [k_mat[ands,0],k_mat[ands,2]]
        start_inds = np.concatenate(bind_rows)
        pattern_row[0,start_inds-1] = a
    
    # Step 2: Check that in fact each annotation has a repeat associated to it
    inds_markers = np.unique(pattern_row)

    # If any of inds_markers[i] == 0, then delete this index
    if np.any(inds_markers == 0):
        inds_markers = np.delete(inds_markers,0)

    if inds_markers.size > 0:
        for na in range (1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in pattern_row
                temp_anno = (pattern_row == IM)
                pattern_row = pattern_row - (IM * temp_anno) + (na * temp_anno)
    
    # Edit the annotations to match the annotations in pattern_row
    if k_mat.size > 0:
        k_lst_out = np.unique(k_mat, axis=0)
        for na in range (1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotaions in k_lst_out
                kmat_temp_anno = (k_lst_out[:,5] == IM)
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + (na*kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    output = (pattern_row,k_lst_out)
    
    return output
