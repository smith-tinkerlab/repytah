import numpy as np

def lightup_pattern_row_gb(k_mat,song_length,band_width):
    """
    Turn the k_mat into marked rows with annotation markers for the start
        indices and zeroes otherwise, after removing the annotations that have overlaps, 
        output k_lst_out which only contains rows that have no overlaps,
        the annotations that have overlaps get removed from k_lst_out
        gets added to overlap_lst.
    
    Args
    ----
    k_mat: np.array
        list of pair of repeats with annotations marked
    
    song_length: int
        number of audio shingles
    
    band_width: int
        the length of repeats encoded in k_mat
    
    Returns
    -------
    pattern_row: np.array
        row that marks where non-overlapping repeats occur, 
        marking the annotation markers for the start indices 
        and 0's otherwise
    
    k_lst_out: np.array
        list of pairs of repeats of length band_width that 
        contain no overlapping repeats with annotations
        marked
    
    overlap_lst: np.array
        list of pairs of repeats of length band_width that
        contain overlapping repeats with annotations marked
    """
    # Step 0: Initialize outputs: Start with a vector of all 0's for
    #         pattern_row and assume that the row has no overlaps
    pattern_row = np.zeros((1,song_length)).astype(int)
    overlap_lst = []
    bw = band_width
    
    # Step 0a: Find the number of distinct annotations
    anno_lst = k_mat[:,5] # Get the elements of k_mat's fifth column
    anno_max = anno_lst.max(0) # Max in each column
    
    # Step 1: Loop over the annotations
    for a in range (1,anno_max+1):
        # Step 1a: Add 1's to pattern_row to the time steps where repeats with
        # annotation a begin
        ands = (anno_lst == a) # Check if anno_lst is equal to a
        bind_rows = [k_mat[ands,0],k_mat[ands,2]]
        start_inds = np.concatenate(bind_rows)
        pattern_row[0,start_inds-1] = a

        # Step 2: check annotation by annotation
        good_check = np.zeros((1,song_length)).astype(int) # Start with row of 0's
        good_check[0,start_inds-1] = 1 # Add 1 to all time steps where repeats with annotation a begin
    
        # Using reconstruct_full_block to check for overlaps
        block_check = reconstruct_full_block(good_check,bw)

        # If there are any overlaps, remove the bad annotations from both
        # the pattern_row and from the k_lst_out
        if block_check.max() > 1:
            # Remove the bad annotations from pattern_row
            pattern_row[0,start_inds-1] = 0
    
            # Remove the bad annotations from k_lst_out and add them to overlap_lst
            remove_inds = ands

            temp_add = k_mat[remove_inds,:]
            overlap_lst.append(temp_add)
            
            if np.any(rm_inds == True):
                # Convert the boolean array rm_inds into an array of integers
                remove_inds = np.array(rm_inds).astype(int)
                remove = np.where(rm_inds==1)
                # Delete the row that meets the condition set by remove_inds
                k_mat = np.delete(k_mat,remove,axis=0)
                
            anno_lst = k_mat[:,5]
           
    inds_markers = np.unique(pattern_row)
    # If any of inds_markers[i] is equal to zero, then remove this index
    if np.any(inds_markers == 0):
        inds_markers = np.delete(inds_markers,0)
    
    # If inds_markers is not empty, then execute this if statement
    if inds_markers.size > 0:
        for na in range(1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in pattern_row
                temp_anno = (pattern_row == IM)
                pattern_row = pattern_row - (IM * temp_anno) + (na * temp_anno)
     
    # If k_mat is not empty, then execute this if statement
    if k_mat.size > 0:
        k_lst_out = np.unique(k_mat,axis=0)
        for na in range(1,len(inds_markers)+1):
            IM = inds_markers[na-1]
            if IM > na:
                # Fix the annotations in k_lst_out
                kmat_temp_anno = (k_lst_out[:,5] == IM)
                k_lst_out[:,5] = k_lst_out[:,5] - (IM * kmat_temp_anno) + (na * kmat_temp_anno)
    else:
        k_lst_out = np.array([])
    
    # Edit the annotations in the overlap_lst so that the annotations start
    # with 1 and increase one each time
    if overlap_lst.size > 0:
        overlap_lst = np.unique(overlap_lst,axis=0)
        overlap_lst = add_annotations(overlap_lst,song_length)

    return pattern_row, k_lst_out, overlap_lst
