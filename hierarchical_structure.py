import numpy as np
def hierarchical_structure(matrix_no,key_no,sn):
    """
     Distills the repeats encoded in MATRIX_NO (and KEY_NO) to 
        the essential structure components and then builds the
        hierarchical representation

    args 
    -----
        matrix_NO: np.array[int]
            binary matrix with 1's where repeats begin and 0's
            otherwise
        key_NO: np.array[int]
            vector containing the lengths of the repeats encoded in matrix_NO
        sn: int
            song length, which is the number of audio shingles

    returns 
    -----
        full_visualization: np.array[int] 
            binary matrix representation for
            full_matrix_NO with blocks of 1's equal to
            the length's prescribed in full_key
        full_key: np.array[int]
            vector containing the lengths of the hierarchical
            structure encoded in full_matrix_NO
        full_matrix_NO: np.array[int]
            binary matrix with 1's where hierarchical 
            structure begins and 0's otherwise
        full_anno_lst: np.array[int]
            vector containing the annotation markers of the 
            hierarchical structure encoded in each row of
            full_matrix_NO
    """
    breakup_tuple = breakup_overlaps_by_intersect(matrix_no, key_no, 0)
    PNO = breakup_tuple[0]
    PNO_key = breakup_tuple[1]

    # Get the block representation for PNO, called PNO_BLOCK
    PNO_block = reconstruct_full_block(PNO, PNO_key)

    # Assign a unique number for each row in PNO. We refer these unique numbers
    # COLORS. 
    num_colors = PNO.shape[0]
    num_timesteps = PNO.shape[1]
    color_mat = np.tile([1:num_colors].T, (1, num_timesteps))

    # For each time step in row i that equals 1, change the value at that time
    # step to i
    PNO_color = color_mat * PNO
    PNO_color_vec = PNO_color.sum(axis=0)
    
    # Find where repeats exist in time, paying special attention to the starts
    # and ends of each repeat of an essential structure component
    PNO_block_vec = PNO_block.sum(axis=0) > 0
    one_vec = (PNO_block_vec[0:sn-1] - PNO_block_vec[1:sn])

    # Matlab ~
    if PNO_block_vec[0] == 0:
        one_vec = np.concatenate((1,one_vec),axis=1)
    elif PNO_block_vec[0] == 1:
        one_vec = np.concatenate((0,one_vec),axis=1)
    
    # Matlab line 67 -- what does this mean?
    PNO_color_vec[one_vec == 1] = (num_colors + 1)
    
    # Python ok
    non_zero_inds = np.array([PNO_color_vec > 0])
    num_NZI = non_zero_inds.sum(axis=0)
    PNO_color_inds_only = PNO_color_vec[0,non_zero_inds-1]
    
    zero_inds_short = (PNO_color_inds_only == (num_colors + 1))
    PNO_color_inds_only[0,zero_inds_short-1] = 0
    
    # Matlab line 100
    PNO_IO_mat = np.kron(ones((num_NZI, 1)),PNO_color_inds_only)
    
    # Matlab ~ 
    PNO_IO_mask = (((PNO_IO_mat > 0) + (PNO_IO_mat.conk().transpose() > 0)) == 2)
    symm_PNO_inds_only = (PNO_IO_mat == PNO_IO_mat.conj().transpose())*PNO_IO_mask
                         
    # Python ok
    NZI_lst = lightup_lst_with_thresh_bw_no_remove(symm_PNO_inds_only, [0:num_NZI])
                          
    remove_inds = (NZI_lst[:,0] == NZI_lst[:,2])
    if np.any(remove_inds == True):
        remove_inds = np.array(remove_inds).astype(int)
        remove = np.where(remove_inds == 1)
        NZI_lst = np.delete(NZI_lst,remove,axis=0)
                          
    NZI_lst_anno = find_complete_list_anno_only(NZI_lst, num_NZI)

    # Matlab line 132
    output_tuple = remove_overlaps(NZI_lst_anno, num_NZI)
    (NZI_matrix_no,NZI_key_no) = output_tuple[1:3]
                          
    NZI_pattern_block = reconstruct_full_block(NZI_matrix_no, NZI_key_no)

    # Matlab
    nzi_rows = NZI_pattern_block.shape[0]
                          
    pattern_starts = (non_zero_inds).nonzero()

    # Matlab
    pattern_ends = np.concatenate(((pattern_starts(1:) - 1), sn)) # is this suppose to be -2 instead of -1? 
    pattern_lengths = np.array((pattern_ends - pattern_starts + 1)) # is this suppose to be 0 instead of -1?
                
    # Python ok
    full_visualization = np.zeros((nzi_rows, sn))
    full_matrix_no = np.zeros((nzi_rows, sn))
                          
    for i in range(1,num_NZI+1):
        # Matlab - fix
        full_visualization(:,[pattern_starts(i):pattern_ends(i)]) = repmat(NZI_pattern_block(:,i),1,pattern_lengths(i))
        full_matrix_no(:,pattern_starts(i)) = NZI_matrix_no(:,i)
    
    # Python ok
    full_key = np.zeros((nzi_rows,1))
    # Matlab
    find_key_mat = full_visualization + full_matrix_no
                          
    
    for i in range(1,nzi_rows+1):
        one_start = np.amin((find_key_mat(i,:) == 2).nonzero())
        # Matlab
        temp_row = find_key_mat[i,:]

        # Python ok
        temp_row[0:one_start] = 1
        find_zero = np.amin((temp_row == 0).nonzero())

        if find_zero.size == 0:
            find_zero = sn + 1

        find_two = np.amin((temp_row == 2).nonzero())
        if find_two.size == 0:
            find_two = sn + 1

        # Matlab - maybe use np.amin? 
        one_end = min(find_zero,find_two);
        full_key[i] = one_end - one_start;
    
    full_key = np.sort(full_key)
    full_key_inds = np.argsort(full_key)

    full_visualization = full_visualization[full_key_inds,:]
    # Matlab
    full_matrix_no = full_matrix_no[full_key_inds,:]
                          
    # Matlab
                          #might need to change to for loop
    inds_remove = (np.sum(full_matrix_no,1) <= 1)

    np.delete(full_key, inds_remove, axis = 0)

    np.delete(full_matrix_no,[inds_remove,:], axis = 0)

    np.delete(full_visualization, [inds_remove,:], axis = 0)
    full_anno_lst = get_annotation_lst(full_key)
                          
    output = (full_visualization,full_key,full_matrix_no,full_anno_lst)
    
    return output
