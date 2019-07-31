import numpy as np

def hierarchical_structure(matrix_no,key_no,sn):
    """
     Distills the repeats encoded in MATRIX_NO (and KEY_NO) to 
        the essential structure components and then builds the
        hierarchical representation
    
    Args 
    -----
        matrix_NO: np.array[int]
            binary matrix with 1's where repeats begin and 0's
            otherwise
        key_NO: np.array[int]
            vector containing the lengths of the repeats encoded in matrix_NO
        sn: int
            song length, which is the number of audio shingles
    
    Returns 
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
    
    # Using PNO and PNO_KEY, we build a vector that tells us the order of the
    # repeats of the essential structure components.
    PNO = breakup_tuple[0]
    PNO_key = breakup_tuple[1]

    # Get the block representation for PNO, called PNO_BLOCK
    PNO_block = reconstruct_full_block(PNO, PNO_key)

    # Assign a unique (nonzero) number for each row in PNO. We refer these 
    # unique numbers COLORS. 
    num_colors = PNO.shape[0]
    num_timesteps = PNO.shape[1]
    
    # Create unique color identifier for num_colors
    color_lst = np.arange(1, num_colors+1)
    
    # Turn it into a column
    color_lst = color_lst.reshape(np.size(color_lst),1)
    color_mat = np.tile(color_lst, (1, num_timesteps))

    # For each time step in row i that equals 1, change the value at that time
    # step to i
    PNO_color = color_mat * PNO
    PNO_color_vec = PNO_color.sum(axis=0)
    
    # Find where repeats exist in time, paying special attention to the starts
    # and ends of each repeat of an essential structure component
    # take sums down columns --- conv to logical
    PNO_block_vec = ( np.sum(PNO_block, axis = 0) ) > 0
    PNO_block_vec = PNO_block_vec.astype(np.float32)

    one_vec = (PNO_block_vec[0:sn-1] - PNO_block_vec[1:sn])
    
    # Find all the blocks of consecutive time steps that are not contained in
    # any of the essential structure components. We call these blocks zero
    # blocks. 
    # Shift PNO_BLOCK_VEC so that the zero blocks are marked at the correct
    # time steps with 1's
    if PNO_block_vec[0] == 0 :
        one_vec = np.insert(one_vec, 1, 1)
    elif PNO_block_vec[0] == 1:
        one_vec = np.insert(one_vec, 1, 0)

    # Assign one new unique number to all the zero blocks
    PNO_color_vec[one_vec == 1] = (num_colors + 1)
    
    # We are only concerned with the order that repeats of the essential
    # structure components occur in. So we create a vector that only contains
    # the starting indices for each repeat of the essential structure
    # components.
    
    # We isolate the starting index of each repeat of the essential structure
    # components and save a binary vector with 1 at a time step if a repeat of
    # any essential structure component occurs there
    # non_zero_inds = PNO_color_vec > 0
    num_NZI = non_zero_inds.sum(axis=0)

    PNO_color_inds_only = PNO_color_vec[non_zero_inds-1]
    
    # For indices that signals the start of a zero block, turn those indices
    # back to 0
    zero_inds_short = (PNO_color_inds_only == (num_colors + 1))
    PNO_color_inds_only[zero_inds_short-1] = 0

    # Create a binary matrix SYMM_PNO_INDS_ONLY such that the (i,j) entry is 1
    # if the following three conditions are true: 
    #     1) a repeat of an essential structure component is the i-th thing in
    #        the ordering
    #     2) a repeat of an essential structure component is the j-th thing in 
    #        the ordering 
    #     3) the repeat occurring in the i-th place of the ordering and the 
    #        one occuring in the j-th place of the ordering are repeats of the
    #        same essential structure component. 
    
    # If any of the above conditions are not true, then the (i,j) entry of
    # SYMM_PNO_INDS_ONLY is 0.
    
    # Turn our pattern row into a square matrix by stacking that row the
    # number of times equal to the columns in that row    
    PNO_IO_mat = np.tile(PNO_color_inds_only,(num_NZI, 1))
    PNO_IO_mat = PNO_IO_mat.astype(np.float32)

    PNO_IO_mask = ((PNO_IO_mat > 0).astype(np.float32) + \
                   (PNO_IO_mat.transpose() > 0).astype(np.float32)) == 2
    symm_PNO_inds_only = (PNO_IO_mat.astype(np.float32) == \
                          PNO_IO_mat.transpose().astype(np.float32)) * \
                          PNO_IO_mask

    # Extract all the diagonals in SYMM_PNO_INDS_ONLY and get pairs of 
    # repeated sublists in the order that repeats of essential structure
    # components.
    
    # These pairs of repeated sublists are the basis of our hierarchical
    # representation.
    NZI_lst = find_all_repeats(symm_PNO_inds_only, [0:num_NZI])                 
    remove_inds = (NZI_lst[:,0] == NZI_lst[:,2])
    
    # Remove any pairs of repeats that are two copies of the same repeat (i.e.
    # a pair (A,B) where A == B)
    if np.any(remove_inds == True):
        remove_inds = np.array(remove_inds).astype(int)
        remove = np.where(remove_inds == 1)
        NZI_lst = np.delete(NZI_lst,remove,axis=0)
        
    # Add the annotation markers to the pairs in NZI_LST
    NZI_lst_anno = find_complete_list_anno_only(NZI_lst, num_NZI)

    output_tuple = remove_overlaps(NZI_lst_anno, num_NZI)
    (NZI_matrix_no,NZI_key_no) = output_tuple[1:3]
                          
    NZI_pattern_block = reconstruct_full_block(NZI_matrix_no, NZI_key_no)

    nzi_rows = NZI_pattern_block.shape[0]
    
    # Find where all blocks start and end
    pattern_starts = np.nonzero(non_zero_inds)[0]

    pattern_ends = np.array([pattern_starts[1: ] - 1]) 
    pattern_ends = np.insert(pattern_ends,np.shape(pattern_ends)[1], sn-1)
    pattern_lengths = np.array(pattern_ends - pattern_starts+1) 

    full_visualization = np.zeros((nzi_rows, sn))
    full_matrix_no = np.zeros((nzi_rows, sn))       

    
    for i in range(0,num_NZI):
        repeated_sect = \
        NZI_pattern_block[:,i].reshape(np.shape(NZI_pattern_block)[0], 1)
        full_visualization[:,pattern_starts[i]:pattern_ends[i]+1] = \
        np.tile(repeated_sect,(1,pattern_lengths[i]))
        full_matrix_no[:,pattern_starts[i]] = NZI_matrix_no[:,i]
    
    # Get FULL_KEY, the matching bandwidth key for FULL_MATRIX_NO
    full_key = np.zeros((nzi_rows,1))
    find_key_mat = full_visualization + full_matrix_no
    
    for i in range(0,nzi_rows):
        one_start = np.where(find_key_mat[i,:] == 2)[0][0]
        temp_row = find_key_mat[i,:]
        temp_row[0:one_start+1] = 1
        find_zero = np.where(temp_row == 0)[0][0]

        if np.size(find_zero) == 0:
            find_zero = sn

        find_two = np.where(temp_row == 2)[0][0]
        if np.size(find_two) == 0:
            find_two = sn

        one_end = np.minimum(find_zero,find_two);
        full_key[i] = one_end - one_start;
      
    full_key_inds = np.argsort(full_key, axis = 0)
    
    # Switch to row
    full_key_inds = full_key_inds[:,0]
    full_key = np.sort(full_key, axis = 0)
    full_visualization = full_visualization[full_key_inds,:]
    full_matrix_no = full_matrix_no[full_key_inds,:]
                        
    # Remove rows of our hierarchical representation that contain only one
    # repeat        
    inds_remove = np.where(np.sum(full_matrix_no,1) <= 1)
    inds_remove = np.array([1])
    full_key = np.delete(full_key, inds_remove, axis = 0)

    full_matrix_no = np.delete(full_matrix_no, inds_remove, axis = 0)
    full_visualization = np.delete(full_visualization, inds_remove, axis = 0)

    full_anno_lst = get_annotation_lst(full_key)
                          
    output = (full_visualization,full_key,full_matrix_no,full_anno_lst)
    
    return output
