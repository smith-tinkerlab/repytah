# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

def csv_to_hs(file_in, file_out, num_fv_per_shingle, thresh):
    """
    Example of full aligned hierarchies pathway
    
    Args
    ----
    file_in: str
        name of .csv file to be processed
    
    file_out: str
        name of file where output will be stored
    
    num_fv_per_shingle: int
        number of feature vectors per shingle
        
    thresh: int
        maximum threshold value
    
    Returns
    -------
    none: .mat file is saved.
    """
    # Import file of feature vectors
    with open(file_in, 'r', newline='\n') as f:
        fv_mat = np.loadtxt(f, delimiter = ',')
    
    # Get pairwise distance matrix/self dissimilarity matrix using cosine distance
    self_dissim_mat = create_sdm(fv_mat, num_fv_per_shingle)
    
    # Get thresholded distance matrix
    song_length = self_dissim_mat.shape[0]
    thresh_dist_mat = (self_dissim_mat >= thresh) * np.ones((song_length,song_length))
    
    # Extract diagonals from thresholded distance matrix, saving the repeat pairs
    # the diagonals represent
    all_lst = lightup_lst_with_thresh_bw(thresh_dist_mat, range(1, song_length + 1), 0)
    
    # Find smaller repeats contained within larger repeats
    complete_lst = find_complete_list(all_lst, song_length)
    
    # Create dictionary of output variables
    outdict = {}
    outdict['thresh'] = thresh
    
    if np.size(complete_lst) != 0:
        # Remove groups of repeats that overlap in time
        output_tuple = remove_overlaps(complete_lst, song_length)
        (mat_no_overlaps, key_no_overlaps) = output_tuple[1:3]
        
        # Distill non-overlapping repeats into essential structure components and
        # use them to build the hierarchical representation
        output_tuple = hierarchical_structure(mat_no_overlaps, key_no_overlaps, song_length)
        (full_key, full_mat_no_overlaps) = output_tuple[1:3]
        
        outdict['full_key'] = full_key
        outdict['full_mat_no_overlaps'] = full_mat_no_overlaps
        
        # Save list of partial representations contatining only the full hierarchical
        # representation for use in comparison code
        outdict['partial_reps'] = [full_mat_no_overlaps]
        outdict['partial_key'] = [full_key]
        outdict['partial_widths'] = song_length
        outdict['partial_num_blocks'] = np.sum(mat_no_overlaps)
        outdict['num_partials'] = 1
        
        # Create output file
        sio.savemat(file_out, outdict)
        
    else:
        outdict['full_key'] = []
        outdict['full_mat_no_overlaps'] = []
        
        # Save empty list of partial representations for use in comparison code
        outdict['partial_reps'] = []
        outdict['partial_key'] = []
        outdict['partial_widths'] = []
        outdict['partial_num_blocks'] = []
        outdict['num_partials'] = 0
        
        # Create output file
        sio.savemat(file_out, outdict)
        
    
    

