import scipy.io as sio
import numpy as np
import pkg_resources
import pandas as pd
import matplotlib.pyplot as plt

from .utilities import create_sdm, find_initial_repeats
from .search import find_complete_list
from .transform import remove_overlaps
from .assemble import hierarchical_structure

"""
example.py

An example module that contains functions to run a complete case of building
aligned hierarchies when a CSV file with extracted features is the input.

The module contains the following functions:
    
    * load_eg_data
        Reads in a csv input file with extracted features.

    * csv_to_aligned_hierarchies
        Example of full aligned hierarchies pathway. 

"""

def load_ex_data(input):
    """
    Reads in a csv input file with extracted features.

    Args
    ----
    input : str
        Name of .csv file to be processed. 
    """

    stream = pkg_resources.resource_stream(__name__, input)
    return pd.read_csv(stream)


def csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle, thresh):
    """
    Example of full aligned hierarchies pathway.
    
    Args
    ----
    file_in : str
        Name of .csv file to be processed. Contains features across time steps 
        to be analyzed, for example chroma features
    
    file_out : str
        Name of file where output will be stored.
    
    num_fv_per_shingle : int
        Number of feature vectors per shingle. Provides "context" of each 
        individual time step, so that for notes CDE if num_fv_per_shingle=2
        shingles would be CD, DE.
        
    num_fv_per_shingle : int
        Number of feature vectors per shingle. Provides "context" of each
        individual time step, so that for notes CDE if num_fv_per_shingle=2
        shingles would be CD, DE.
        
    thresh : int
        Maximum threshold value. Largest length repeated structure to 
        search for.
    
    Returns
    -------
    none : .mat file is saved. Contains variables created for aligned 
        hierarchies. 

    Example
    --------
    ### Run on example file
    >>> file_in = pd.read_csv(os.path.join(os.path.dirname(__file__),
                              "../input.csv")).to_numpy()
    >>> file_out = "hierarchical_out_file.mat"
    >>> num_fv_per_shingle = 12
    >>> thresh = 0.02
    >>> csv_to_aligned_hierarchies(file_in, file_out,
                                   num_fv_per_shingle, thresh)
                                   
    """
    
    # Import file of feature vectors
    fv_mat = file_in
    
    # Get pairwise distance matrix/self dissimilarity matrix using cosine 
    # distance
    self_dissim_mat = create_sdm(fv_mat, num_fv_per_shingle)
    
    # Get thresholded distance matrix
    song_length = self_dissim_mat.shape[0]
    thresh_dist_mat = (self_dissim_mat <= thresh) 
    
    # Extract the diagonals from thresholded distance matrix, saving the 
    # repeat pairs the diagonals represent
    all_lst = find_initial_repeats(thresh_dist_mat, 
                                   np.arange(1, song_length + 1), 0)
    
    # Find smaller repeats which are contained within larger repeats
    complete_lst = find_complete_list(all_lst, song_length)

    # Create the dictionary of output variables
    outdict = {}
    outdict['thresh'] = thresh
    
    if np.size(complete_lst) != 0:
        # Remove groups of repeats that overlap in time
        output_tuple = remove_overlaps(complete_lst, song_length)

        (mat_no_overlaps, key_no_overlaps) = output_tuple[1:3]

        # Distill non-overlapping repeats into essential structure components 
        # and use them to build the hierarchical representation
        output_tuple = hierarchical_structure(mat_no_overlaps, key_no_overlaps,
                                              song_length, True)
        (full_key, full_mat_no_overlaps) = output_tuple[1:3]
        
        outdict['full_key'] = full_key
        outdict['full_mat_no_overlaps'] = full_mat_no_overlaps
        
        # Save list of partial representations containing only the full
        # hierarchical representation for use in comparison code
        outdict['partial_reps'] = [full_mat_no_overlaps]
        outdict['partial_key'] = [full_key]
        outdict['partial_widths'] = song_length
        outdict['partial_num_blocks'] = np.sum(mat_no_overlaps)
        outdict['num_partials'] = 1
        
        # Create the output file
        sio.savemat(file_out, outdict)
        
    else:
        outdict['full_key'] = []
        outdict['full_mat_no_overlaps'] = []
        
        # Save the empty list of partial representations for use in comparison 
        # code
        outdict['partial_reps'] = []
        outdict['partial_key'] = []
        outdict['partial_widths'] = []
        outdict['partial_num_blocks'] = []
        outdict['num_partials'] = 0
        
        # Create the output file
        sio.savemat(file_out, outdict)

def visualize_all_lst(all_lst, thresh_dist_mat):
    """
    Produce a visualization to highlight all pairs of repeats.

    Args
    ----
    all_lst : np.array
        Pairs of repeats that correspond to diagonals in thresh_mat.

    thresh_dist_mat : np.ndarray
        Thresholded dissimilarity matrix that we extract diagonals from.
    
    
    Returns
    -------
    none : A visualization of all pairs of repeated is produced.

    """

    # Produce a visualization of the SDM
    SDM = plt.imshow(thresh_dist_mat, cmap="Greys")
    colors = ["red", "blue", "orange"]

    # Find the length of the largest diagonal so that we don't
    # color-code this particular diagonal
    max_length = all_lst[len(all_lst) - 1][4]

    for pair_of_repeat in all_lst:
        # Obtain repeat length
        repeat_length = pair_of_repeat[4]

        # If repeat length is greater than 1 and not the maximum length,
        # produce the visualization for that pair of repeat
        if repeat_length > 1 and repeat_length != max_length:
            first_repeat = pair_of_repeat[0:2]
            second_repeat = pair_of_repeat[2:4]
            rgb = np.random.rand(3,)
            plt.plot(first_repeat, second_repeat, color = rgb)

    plt.show()

def visualize_complete_lst(all_lst, complete_lst, thresh_dist_mat):
    """
    Produce a visualization to highlight all pairs of smaller repeats
    that are contained in larger diagonals.

    Args
    ----
    complete_lst : np.ndarray 
        List of pairs of repeats with smaller repeats added.

    thresh_dist_mat : np.ndarray
        Thresholded dissimilarity matrix that we extract diagonals from.
    
    
    Returns
    -------
    none : A visualization of all pairs of repeated smaller repeats
           that are contained in larger diagonals is produced.


    """

    # Produce a visualization of the SDM
    SDM = plt.imshow(thresh_dist_mat, cmap="Greys")
    colors = ["red", "blue", "orange", "green", "purple", "yellow"]

    for pair_of_repeat in complete_lst:
        pair_of_repeat_without_anno = pair_of_repeat[0:5]
        # Check if the pair of repeat is in all_lst. If yes then skip the pair
        # of repeat. Otherwise produce a visualization for that repeat
        if not any(np.equal(all_lst, pair_of_repeat_without_anno).all(1)):
            # Obtain the repeat length
            repeat_length = pair_of_repeat[4]

            # If repeat length is greater than 1, produce the visualization 
            # for that pair of repeat
            if repeat_length > 1:
                first_repeat = pair_of_repeat[0:2]
                second_repeat = pair_of_repeat[2:4]
                rgb = np.random.rand(3,)
                plt.plot(first_repeat, second_repeat, color = rgb)

    plt.show()

