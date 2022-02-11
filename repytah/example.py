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

    * visualize_all_lst
        Produces a visualization to highlight all pairs of repeats
        in the Mazurka data set.

    * visualize_complete_lst
        Produces a visualization to highlight all pairs of smaller repeats
        that are contained in larger diagonals in the Mazurka data set.

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
    Produces a visualization to highlight all pairs of repeats
    in the Mazurka data set.

    Args
    ----
    all_lst : np.array
        Pairs of repeats that correspond to diagonals in thresh_mat.

    thresh_dist_mat : np.ndarray
        Thresholded dissimilarity matrix that we extract diagonals from.
    
    
    Returns
    -------
    none : A visualization of all pairs of repeat is produced.

    """

    # Produce a visualization of the SDM
    SDM = plt.imshow(thresh_dist_mat, cmap="Greys")

    # For [247, 250, 271, 274, 4] in all_lst
    x0 = [247, 250]
    y0 = [271, 274]
    plt.plot(x0, y0, color = "darkorange")

    # For [124, 156, 292, 324, 33] in all_lst
    x0 = [124, 156]
    y0 = [292, 324]
    plt.plot(x0, y0, color = "blue")

    # For [2, 36, 290, 324, 35] in all_lst
    x1 = [2, 36]
    y1 = [290, 324]
    plt.plot(x1, y1, color = "red")

    # For [3, 38, 123, 158, 36] in all_lst
    x2 = [3, 38]
    y2 = [123, 158]
    plt.plot(x2, y2, color = "fuchsia")

    # For [3, 39, 195, 231, 37] in all_lst
    x2 = [3, 39]
    y2 = [195, 231]
    plt.plot(x2, y2, color = "green")

    # For [1, 38, 49, 86, 38] in all_lst
    x2 = [1, 38]
    y2 = [49, 86]
    plt.plot(x2, y2, color = "deepskyblue")

    # For [51, 122, 123, 194, 72] in all_lst
    x2 = [51, 122]
    y2 = [123, 194]
    plt.plot(x2, y2, color = "teal")

    plt.show()

def visualize_complete_lst(all_lst, complete_lst, thresh_dist_mat):
    """
    Produces a visualization to highlight all pairs of smaller repeats
    that are contained in larger diagonals in the Mazurka data set.

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

    # Breaking down [124, 156, 292, 324, 33]
    # For [124, 145, 292, 313, 22, 1] in complete_lst
    x = [124, 145]
    y = [292, 313]
    plt.plot(x, y, color = "red")

    # For [146, 146, 314, 314, 1, 2] in complete_lst
    x = [146, 146]
    y = [314, 314]
    plt.plot(x, y, color = "yellow")

    # For [147, 156, 315, 324, 10, 1] in complete_lst
    x = [147, 156]
    y = [315, 324]
    plt.plot(x, y, color = "blue")

    # Breaking down [2, 36, 290, 324, 35]
    # For  [2, 3, 290, 291, 2, 2] in complete_lst
    x = [2, 3]
    y = [290, 291]
    plt.plot(x, y, color = "red")

    # For [4, 36, 292, 324, 33, 1] in complete_lst
    x = [4, 36]
    y = [292, 324]
    plt.plot(x, y, color = "blue")

    # Breaking down [3, 38, 123, 158, 36]
    # For [3, 3, 123, 123, 1, 3] in complete_lst
    x = [3, 3]
    y = [123, 123]
    plt.plot(x, y, color = "yellow")

    # For [4, 36, 124, 156, 33, 1] in complete_lst
    x = [4, 36]
    y = [124, 156]
    plt.plot(x, y, color = "red")

    # For [37, 38, 157, 158, 2, 3] in complete_lst
    x = [37, 38]
    y = [157, 158]
    plt.plot(x, y, color = "blue")

    # Breaking down [3, 39, 195, 231, 37]
    # For [3, 3, 195, 195, 1, 3] in complete_lst
    x = [3, 3]
    y = [195, 195]
    plt.plot(x, y, color = "yellow")

    # For [4, 36, 196, 228, 33, 1] in complete_lst
    x = [4, 36]
    y = [196, 228]
    plt.plot(x, y, color = "red")

    # For [37, 39, 229, 231, 3, 1] in complete_lst
    x = [37, 39]
    y = [229, 231]
    plt.plot(x, y, color = "blue")

    # Breaking down [1, 38, 49, 86, 38]
    # For [1, 1, 49, 49, 1, 1] in complete_lst
    x = [1, 1]
    y = [49, 49]
    plt.plot(x, y, color = "yellow")

    # For [2, 36, 50, 84, 35, 1] in complete_lst
    x = [2, 36]
    y = [50, 84]
    plt.plot(x, y, color = "red")

    # For [37, 38, 85, 86, 2, 3] in complete_lst
    x = [37, 38]
    y = [85, 86]
    plt.plot(x, y, color = "blue")

    # Breaking down [51, 122, 123, 194, 72]
    # For [51, 73, 123, 145, 23, 1] in complete_lst
    x = [51, 73]
    y = [123, 145]
    plt.plot(x, y, color = "red")

    # For [74, 74, 146, 146, 1, 2] in complete_lst
    x = [74, 74]
    y = [146, 146]
    plt.plot(x, y, color = "yellow")

    # For [75, 122, 147, 194, 48, 1] in complete_lst
    x = [75, 122]
    y = [147, 194]
    plt.plot(x, y, color = "blue")

    plt.show()

