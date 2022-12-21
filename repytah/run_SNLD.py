"""
run_SE.py

A python module to create start end diagrams from aligned hierarchies and
use them to compare songs to each other.


The module contains the following functions:

    * get_SNLDs
        Constructs start normalized length diagrams from chroma vectors or
        aligned hierarchies

    * get_SNLD_directory
        Constructs start normalized length diagrams for a folder of chroma
        vectors or aligned hierarchies.

    * get_dist_mat
        Compares pairs of songs to each other using distance metrics.

Original Author: Melissa McGuirl
"""


import os
import glob
import pkg_resources
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat

import persim
from example import csv_to_aligned_hierarchies, save_to_mat


class SNLD_directory:
    """
    SNLD class to hold directory names, song labels, and SNLDs

    Attributes:
        labels (list):
            contains all song names
        SNLDS (list):
            contains all Start Normalized Length Diagrams
        className (list):
            a list that contains directory names
    """

    labels = []
    SNLDs = []
    className = []

    def __init__(self, name):
        self.name = name

    def add_class(self, class_name):
        self.className.append(class_name)

    def add_labels(self, label):
        self.labels.append(label)

    def add_SNLDs(self, SNLD):
        self.SNLDs.append(SNLD)


def get_AH(filepath, num_fv_per_shingle, thresh, vis=False):
    """

    Wrapper function to retrieve Aligned Hierarchies

    Args:
        filepath (str):
            Name of file to be processed. Contains features across time
            steps to be analyzed, for example chroma features
        num_fv_per_shingle (int):
            Number of feature vectors per shingle. Provides "context" of each
            individual time step, so that for notes CDE if
            num_fv_per_shingle = 2, shingles would be CD, DE.
        thresh (int):
            Maximum threshold value.
        vis (bool):
            Shows visualizations if True. Default is False.

    Returns:
        AH_dict (dictionary):
            Aligned hierarchies with keys full_key, full_mat_no_overlaps,
            partial_reps, partial_key, partial_widths, partial_num_blocks, and
            num_partials

    """

    # convert to absolute filepath
    if not os.path.isabs(filepath):
        filepath = pkg_resources.resource_stream(__name__, filepath)

    # read in chroma vector file and convert to AH
    file_in = pd.read_csv(filepath, header=None).to_numpy()
    AH_dict = csv_to_aligned_hierarchies(file_in, num_fv_per_shingle, thresh, vis)

    return AH_dict


def get_SNLDs(filepath, num_fv_per_shingle, thresh, norm_type, alpha, isChroma=True, save=False):
    """

    Extracts Start Normalized Length Diagrams from Aligned Hierarchies

    Args:
        filepath (str):
            Name of file to be processed. Can be a chroma vector file in
            .csv form, or an AH file in .mat or .csv form.
        num_fv_per_shingle (int):
            Number of feature vectors per shingle. Provides "context" of each
            individual time step, so that for notes CDE if
            num_fv_per_shingle = 2, shingles would be CD, DE.
        thresh (float):
            Maximum threshold value for converting chroma vectors to AHs.
        norm_type (str):
            "none" for no normalization, "std" for standard normalization, or
            "cheb" for Chebyshev normalization
        alpha (int):
            Scaling parameter for normalization, alpha=1 indicates no scaling.
        isChroma (boolean):
            True if starting from chroma vector files, False if starting from
            previously saved Aligned Hierarchies files. Default is True.
        save (boolean):
            True if you want to save the intermediate Aligned Hierarchies.
            Default is False.

    Returns:
        SNL_diagram (list):
            Start Normalized Length Diagram.

    """
    AH_dict = None
    # start from chroma vectors
    if isChroma:
        AH_dict = get_AH(filepath, num_fv_per_shingle, thresh)
        # save the AH files in a subfolder as .mat files
        if save:
            parts = filepath.split(os.sep)
            file_base = os.sep.join(parts[:-3])
            file_out = os.path.join(file_base,
                                    f"AH_Thresh{int(thresh * 100):02}"
                                    f"_ShingleNumber{num_fv_per_shingle}",
                                    parts[-2],
                                    f"{parts[-1][:-4]}.mat")

            # create folder
            # might want to optimize this, probably save into file_base_dir
            # or smth
            if not os.path.exists(os.path.split(file_out)[0]):
                os.makedirs(os.path.split(file_out)[0])

            save_to_mat(file_out, AH_dict)

    # start from previously saved AHs
    else:
        extension = os.path.splitext(filepath)[1]

        # force file path to be absolute
        if not os.path.isabs(filepath):
            filepath = pkg_resources.resource_stream(__name__, filepath)

        if extension == ".mat":
            AH_dict = loadmat(filepath, squeeze_me=True)
        elif extension == ".csv":
            # read in file
            AH_df = pd.read_csv(filepath)

            # only create dictionary for keys needed
            full_key = []
            full_mat_no_overlaps = []
            # iterate over dataframe to get key values
            for i, rows in AH_df.iterrows():
                # turn string of chars into numpy array of integers
                temp_full_key_list = np.array([int(rows['full_key'])])
                temp_mat = np.array([int(s) for s in rows['full_mat_no_overlaps'].split(' ')])

                full_key.append(temp_full_key_list)
                full_mat_no_overlaps.append(temp_mat)

            AH_dict = {'full_key': np.array(full_key), 'full_mat_no_overlaps': np.array(full_mat_no_overlaps)}
        else:
            warnings.warn("Only .mat and .csv file types supported for Aligned Hierarchies", RuntimeWarning)



    key = AH_dict['full_key']
    if type(key) == int:
        key = [key]  # make array to handle single row case
    AH = AH_dict['full_mat_no_overlaps']
    blocks = np.nonzero(AH)
    # when there's single row, python swaps the order
    if len(blocks) > 1:
        rows = blocks[0]
        cols = blocks[1]
    else:
        cols = blocks[0]
        rows = np.zeros(len(cols))

    SNL_diagram = []
    start_norm = 0
    start_min = 10000000000
    for k in range(len(rows)):
        start_temp = int(cols[k]) + 1
        start_norm = max(start_norm, start_temp)
        start_min = min(start_temp, start_min)
        len_temp = key[int(rows[k])]
        SNL_diagram.append((start_temp, len_temp))
    # check to make sure SL diagram is nonempty. If so, add (0,0)
    if SNL_diagram == []:
        SNL_diagram.append((0, 0))
        print('Warning: empty diagram found.')
    else:
        # now normalize
        if norm_type == 'std':
            # norm factor is alpha/max(start times)
            norm_factor = float(alpha) / float(start_norm)
            SNL_diagram = [(norm_factor * s, l) for (s, l) in SNL_diagram]
        elif norm_type == 'cheb':
            r = 0.5
            T = start_norm - start_min
            SNL_diagram = [(float(alpha) * r * (1 - np.cos((s - start_min) * np.pi / T)), l) for (s, l) in SNL_diagram]

    return SNL_diagram


# a function to get SNLD directory
def get_SNLD_directory(IN, num_fv_per_shingle, thresh, norm_type, alpha,
                       isChroma=True, save=False):
    """

    Args:
        IN (str):
            Input file directory.
        num_fv_per_shingle (int):
            Number of feature vectors per shingle for converting chroma vectors
            to Aligned Hierarchies. Provides "context" of each individual time
            step, so that for notes CDE if num_fv_per_shingle = 2, shingles
            would be CD, DE.
        thresh (float):
            Maximum threshold value for converting chroma vectors to AHs.
        norm_type (str):
            "none" for no normalization, "std" for standard normalization, or
            "cheb" for Chebyshev normalization.
        alpha (int):
            Scaling parameter for normalization, alpha=1 indicates no scaling.
        isChroma (boolean):
            True if starting from chroma vector files, False if starting from
            previously saved Aligned Hierarchies files. Default is True.
        save (boolean):
            True if you want to save the intermediate Aligned Hierarchies.
            Default is False.

    Returns:
        SNLD_all (SNLD_directory):
            Directory that contains all SNLDs and corresponding information.

    """

    # define variable to store all SNLDs
    SNLD_all = SNLD_directory('ALL SNLDs ' + str(alpha))

    # check if directory path is absolute
    if not os.path.isabs(IN):
        IN = pkg_resources.resource_filename(__name__, IN)

    # loop through each subdirectory
    subdirs = glob.glob(IN + '/*')
    for subdir in subdirs:
        SNLDs = []
        labels = []
        subdir_name = subdir[len(IN)::]
        if isChroma:
            files = glob.glob(subdir + '/*.csv')
        else:
            files = glob.glob(subdir + '/*.mat')
        # loop through each file per directory, get SNLD and labels
        for file in files:
            SNLDs.append(get_SNLDs(file, num_fv_per_shingle, thresh, norm_type,
                                   alpha, isChroma, save))
            labels.append(file[len(subdir) + 1:-4])
        # store all SNLD info in SNLD_directory class, add to SNLD_all
        SNLD_all.add_class(subdir_name)
        SNLD_all.add_labels(labels)
        SNLD_all.add_SNLDs(SNLDs)

    return SNLD_all


def get_dist_mat(SNLD_all, metric):
    """

    Args:
        SNLD_all (SNLD_directory):
            Directory that contains all SNLDs and corresponding information.
        metric (str):
            'b' for bottleneck distance, 'w' for wasserstein distance.

    Returns:
        D (np.ndarray):
            Distance matrix for all SNLD diagrams in directory.
        labels (list):
            list of all song names.

    """
    # create distance matrix
    N_tot = sum([len(SNLD_all.SNLDs[i]) for i in range(len(SNLD_all.SNLDs))])
    D = np.zeros((N_tot, N_tot))

    SNLD = []
    labels = []

    # combine all SNLDs and labels into their own list
    for i in range(len(SNLD_all.SNLDs)):
        for k in range(len(SNLD_all.SNLDs[i])):
            SNLD.append(SNLD_all.SNLDs[i][k])
            labels.append(SNLD_all.labels[i][k])

    # perform distance metric for every possible pair of SNLD diagrams
    for a in range(N_tot):
        # extract ath diagram
        list_a = []
        for c in range(len(SNLD[a])):
            # ex. reformatting (1.0, [1]) to [1.0, 1]
            temp_a = [SNLD[a][c][0], SNLD[a][c][1][0]]

            list_a.append(temp_a)
        # extract bth diagram
        for b in range(a+1, N_tot):
            list_b = []
            for d in range(len(SNLD[b])):
                temp_b = [SNLD[b][d][0], SNLD[b][d][1][0]]
                list_b.append(temp_b)

            if metric == 'b':
                D[a][b] = persim.bottleneck(list_a, list_b)
            else:
                D[a][b] = persim.wasserstein(list_a, list_b)
            # make symmetric across diagonal
            D[b][a] = D[a][b]

    return D, labels
