"""
run_SE.py

A python module to create start end diagrams from aligned hierarchies and
use them to compare songs to each other.


The module contains the following functions:

    * get_SEs
        Constructs start end diagrams from chroma vectors or aligned hierarchies

    * get_SE_directory
        Constructs start end diagrams for a folder of chroma vectors or aligned
        hierarchies.

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


class SE_directory:
    """
    SE class to hold directory names, song labels, and SEs
    """

    labels = []
    SEs = []
    className = []

    def __init__(self, name):
        self.name = name

    def add_class(self, class_name):
        self.className.append(class_name)

    def add_labels(self, label):
        self.labels.append(label)

    def add_SEs(self, SE):
        self.SEs.append(SE)


def get_AH(filepath, num_fv_per_shingle, thresh, vis=False):

    # convert to absolute filepath if not already
    if not os.path.isabs(filepath):
        filepath = pkg_resources.resource_stream(__name__, filepath)

    # read in chroma vector file and convert to AH
    file_in = pd.read_csv(filepath, header=None).to_numpy()
    AH_dict = csv_to_aligned_hierarchies(file_in, num_fv_per_shingle, thresh, vis)

    return AH_dict


def get_SEs(filepath, num_fv_per_shingle, thresh, isChroma=True, save=False):
    """

    Extracts Start End Diagrams from Aligned Hierarchies

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

    # filepath = C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6\\Expanded\\mazurka-50.mat
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

            AH_dict = {'full_key': np.array(full_key), 'full_mat_no_overlaps':np.array(full_mat_no_overlaps)}
        else:
            warnings.warn("Only .mat and .csv file types supported for Aligned Hierarchies", RuntimeWarning)

    key = AH_dict['full_key']
    if type(key) == int:
        key = [key]  # make array to handle single row case
    # first values start at col = 2 for expanded mazurka 50
    AH = AH_dict['full_mat_no_overlaps']
    blocks = np.nonzero(AH)
    # when there's single row, python swaps the order
    if len(blocks) > 1:
        rows = blocks[0]
        cols = blocks[1]
    else:
        cols = blocks[0]
        rows = np.zeros(len(cols))

    SE_diagram = np.empty((0, 2), int)
    # add start and end
    for k in range(len(rows)):
        start_temp = cols[k]  # no need for +1 I think
        len_temp = int(key[rows[k]])
        SE_diagram = np.vstack((SE_diagram, (start_temp, len_temp + start_temp)))

    # check to make sure SL diagram is nonempty. If so, add (0,0)
    if SE_diagram.size == 0:
        SE_diagram = np.vstack((SE_diagram, (0, 0)))
        print('Warning: empty diagram found.', filepath)

    # minimum start temp is 2 for expanded mazurka 50 but 1 for mazurka 51 so
    # i think mazurka 50 just be weird
    return SE_diagram

# a function to get SE directory
def get_SE_directory(IN, num_fv_per_shingle, thresh, isChroma=True, save=False):

    """

    Args:
        IN (str):
            Input file directory.
        num_fv_per_shingle (int):
            Number of feature vectors per shingle for converting chroma vectors
            to Aligned Hierarchies. Provides "context" of each individual time
            step, so that for notes CDE if num_fv_per_shingle = 2, shingles
        thresh (float):
            Maximum threshold value for converting chroma vectors to AHs.
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

    # define variable to store all SEs
    SE_all = SE_directory('ALL SEs')

    # check if directory path is absolute
    if not os.path.isabs(IN):
        IN = pkg_resources.resource_filename(__name__, IN)

    # loop through each subdirectory
    subdirs = glob.glob(IN + '/*')
    for subdir in subdirs:
        SEs = []
        labels = []
        subdir_name = subdir[len(IN) + 1::]  # ex Expanded or NotExpanded
        if isChroma:
            files = glob.glob(subdir + '/*.csv')
        else:
            files = glob.glob(subdir + '/*.mat')
        # loop through each file per directory, get SE and labels
        for file in files:
            SEs.append(get_SEs(file, num_fv_per_shingle, thresh, isChroma, save))
            labels.append(file[len(subdir) + 1:-4])  # ex "mazurka-51"
        # store all SE info in SE_directory class, add to SE_all
        SE_all.add_class(subdir_name)
        SE_all.add_labels(labels)
        SE_all.add_SEs(SEs)

    return SE_all


def get_dist_mat(SE_all, metric):
    """

    Args:
        SE_all (SNLD_directory):
            Directory that contains all SEs and corresponding information.
        metric (str):
            'b' for bottleneck distance, 'w' for wasserstein distance.

    Returns:
        D (np.ndarray):
            Distance matrix for all SE diagrams in directory.
        labels (list):
            list of all song names.

    """
    # create distance matrix
    N_tot = sum([len(SE_all.SEs[i]) for i in range(len(SE_all.SEs))])
    D = np.zeros((N_tot, N_tot))

    SE = []
    labels = []

    # combine all SEs and labels into their own list
    for i in range(len(SE_all.SEs)):
        for k in range(len(SE_all.SEs[i])):
            SE.append(SE_all.SEs[i][k])
            labels.append(SE_all.labels[i][k])

    # perform distance metric for every possible pair of SE diagrams
    for j in range(N_tot):
        for k in range(j + 1, N_tot):
            if metric == 'b':
                D[j][k] = persim.bottleneck(SE[j], SE[k])
            else:
                D[j][k] = persim.wasserstein(SE[j], SE[k])
            # make symmetrical across diagonal
            D[k][j] = D[j][k]

    return D, labels
