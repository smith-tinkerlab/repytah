"""
run_SE.py

[more detailed module summary]
A python module to extract start-end diagrams from aligned hierarchies.


The module contains the following functions:

[do I need to include the class defined]

    * get_SEs
        insert

    * get_SE_directory
        insert

    * get_dist_mat
        insert

    * get_truth_mat
        insert

Author: Melissa McGuirl
Date: 06/11/18

Sample function call: python run_SE.py -I path_to_files -w 2 -W2
"""

import glob
import argparse
import pathlib
import os
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat
import persim
from computePDDists import computeWass_orig, computeBottleneck_orig
from mutual_knn import mutual_knn, pr_values
from example import csv_to_aligned_hierarchies, save_to_mat


class SE_directory:
    """
    SE class to hold directory names, song labels, and SEs
    """

    labels = []  # can't change to be np empty array without knowing shape
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


def get_AH(filepath, num_fv_per_shingle, thresh):
    file_in = pd.read_csv(filepath, header=None).to_numpy()
    AH_dict = csv_to_aligned_hierarchies(file_in, num_fv_per_shingle, thresh, vis=False)

    return AH_dict

def get_SEs_from_AH_file(AH_mat):
    mat = loadmat(AH_mat, squeeze_me=True, variable_names=['full_key', 'full_matrix_no'])
    key = mat['full_key']
    if type(key) == int:
        key = [key]  # make array to handle single row case
    AH = mat['full_matrix_no']
    blocks = np.nonzero(AH)
    # when there's single row, python swaps the order
    if len(blocks) > 1:
        rows = blocks[0]
        cols = blocks[1]
    else:
        cols = blocks[0]
        rows = np.zeros(len(cols))

    SE_diagram = []
    for k in range(len(rows)):
        start_temp = int(cols[k]) + 1
        len_temp = key[int(rows[k])]
        SE_diagram.append((start_temp, len_temp + start_temp))
    # check to make sure SL diagram is nonempty. If so, add (0,0)
    if SE_diagram == []:
        SE_diagram.append((0, 0))
        print('Warning: empty diagram found.')

    return SE_diagram



def get_SEs(filepath, num_fv_per_shingle, thresh, isChroma=True, save=False):
    """
    Extracts SEs from chroma feature vector csv files

    Args:
        filepath (string):
            Path to file that contains the chroma feature vectors or the aligned hierarchies

    Returns:
        SE_diagram is an array that [insert from paper]
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

            # create directory unless it already exists
            # might want to optimize this, probably save into file_base_dir
            # or smth
            if not os.path.exists(os.path.split(file_out)[0]):
                os.makedirs(os.path.split(file_out)[0])

            save_to_mat(file_out, AH_dict)

    # start from previously saved AHs
    else:
        extension = os.split(filepath)[1]
        if extension == ".mat":
            AH_dict = loadmat(filepath, squeeze_me=True, variable_names=['full_key', 'full_matrix_no'])
        elif extension == ".csv":
            # need to check this
            AH_dict = pd.read_csv(filepath, header=None).to_numpy()
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


def get_SE_directory_shortcut(IN, dirs):
    # define variable to store all SEs
    SE_all = SE_directory('ALL SEs')
    # loop through each directory
    for dir in dirs: # dir: 'C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6\\Expanded'
        SEs = []
        labels = []
        dir_name = dir[len(IN)::]
        AHs = glob.glob(dir + '/*.mat')
        # loop through each mat file per directory, get SE and labels
        for AH in AHs: # AH: 'C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6\\Expanded\\mazurka-50.mat'
            SEs.append(get_SEs_from_AH_file(AH))
            labels.append(AH[len(dir)+1:-4])
        # store all SE info in SE_directory class, add to SE_all
        SE_all.add_class(dir_name)
        SE_all.add_labels(labels)
        SE_all.add_SEs(SEs)

    return SE_all


# a function to get SE directory
def get_SE_directory(IN, dirs, num_fv_per_shingle, thresh, isChroma=True, save=False):
    # define variable to store all SEs
    SE_all = SE_directory('ALL SEs')
    # loop through each directory
    for dir in dirs:
        # convert to numpy
        SEs = []
        labels = []
        dir_name = dir[len(IN) + 1::]  # ex "Expanded" or NotExpanded


        CV_files = glob.glob(dir + '/*.csv')
        # loop through each mat file per directory, get SE and labels
        for file in CV_files:
            SEs.append(get_SEs(file, num_fv_per_shingle, thresh, isChroma, save))
            labels.append(file[len(dir) + 1:-4])  # ex "mazurka-51"
        # store all SE info in SE_directory class, add to SE_all
        SE_all.add_class(dir_name)
        SE_all.add_labels(labels)
        SE_all.add_SEs(SEs)

    return SE_all


# I will probably have to change the parameters so that it only has the choice wasserstein (inf, 2) and bottleneck
# ask katherine which one to set as default (like wasserstein = True) or to have an option parameter
# like 1 = wasserstein and 2 = bottleneck or option = 'w' or 'b'

# default is saving AH on the way, they would have to flip the flag to not save

# change get_SEs to a wrapper function that calls Melissa's original get_SE

# KK said no default but make somebody choose
def get_dist_mat(SE_all, metric):
    err = 0.01  # err for wasserstein
    N_tot = sum([len(SE_all.SEs[i]) for i in range(len(SE_all.SEs))])
    D = np.zeros((N_tot, N_tot))


    SE = []
    labels = []

    # SE_all.SEs is split into 2 lists, one list that has all the expanded SEs and one list that has all the nonexpanded SEs
    # this loop takes all of them and combines them into one list of all the SEs
    # and labels holds the information of which song is which
    # except labels is 2D for expanded and nonexpanded
    # I wonder if there's a better way of doing this ?
    # also really intresting usage of loop variables, why i and k and then j and k
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
            # assumes that wasserstein/bottleneck works like that
            D[k][j] = D[j][k]

    return D, labels

# might not put knn python in repytah
# bc it was the part KK got flamed for
# since wasserstein and bottleneck are similarity matrices
# could write a Jupyter notebook that uses a distance matrix for the mutual knn 1-nearest neighbor as an exmpale
# for what you COULD do witht he distance matrix


def get_truth_mat(labels):
    truth = np.zeros((len(labels), len(labels)))
    for k in range(len(labels)):
        temp = [i for i in np.where(np.array(labels) == labels[k])[0] if i != k]
        truth[k][temp[0]] = 1
    return truth


def main():

    # set up argparse
    # descriptor = "A Python module that converts aligned hierarchies to start-end diagrams"
    # parser = argparse.ArgumentParser(description=descriptor)
    # parser.add_argument('-I', '--indir', required=True,
    #                     help='provide path to folder containing all aligned hieraarchies in separate folders')
    # parser.add_argument('-w', '--inner', required=True, help='specify p for inner norm in Wasserstein computation.')
    # parser.add_argument('-W', '--outer', required=True, help='specify p for outer norm in Wasserstein computation.')
    #
    # # get arguments
    # args = parser.parse_args()
    # IN = args.indir
    # dirs = glob.glob(IN + '/*')
    # inner = args.inner
    # outer = args.outer
    #
    # if outer == 'inf' and inner != 'inf':
    #     print('Warning: Bottleneck distance can only be computed with l-infinity inner norm. Proceeding accordingly.')
    #
    # # get all SES
    # SE_all = get_SE_directory(IN, dirs)
    # print('SE Generation Complete.')
    # Dists, labels = get_dist_mat(SE_all, inner, outer)
    # print('Distance matrix computations complete.')
    # mnn_M = mutual_knn(Dists)
    # truth = get_truth_mat(labels)
    # print('Mutual KNN complete.')
    # p, r, mismatched, unmatched = pr_values(mnn_M, truth)
    # print('Precision = %s, Recall = %s' % (p, r))

    # python run_SE.py -I path_to_files -w 2 -W 2
    # path has to be 2 folders above actual .mat files ex cannot be all the way to \\Expanded
    # IN = "C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6"
    #
    # dirs = glob.glob(IN + '/*')
    # inner = 2
    # outer = 2
    # SE_all = get_SE_directory(IN, dirs)
    # Dists, labels = get_dist_mat(SE_all, inner, outer)

    # ----------------- Testing run_SE ------------------------------
    # AH_mat = "C:\\Users\\quind\\repytah\\repytah\\hierarchical_out_file_one_index.mat"
    # mat = loadmat(AH_mat, squeeze_me=True, variable_names=['full_key', 'full_mat_no_overlaps']) # DELETE LATER need to change to full_mat_no_overlaps
    # key = mat['full_key']
    # if type(key) == int:
    #     key = [key]  # make array to handle single row case
    # AH = mat['full_mat_no_overlaps']
    # blocks = np.nonzero(AH)
    # # when there's single row, python swaps the order
    # if len(blocks) > 1:
    #     rows = blocks[0]
    #     cols = blocks[1]
    # else:
    #     cols = blocks[0]
    #     rows = np.zeros(len(cols))

    # SE_all = SE_directory('ALL SEs')
    #
    # IN = "C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6"
    #
    # dirs = glob.glob(IN + '/*')
    #
    # SE_diagram_np = np.empty((0, 2), int)
    #
    # for k in range(len(rows)):
    #     start_temp = int(cols[k])
    #     len_temp = key[int(rows[k])]  # effect of 1-indexing here, will change once i only pad columns not rows
    #     SE_diagram_np = np.vstack((SE_diagram_np, (start_temp, len_temp)))
    #

    #np.savetxt("SE_np_one_index.csv", SE_diagram_np, delimiter=',')

    # SE_np_one_index = np.loadtxt("C:\\Users\\quind\\repytah\\repytah\\data\\SE_np_one_index.csv", delimiter=',', dtype=int)
    # SE_np = np.loadtxt("C:\\Users\\quind\\repytah\\repytah\\data\\SE_np.csv", delimiter=',', dtype=int)
    #
    # print(np.array_equal(SE_np, SE_np_one_index))
    #
    # SE1 = [[1, 2], [1, 5], [2, 4]]
    # SE2 = [[1, 3], [1, 5], [2, 3]]
    # #print(gudhi.hera.wasserstein_distance(SE1, SE2, order=2, internal_p=2, delta = 0.01))
    # takes around 4 min to run the expanded directory

    get_SEs("C:\\Users\\quind\\SE_SNL_analysis\\data\\chromavectors\\nonexpanded\\input.csv", num_fv_per_shingle=12, thresh=0.02, isChroma=True, save=True)




    #IN = "D:\\Projects\\Smith-College\\repytah"
    IN = "C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6"

    # 'C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6\\Expanded' and nonexpanded
    dirs = glob.glob(IN + '/*')  # get two folders expanded and nonexpanded
    num_fv_per_shingle = 12
    thresh = 0.02
    #SE_all = get_SE_directory_shortcut(IN, dirs, num_fv_per_shingle, thresh)
    SE_all = get_SE_directory_shortcut(IN, dirs)
    D, labels = get_dist_mat(SE_all, 2, 2)

if __name__ == "__main__":
    main()
