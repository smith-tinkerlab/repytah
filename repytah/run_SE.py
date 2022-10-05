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
import numpy as np
from scipy.io import loadmat
from computePDDists import computeWass_orig, computeBottleneck_orig
from mutual_knn import mutual_knn, pr_values


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


# A function to extract SEs from AHs
# change to take aligned mat straight up, not file path
def get_SEs(AH_mat):
    """
    Extracts SEs from AHs

    Args:
        AH_mat (string):
            Path to .mat file that contains the aligned hierarchies

    Returns:
        SE_diagram is an array that [insert from paper]
    """
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
    # add start and end
    for k in range(len(rows)):
        start_temp = int(cols[k]) + 1  # I think this is 1 indexing again...
        len_temp = key[int(rows[k])]
        SE_diagram.append((start_temp, len_temp + start_temp))
    # check to make sure SL diagram is nonempty. If so, add (0,0)
    if SE_diagram == []:
        SE_diagram.append((0, 0))
        print('Warning: empty diagram found.')

    return SE_diagram


# a function to get SE directory
def get_SE_directory(IN, dirs):
    # define variable to store all SEs
    SE_all = SE_directory('ALL SEs')
    # loop through each directory
    for dir in dirs:
        SEs = []
        labels = []
        dir_name = dir[len(IN)::]  # ex "\\Expanded"
        AHs = glob.glob(dir + '/*.mat')
        # loop through each mat file per directory, get SE and labels
        for AH in AHs:
            SEs.append(get_SEs(AH))
            labels.append(AH[len(dir) + 1:-4])  # name of file ex "mazurka-51"
        # store all SE info in SE_directory class, add to SE_all
        SE_all.add_class(dir_name)
        SE_all.add_labels(labels)
        SE_all.add_SEs(SEs)

    return SE_all


def get_dist_mat(SE_all, inner, outer):
    err = 0.01  # err for wasserstein
    N_tot = sum([len(SE_all.SEs[i]) for i in range(len(SE_all.SEs))])
    D = np.zeros((N_tot, N_tot))
    SE = []
    labels = []

    for i in range(len(SE_all.SEs)):
        for k in range(len(SE_all.SEs[i])):
            SE.append(SE_all.SEs[i][k])
            labels.append(SE_all.labels[i][k])

    for j in range(N_tot):
        # write jth diagram to test file
        open('test1', 'w').write('\n'.join('%s %s' % x for x in SE[j]))
        # now loop through remaining files
        for k in range(j + 1, N_tot):
            open('test2', 'w').write('\n'.join('%s %s' % x for x in SE[k]))
            if outer == 'inf':
                D[j][k] = computeBottleneck_orig('test1', 'test2')
            else:
                D[j][k] = computeWass_orig('test1', 'test2', outer, err, inner)
            D[k][j] = D[j][k]

    return D, labels


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
    IN = "C:\\Users\\quind\\SE_SNL_analysis\\data\\Thresh01_ShingleNumber6"

    dirs = glob.glob(IN + '/*')
    inner = 2
    outer = 2
    SE_all = get_SE_directory(IN, dirs)
    Dists, labels = get_dist_mat(SE_all, inner, outer)




if __name__ == "__main__":
    main()
