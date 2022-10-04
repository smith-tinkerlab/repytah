"""
A python module to extract start-normalized length  diagrams from aligned hierarchies.

Author: Melissa McGuirl
Date: 06/11/18

Sample function call: python run_SNLD.py -I path_to_files -N std -alpha 5 -w 2 -W2
"""

import glob
import argparse
import numpy as np
from scipy.io import loadmat
from computePDDists import computeWass, computeBottleneck
from mutual_knn import mutual_knn, pr_values


# SNLD class to hold directory names, song labels, and SNLDs
class SNLD_directory:
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


# A function to extract SNLDs from AHs
def get_SNLDs(AH_mat, N, alpha):
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

    SL_diagram = []
    start_norm = 0
    start_min = 10000000000
    for k in range(len(rows)):
        start_temp = int(cols[k]) + 1
        start_norm = max(start_norm, start_temp)
        start_min = min(start_temp, start_min)
        len_temp = key[int(rows[k])]
        SL_diagram.append((start_temp, len_temp))
    # check to make sure SL diagram is nonempty. If so, add (0,0)
    if SL_diagram == []:
        SL_diagram.append((0, 0))
        print('Warning: empty diagram found.')
    else:
        # now normalize
        if N == 'std':
            # norm factor is alpha/max(start times)
            norm_factor = float(alpha) / float(start_norm)
            SL_diagram = [(norm_factor * s, l) for (s, l) in SL_diagram]
        elif N == 'cheb':
            r = 0.5
            T = start_norm - start_min
            SL_diagram = [(float(alpha) * r * (1 - np.cos((s - start_min) * np.pi / T)), l) for (s, l) in SL_diagram]

    return SL_diagram


# a function to get SNLD directory
def get_SNLD_directory(IN, dirs, N, alpha):
    # define variable to store all SNLDs
    SNLD_all = SNLD_directory('ALL SNLDs ' + str(alpha))
    # loop through each directory
    for dir in dirs:
        SNLDs = []
        labels = []
        dir_name = dir[len(IN)::]
        AHs = glob.glob(dir + '/*.mat')
        # loop through each mat file per directory, get SNLD and labels
        for AH in AHs:
            SNLDs.append(get_SNLDs(AH, N, alpha))
            labels.append(AH[len(dir) + 1:-4])
        # store all SNLD info in SNLD_directory class, add to SNLD_all
        SNLD_all.add_class(dir_name)
        SNLD_all.add_labels(labels)
        SNLD_all.add_SNLDs(SNLDs)

    return SNLD_all


def get_dist_mat(SNLD_all, inner, outer):
    err = 0.01  # err for wasserstein
    N_tot = sum([len(SNLD_all.SNLDs[i]) for i in range(len(SNLD_all.SNLDs))])
    D = np.zeros((N_tot, N_tot))
    # reshape all SNLDs into one array, do the same with labels preserving order.
    # SNLD = np.reshape(SNLD_all.SNLDs, [N_tot], order='c')
    # labels = np.reshape(SNLD_all.labels, [N_tot], order='c')
    SNLD = []
    labels = []

    for i in range(len(SNLD_all.SNLDs)):
        for k in range(len(SNLD_all.SNLDs[i])):
            SNLD.append(SNLD_all.SNLDs[i][k])
            labels.append(SNLD_all.labels[i][k])

    for j in range(N_tot):
        # write jth diagram to test file
        open('test1', 'w').write('\n'.join('%s %s' % x for x in SNLD[j]))
        # now loop through remaining files
        for k in range(j + 1, N_tot):
            open('test2', 'w').write('\n'.join('%s %s' % x for x in SNLD[k]))
            if outer == 'inf':
                D[j][k] = computeBottleneck('test1', 'test2')
            else:
                D[j][k] = computeWass('test1', 'test2', outer, err, inner)
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
    descriptor = "A Python module that converts aligned hierarchies to start-end diagrams"
    parser = argparse.ArgumentParser(description=descriptor)
    parser.add_argument('-I', '--indir', required=True,
                        help='provide path to folder containing all aligned hieraarchies in separate folders')
    parser.add_argument('-N', '--norm', required=True,
                        help='specify desired normalization. Options: none for no normalization, std for ordinary normalization, cheb for chebyshev normalization.')
    parser.add_argument('-A', '--alpha', required=True,
                        help='specify scaling parameter alpha, alpha = 1 indicates no scaling.')
    parser.add_argument('-w', '--inner', required=True, help='specify p for inner norm in Wasserstein computation.')
    parser.add_argument('-W', '--outer', required=True, help='specify p for outer norm in Wasserstein computation.')

    # get arguments
    args = parser.parse_args()
    IN = args.indir
    dirs = glob.glob(IN + '/*')
    norm = args.norm
    alpha = args.alpha
    inner = args.inner
    outer = args.outer

    if outer == 'inf' and inner != 'inf':
        print('Warning: Bottleneck distance can only be computed with l-infinity inner norm. Proceeding accordingly.')

    # get all SNLDS
    SNLD_all = get_SNLD_directory(IN, dirs, norm, alpha)
    print('SNLD Generation Complete.')
    Dists, labels = get_dist_mat(SNLD_all, inner, outer)
    print('Distance matrix computations complete.')
    mnn_M = mutual_knn(Dists)
    truth = get_truth_mat(labels)
    print('Mutual KNN complete.')
    p, r, mismatched, unmatched = pr_values(mnn_M, truth)
    print('Precision = %s, Recall = %s' % (p, r))


if __name__ == "__main__":
    main()
