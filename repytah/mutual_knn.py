# /bin/env python

'''
A python code for running mutual k-nearest neighbors on a distance matrix.

Author: Melissa McGuirl, Brown University

'''

import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform


def mutual_knn(D):
    num_songs = len(D)
    mnn_M = np.zeros((num_songs, num_songs))
    # inflate diagonal to avoid self matches
    D_update = D + 1000000 * np.eye(num_songs)

    minM = np.zeros((len(D_update), len(D_update)))
    for j in range(len(minM)):
        min_dist = np.min(D_update[j])
        min_ind = np.where(D_update[j] == min_dist)
        for k in min_ind[0]:
            minM[j][k] = 1
        # find mnn by adding minM to its transpose, matches have entry 1
    compare_M = minM + minM.T
    [row, col] = np.where(compare_M == 2)
    for j in range(len(row)):
        mnn_M[row[j]][col[j]] = 1
    return mnn_M


def pr_values(mnn_M, truth):
    a = len(mnn_M)
    num_matches_mnn = sum(sum(mnn_M))
    num_matches_true = sum(sum(truth))
    compare = mnn_M + truth
    [row, col] = np.where(compare == 2)
    num_correct = len(row)

    precision = float(num_correct) / float(num_matches_mnn)
    recall = float(num_correct) / float(num_matches_true)

    unmatched_ind = np.where(sum(mnn_M) == 0)[0]
    mismatched_ind = [i for i in np.where(compare == 1)[0] if i not in unmatched_ind]

    return precision, recall, mismatched_ind, unmatched_ind


def main():
    # for sample purposes
    # iris = load_iris
    # data = iris.data()
    # D = squareform(pdist(D))
    #  mnn = mutual_knn(D, 1)
    D = np.array([(0, 5), (5, 0)])


if __name__ == "__main__":
    main

