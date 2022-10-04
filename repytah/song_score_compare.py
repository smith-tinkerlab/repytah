#!/bin/bash python
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from run_SNLD import *
import glob
import numpy as np

def main():
    IN = '../data/ScoreData/Thresh04_ShingleNumber12'
    dirs = glob.glob(IN + '/*')
    norm = 'std'
    alpha = 100
    SNLD_all = get_SNLD_directory(IN, dirs, norm, alpha)
    Expanded_Scores_SNL = SNLD_all.SNLDs[0]
    SNL_score_labels = SNLD_all.labels[0]
    song_files = '../data/song_SNL_alpha100/Thresh04/Shingle12/'
    song_files = glob.glob(song_files + '/*.txt')
    new_files = []
    for file in song_files:
        inFile = open(file, 'r').readlines()
        if len(inFile) > 10:
                new_files.append(file)

    new_files = np.sort(new_files)
    D = np.zeros((52, len(new_files)))
    for j in range(52):
        open('score', 'w').write('\n'.join('%s %s' % x for x in Expanded_Scores_SNL[j]))
        for k in range(len(new_files)):
            D[j][k] = computeWass('score', new_files[k], 2, 0.01, 2)
        print('done: ', j)
    Truth = np.zeros((52, len(new_files)))
    for k in range(len(new_files)):
        for j in range(52):
            if new_files[k].find(SNL_score_labels[j]) > -1:
                Truth[j][k] =1
    bad_count = 0
    total_ap = 0
    for s in range(len(new_files)):
        dist_row = -D.T[s]
        ranks = rankdata(dist_row, method="min")
        truth_row = Truth.T[s]
        row_ap = average_precision_score(truth_row,ranks)
        total_ap = total_ap + row_ap

    print('MAP score: ', total_ap/float(len(new_files)))

if __name__ == "__main__":
     main()
