# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies
"""

import numpy as np
import pytest
import scipy.io

# not possible yet
#from utilities import create_sdm

# maybe not necessary? at least for simple tests
#@pytest.mark.parametrize('infile', )

# infile must be a string with the name of a .mat file in the same folder
def test_sdm(infile):
    
    # could pull it out into its own function for many tests (like librosa)
    DATA = scipy.io.loadmat(infile, chars_as_strings = True)
    
    with open(DATA['inputfile'], 'r', newline='\n') as f:
        fv_mat = np.loadtxt(f, delimiter = ',')
        
    num_fv = DATA['num_fv_per_shingle']
    
    self_dissim_mat = create_sdm(fv_mat, num_fv)
    
    # should this be multiple tests?
    
    # Verify that file inputs correctly
    assert fv_mat == DATA['find variable in .m']
    
    # Verify that self dissimilarity matrix is correct
    assert self_dissim_mat == DATA['matAS']