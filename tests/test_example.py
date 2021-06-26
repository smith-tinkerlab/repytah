# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, utilities.py 
"""

import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '../aligned-hierarchies'))

import unittest 

from utilities import * 
from example import *

import os.path
from os import path

class TestExample(unittest.TestCase): 
    
    #Tests specific to create_sdm 
    def test_csv_to_aligned_hierarchies_none_returned(self): 
        """
        EXPLANATION 
        Tests that nothing is returned.   
        """

        # file_in = "input.csv"
        file_in = pd.read_csv(os.path.join(os.path.dirname(__file__), "../input.csv")).to_numpy()
        file_out = "hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        output = csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle, thresh)

        self.assertIs(output, None, "Should be none")


    def test_csv_to_aligned_hierarchies_file_saved(self): 
        """
        EXPLANATION 
        Tests that a file is saved.   
        """

        # file_in = "input.csv"
        file_in = pd.read_csv(os.path.join(os.path.dirname(__file__), "../input.csv")).to_numpy()
        file_out = "hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle, thresh)

        self.assertTrue(path.exists("hierarchical_out_file.mat"))


    def test_csv_to_aligned_hierarchies_file_not_empty(self): 
        """
        EXPLANATION 
        Tests that the file saved isn't empty.   
        """

        # file_in = "input.csv"
        file_in = pd.read_csv(os.path.join(os.path.dirname(__file__), "../input.csv")).to_numpy()
        file_out = "hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle, thresh)

        self.assertFalse(os.stat("hierarchical_out_file.mat").st_size == 0)
    
        
#os.stat("file").st_size == 0
if __name__ == '__main__':
    unittest.main() 
