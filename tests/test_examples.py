# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, utilities.py 
"""
import unittest 

import scipy.io
from utilities import * 
from example import * 
import numpy as np

class TestExample(unittest.TestCase): 
    
    #Tests specific to create_sdm 
    def test_create_sdm(self): 
        """
        EXPLANATION 
        Tests inputs types for creating the sdm.   
        """
        file_in = "input.csv"
        file_out = "hierarchical_out_file.mat"
        num_fv_per_shingle = 12
        thresh = 10
        [fv_mat, num_fv_per_shingle] = csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle, thresh) 
        self.assertIs(type(fv_mat), numpy.ndarray, "Should be numpy array")
        self.assertIs(type(num_fv_per_shingle), int, "Should be integer")
    
    def test_thresholding(self):
        
        result = create_sdm()
        self.assertIs(type(result), numpy.ndarray, "Should be numpy array")
        
    def test_thresholding_output(self): 
        result = thresholding() 
        self.assertIs(type(result), numpy.ndarray, "Should be numpy array")
        
    
if __name__ == '__main__':
    unittest.main() 
