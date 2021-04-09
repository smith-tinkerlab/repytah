# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, utilities.py

"""

import os
import sys
sys.path.insert(0, os.path.abspath('../ah/aligned-hierarchies'))
import unittest
import assemble
from assemble import check_overlaps
import numpy as np


class test_utilities(unittest.TestCase):

    #def test_breakup_overlaps_by_intersect(self):
    
    def test_check_overlaps(self):
        input_mat = np.array([[1, 1, 0, 1, 0, 0,],
                             [1, 1, 1, 0, 1, 0],
                             [0, 1, 1, 0, 0, 1],
                             [1, 0, 0, 1, 0, 0], 
                             [0, 1, 0, 0, 1, 0], 
                             [0, 0, 1, 0, 0, 1]])
        
        expect_output = np.array([[0,1,1,1,1,0],
                                [0,0,1,1,1,1],
                                [0,0,0,0,1,1],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0],
                                [0,0,0,0,0,0]])
        
        output = check_overlaps(input_mat)

        self.assertIs(type(output), np.ndarray)
        self.assertEqual(np.size(output),np.size(expect_output))
        self.assertEqual(output.tolist(),expect_output.tolist())

    #def test_compare_and_cut(self):

    #def test_num_of_parts(self):
    #def test_inds_to_rows(self):
    #def test_merge_based_on_length(self):
    #def test_merge_rows(self):
    #def test_hierarchical_structure(self):

        
        
    
    
               
         
if __name__ == '__main__':
    unittest.main()