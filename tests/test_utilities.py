# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, utilities.py

"""

import sys
import os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\aligned-hierarchies")

import unittest
import numpy as np

import utilities
from utilities import __find_song_pattern as _test_find_song_pattern


class test_utilities(unittest.TestCase):

    def test_add_annotations(self):
        
        input_mat = np.array([[1,1,2,2,1,1],
                      [3,6,7,10,4,2]])
        song_length = 16
        output = utilities.add_annotations(input_mat, song_length)
        expect_output = np.array([[1,1,2,2,1,1],[3,6,7,10,4,2]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output),np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
    
    def test_create_sdm(self):
        
        my_data = np.array([[0,0.5,0,0,0,1,0,0],
                    [0,2,0,0,0,0,0,0],
                    [0,0,0,0,0,0,3,0],
                    [0,3,0,0,2,0,0,0],
                    [0,1.5,0,0,5,0,0,0]])

        num_fv_per_shingle = 3
        output = utilities.create_sdm(my_data,num_fv_per_shingle)
        expect_output = np.array([[0.0, 1.0, 1.0, 0.3739524907237728, 0.9796637041304479, 1.0],
                                  [1.0, 0.0, 1.0, 1.0, 0.45092001152209327, 0.9598390335548751],
                                  [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                                  [0.3739524907237728, 1.0, 1.0, 0.0, 1.0, 1.0],
                                  [0.9796637041304479, 0.45092001152209327, 1.0, 1.0, 0.0, 1.0],
                                  [1.0, 0.9598390335548751, 1.0, 1.0, 1.0, 0.0]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
    def test_stretch_diags(self):
        
        thresh_diags = np.array([[0,0,1,0,0],
                         [0,1,0,0,0],
                         [0,0,1,0,0],
                         [0,0,0,0,0],
                         [0,0,0,0,0]])
        band_width = 3
        output = utilities.stretch_diags(thresh_diags,band_width)
        
        expect_output = [[False,False,False,False,False,False,False],
                         [False,True,False,False,False,False,False],
                         [ True,False,True,False,False,False,False],
                         [False,True,False,True,False,False,False],
                         [False,False,True,False,True,False,False],
                         [False,False,False,False,False,False,False],
                         [False,False,False,False,False,False,False]]
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output)
        
    def test_find_initial_repeats(self):
        
        # Input with single row && bandwidth_vec>thresh_bw
        thresh_mat = np.array([[1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
        
        expect_output = np.array([[1,1,1,1,1]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
        
        # Input with single row && bandwidth_vec<thresh_bw
        thresh_mat = np.array([[1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 3
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
        
        expect_output = np.array([])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
        
        # No diagonals of length band_width
        thresh_mat = np.array([[0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0],
                                [0,0,0,0]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output),0)
        
        
        # Thresh_mat with middle overlaps
        thresh_mat = np.array([[1,0,0,1],
                               [0,1,0,0],
                               [0,0,1,0],
                               [1,0,0,1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
        
        expect_output = np.array([[1,1,1,1,1],
                                  [2,2,2,2,1],
                                  [3,3,3,3,1],
                                  [1,1,4,4,1],
                                  [4,4,4,4,1]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
        
        # Big input without middle overlaps
        thresh_mat = np.array([[1,0,0,1,0,0,0,1,0,0],
                       [0,1,0,0,1,1,0,0,1,0],
                       [0,0,1,0,0,1,1,0,0,1],
                       [1,0,0,1,0,0,1,1,0,0],
                       [0,1,0,0,1,0,1,0,0,0],
                       [0,1,1,0,0,1,0,1,1,0],
                       [0,0,1,1,1,0,1,0,1,0],
                       [1,0,0,1,0,1,0,1,0,1],
                       [0,1,0,0,0,1,1,0,1,0],
                       [0,0,1,0,0,0,0,1,0,1]])

        bandwidth_vec = np.array([1,2,3,4,5,6,7,8,9,10])
        thresh_bw = 0
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
        
        expect_output = np.array([[6,6,9,9,1],
                                  [5,6,7,8,2],
                                  [7,8,9,10,2],
                                  [1,3,4,6,3],
                                  [2,4,5,7,3],
                                  [2,4,6,8,3],
                                  [1,3,8,10,3],
                                  [1,10,1,10,10]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
        
        # Big input with middle overlaps
        thresh_mat = np.array([[1,0,0,0,0,0,0,0,0,1,0,0,0],
                       [0,1,0,0,0,0,0,0,0,0,0,0,0],
                       [0,0,1,0,0,0,0,0,1,0,0,0,0],
                       [0,0,0,1,0,0,0,0,0,0,0,0,0],
                       [0,0,0,0,1,0,0,0,0,0,0,0,0],
                       [0,0,0,0,0,1,1,0,0,0,0,0,0],
                       [0,0,0,0,0,1,1,0,0,0,1,0,0],
                       [0,0,0,0,0,0,0,1,0,0,1,0,0],
                       [0,0,1,0,0,0,0,0,1,0,0,0,0],
                       [1,0,0,0,0,0,0,0,0,1,0,0,0],
                       [0,0,0,0,0,0,1,1,0,0,1,0,0],
                       [0,0,0,0,0,0,0,0,0,0,0,1,0],
                       [0,0,0,0,0,0,0,0,0,0,0,0,1]])

        bandwidth_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
        thresh_bw = 0
        output = utilities.find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)
       
        expect_output = np.array([[6,6,7,7,1],
                                  [3,3,9,9,1],
                                  [1,1,10,10,1],
                                  [7,7,11,11,1],
                                  [8,8,11,11,1],
                                  [6,8,5,7,3],
                                  [1,13,1,13,13]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
    
    def test_reconstruct_full_block(self):
        
        # Input without overlaps 
        pattern_mat = np.array([[0,0,0,0,1,0,0,0,0,1],
              [0,1,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,0,0,1,0],
              [1,0,0,0,0,0,1,0,0,0],
              [1,0,0,0,0,0,1,0,0,0]])

        pattern_key = np.array([1,2,2,3,4])
        output = utilities.reconstruct_full_block(pattern_mat, pattern_key)
        
        expect_output = np.array([[0,0,0,0,1,0,0,0,0,1],
                                  [0,1,1,0,0,0,0,1,1,0],
                                  [0,0,1,1,0,0,0,0,1,1],
                                  [1,1,1,0,0,0,1,1,1,0],
                                  [1,1,1,1,0,0,1,1,1,1]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim,2)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
        # Input with overlaps
        pattern_mat = np.array([0,1,0,0,1,1,0,0,0,0])
        pattern_key = np.array([3])
        output = utilities.reconstruct_full_block(pattern_mat, pattern_key)
       
        expect_output = np.array([[0,1,1,1,1,2,2,1,0,0]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim,2)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
   
    
    def test_reformat(self):
        
        pattern_mat = np.array([[0,0,0,0,1,0,0,0,0,1],
              [0,1,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,0,0,1,0],
              [1,0,0,0,0,0,1,0,0,0],
              [1,0,0,0,0,0,1,0,0,0]])
        pattern_key = np.array([1,2,2,3,4])
        output = utilities.reformat(pattern_mat, pattern_key)
        
        expect_output = np.array([[5,5,10,10,1],
                                  [2,3,8,9,2],
                                  [3,4,9,10,2],
                                  [1,3,7,9,3],
                                  [1,4,7,10,4]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim,2)
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
    
    def test_find_song_pattern(self):
        
        thresh_diags = np.array([[1, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 1, 0, 0],
                       [0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 1]])
        output = _test_find_song_pattern(thresh_diags)
       
        expect_output = np.array([1,2,2,2,3])
        
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
        
    def test_get_annotation_lst(self):
        
        key_lst = np.array([1,1,3,4,5])
        output = utilities.get_annotation_lst (key_lst)
        
        expect_output = np.array([1,2,1,1,1])
        
        # Test output result
        self.assertEqual(output.tolist(),expect_output.tolist())
               
         
if __name__ == '__main__':
    unittest.main()