# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, search.py

"""

import sys
import os
# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path+"\\aligned-hierarchies")
sys.path.append(os.path.join(os.path.dirname('__file__'), '../aligned-hierarchies'))
    
import unittest
import numpy as np

import search
from search import __find_add_srows as find_add_srows
from search import __find_add_erows as find_add_erows
from search import __find_add_mrows as find_add_mrows
from search import find_all_repeats as find_all_repeats
from search import find_complete_list_anno_only as find_complete_list_anno_only


class test_search(unittest.TestCase):

    def test_find_complete_list(self):
        
        input_mat = np.array([[  8,   8,  14,  14,   1],
                              [ 14,  14,  56,  56,   1],
                              [  8,   8,  62,  62,   1],
                              [ 56,  56,  62,  62,   1],
                              [ 14,  14, 104, 104,   1],
                              [ 62,  62, 104, 104,   1],
                              [  8,   8, 110, 110,   1],
                              [ 56,  56, 110, 110,   1],
                              [104, 104, 110, 110,   1],
                              [  4,  14,  52,  62,  11],
                              [  4,  14, 100, 110,  11],
                              [ 26,  71,  74, 119,  46],
                              [  1, 119,   1, 119, 119]])

        song_length = 119

        output = search.find_complete_list(input_mat, song_length)

        expect_output = np.array([[  8,   8,  14,  14,  1, 1],
                                  [  8,   8,  56,  56,  1, 1],
                                  [  8,   8,  62,  62,  1, 1],
                                  [  8,   8, 104, 104,  1, 1],
                                  [  8,   8, 110, 110,  1, 1],
                                  [ 14,  14,  56,  56,  1, 1],
                                  [ 14,  14,  62,  62,  1, 1],
                                  [ 14,  14, 104, 104,  1, 1],
                                  [ 14,  14, 110, 110,  1, 1],
                                  [ 56,  56,  62,  62,  1, 1],
                                  [ 56,  56, 104, 104,  1, 1],
                                  [ 56,  56, 110, 110,  1, 1],
                                  [ 62,  62, 104, 104,  1, 1],
                                  [ 62,  62, 110, 110,  1, 1],
                                  [104, 104, 110, 110,  1, 1],
                                  [  4,   7,  52,  55,  4, 1],
                                  [  4,   7, 100, 103,  4, 1],
                                  [  9,  14,  57,  62,  6, 1],
                                  [  9,  14, 105, 110,  6, 1],
                                  [ 63,  71, 111, 119,  9, 1],
                                  [  4,  13,  52,  61, 10, 1],
                                  [  4,  13, 100, 109, 10, 1],
                                  [  4,  14,  52,  62, 11, 1],
                                  [  4,  14, 100, 110, 11, 1],
                                  [ 52,  62, 100, 110, 11, 1],
                                  [ 57,  71, 105, 119, 15, 1],
                                  [ 26,  51,  74,  99, 26, 1],
                                  [ 26,  55,  74, 103, 30, 1],
                                  [ 26,  61,  74, 109, 36, 1],
                                  [ 26,  71,  74, 119, 46, 1]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
        

    def test__find_add_srows(self): 
        lst_no_anno = np.array([[ 1, 15, 31, 45, 15],
                                [ 1, 10, 46, 55, 10],
                                [31, 40, 46, 55, 10],
                                [10, 20, 40, 50, 11]])
        check_inds = np.array([1, 31, 46])
        k = 10
        
        output = find_add_srows(lst_no_anno, check_inds, k)
        
        expect_output = np.array([[ 1, 10, 31, 40, 10],
                                  [11, 15, 41, 45,  5],
                                  [ 1, 10, 31, 40, 10],
                                  [11, 15, 41, 45,  5]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
        
        
    def test__find_add_erows(self):
        
        lst_no_anno= np.array([[  8,   8,  14,  14,  1],
                               [ 14,  14,  56,  56,  1],
                               [  8,   8,  62,  62,  1],
                               [ 56,  56,  62,  62,  1],
                               [ 14,  14, 104, 104,  1],
                               [ 62,  62, 104, 104,  1],
                               [  8,   8, 110, 110,  1],
                               [ 56,  56, 110, 110,  1],
                               [104, 104, 110, 110,  1],
                               [  4,  14,  52,  62, 11],
                               [  4,  14, 100, 110, 11],
                               [ 26,  71,  74, 119, 46]])
        check_inds = np.array([8, 14, 56, 62, 104, 110])
        k = 1

        
        output = find_add_erows(lst_no_anno, check_inds, k)
        
        expect_output = np.array([[ 14, 14,  62,  62,  1],
                                  [ 14, 14, 110, 110,  1],
                                  [  4, 13,  52,  61, 10],
                                  [  4, 13, 100, 109, 10],
                                  [ 14, 14,  62,  62,  1],
                                  [  4, 13,  52,  61, 10],
                                  [ 14, 14, 110, 110,  1],
                                  [  4, 13, 100, 109, 10]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
         

    def test__find_add_mrows(self):
        lst_no_anno= np.array([[  8,   8,  14,  14,  1],
                               [ 14,  14,  56,  56,  1],
                               [  8,   8,  62,  62,  1],
                               [ 56,  56,  62,  62,  1],
                               [ 14,  14, 104, 104,  1],
                               [ 62,  62, 104, 104,  1],
                               [  8,   8, 110, 110,  1],
                               [ 56,  56, 110, 110,  1],
                               [104, 104, 110, 110,  1],
                               [  4,  14,  52,  62, 11],
                               [  4,  14, 100, 110, 11],
                               [ 26,  71,  74, 119, 46]])
        check_inds =np.array([4, 52, 100])
        k = 11
        
        output = find_add_mrows(lst_no_anno, check_inds, k)
        
        expect_output = np.array([[ 26,  51,  74,  99, 26],
                                  [ 52,  62, 100, 110, 11],
                                  [ 63,  71, 111, 119,  9],
                                  [ 26,  51,  74,  99, 26],
                                  [ 52,  62, 100, 110, 11],
                                  [ 63,  71, 111, 119,  9]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
        

    def test_find_all_repeats(self):
        
        thresh_temp = np.array([[1, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0],
                                [1, 0, 1, 0, 1],
                                [0, 1, 0, 1, 0],
                                [0, 0, 1, 0, 1]])

        band_width_vec = np.array([1, 2, 3, 4, 5])
        
        output = find_all_repeats(thresh_temp, band_width_vec)
        
        expect_output = np.array([[1, 1, 3, 3, 1],
                                  [2, 2, 4, 4, 1],
                                  [3, 3, 5, 5, 1],
                                  [1, 2, 3, 4, 2],
                                  [2, 3, 4, 5, 2],
                                  [1, 2, 3, 4, 2],
                                  [2, 3, 4, 5, 2]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
        

    def test_find_complete_list_anno_only(self):
        pair_list = np.array([[3,  3,  5,  5, 1],
                              [2,  2,  8,  8, 1],
                              [3,  3,  9,  9, 1],
                              [2,  2, 15, 15, 1],
                              [8,  8, 15, 15, 1],
                              [4,  4, 17, 17, 1],
                              [2,  3,  8,  9, 2],
                              [3,  4,  9, 10, 2],
                              [2,  3, 15, 16, 2],
                              [8,  9, 15, 16, 2],
                              [3,  4, 16, 17, 2],
                              [2,  4,  8, 10, 3],
                              [3,  5,  9, 11, 3],
                              [7,  9, 14, 16, 3],
                              [2,  4, 15, 17, 3],
                              [3,  5, 16, 18, 3],
                              [9, 11, 16, 18, 3],
                              [7, 10, 14, 17, 4],
                              [7, 11, 14, 18, 5],
                              [8, 12, 15, 19, 5],
                              [7, 12, 14, 19, 6]])
        
        song_length = 19
        
        output = find_complete_list_anno_only(pair_list, song_length)
        
        expect_output = np.array([[2,  2,  8,  8, 1, 1],
                                  [2,  2, 15, 15, 1, 1],
                                  [8,  8, 15, 15, 1, 1],
                                  [3,  3,  5,  5, 1, 2],
                                  [3,  3,  9,  9, 1, 2],
                                  [4,  4, 17, 17, 1, 3],
                                  [2,  3,  8,  9, 2, 1],
                                  [2,  3, 15, 16, 2, 1],
                                  [8,  9, 15, 16, 2, 1],
                                  [3,  4,  9, 10, 2, 2],
                                  [3,  4, 16, 17, 2, 2],
                                  [2,  4,  8, 10, 3, 1],
                                  [2,  4, 15, 17, 3, 1],
                                  [3,  5,  9, 11, 3, 2],
                                  [3,  5, 16, 18, 3, 2],
                                  [9, 11, 16, 18, 3, 2],
                                  [7,  9, 14, 16, 3, 3],
                                  [7, 10, 14, 17, 4, 1],
                                  [7, 11, 14, 18, 5, 1],
                                  [8, 12, 15, 19, 5, 2],
                                  [7, 12, 14, 19, 6, 1]])
        
        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())
        
         
if __name__ == '__main__':
    unittest.main()