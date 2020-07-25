#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, transform.py
"""

import unittest 
import scipy.io 
import numpy as np 
import transform
from transform import create_anno_remove_overlaps 

class TestTransform(unittest.TestCase): 
    
    def test_create_anno_remove_overlaps(self): 
        
        inputMat = np.array([2,2,4,4,1,1])
        song_length = 10 
        bandwidth = 1
        
        expect_pattern_row = np.array([0,1,0,1,0,0,0,0,0,0])
        expect_k_lst_out = np.array([2,2,4,4,1,1])
        expect_overlaps_lst = np.array([])
        
        print("okay")
        
        output_tuple = create_anno_remove_overlaps(inputMat, song_length, bandwidth)
        
        print("okay2") 
        self.assertEqual(output_tuple[1], expect_pattern_row)
        
        
if __name__ == '__main__': 
    unittest.main() 