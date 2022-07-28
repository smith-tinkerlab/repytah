# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, example.py 
"""

import os
import pandas as pd
import unittest

from repytah.utilities import *
from repytah.example import *


class TestExample(unittest.TestCase):

    # Tests specific to create_sdm
    def test_csv_to_aligned_hierarchies_none_returned(self):
        """
        Tests that nothing is returned.
        """

        file_in = load_ex_data('data/input.csv').to_numpy()
        file_out = "test/hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        output = csv_to_aligned_hierarchies(file_in, file_out,
                                            num_fv_per_shingle, thresh, False)

        self.assertIs(output, None, "Should be none")

    def test_csv_to_aligned_hierarchies_file_saved(self):
        """
        Tests that a file is saved.
        """

        file_in = load_ex_data('data/input.csv').to_numpy()
        file_out = "test/hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle,
                                   thresh, True)

        self.assertTrue(os.path.exists("test/hierarchical_out_file.mat"))

    def test_csv_to_aligned_hierarchies_file_not_empty(self):
        """
        Tests that the file saved isn't empty.
        """

        file_in = load_ex_data('data/input.csv').to_numpy()
        file_out = "test/hierarchical_out_file.mat"
        num_fv_per_shingle = 3
        thresh = 0.01

        csv_to_aligned_hierarchies(file_in, file_out, num_fv_per_shingle,
                                   thresh, True)

        self.assertFalse(
            os.stat("test/hierarchical_out_file.mat").st_size == 0)


if __name__ == '__main__':
    unittest.main()