# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, search.py
"""

import unittest
import numpy as np

from repytah.search import find_complete_list
from repytah.search import __find_add_rows as find_add_rows
from repytah.search import find_all_repeats
from repytah.search import find_complete_list_anno_only


class TestSearch(unittest.TestCase):

    def test_find_complete_list(self):
        """
        Tests if find_complete_list finds the correct smaller diagonals
        (and the associated pairs of repeats).
        """

        input_mat = np.array([[8, 8, 14, 14, 1],
                              [14, 14, 56, 56, 1],
                              [8, 8, 62, 62, 1],
                              [56, 56, 62, 62, 1],
                              [14, 14, 104, 104, 1],
                              [62, 62, 104, 104, 1],
                              [8, 8, 110, 110, 1],
                              [56, 56, 110, 110, 1],
                              [104, 104, 110, 110, 1],
                              [4, 14, 52, 62, 11],
                              [4, 14, 100, 110, 11],
                              [26, 71, 74, 119, 46],
                              [1, 119, 1, 119, 119]])

        song_length = 119

        output = find_complete_list(input_mat, song_length)

        expect_output = np.array([[8, 8, 14, 14, 1, 1],
                                  [8, 8, 56, 56, 1, 1],
                                  [8, 8, 62, 62, 1, 1],
                                  [8, 8, 104, 104, 1, 1],
                                  [8, 8, 110, 110, 1, 1],
                                  [14, 14, 56, 56, 1, 1],
                                  [14, 14, 62, 62, 1, 1],
                                  [14, 14, 104, 104, 1, 1],
                                  [14, 14, 110, 110, 1, 1],
                                  [56, 56, 62, 62, 1, 1],
                                  [56, 56, 104, 104, 1, 1],
                                  [56, 56, 110, 110, 1, 1],
                                  [62, 62, 104, 104, 1, 1],
                                  [62, 62, 110, 110, 1, 1],
                                  [104, 104, 110, 110, 1, 1],
                                  [4, 7, 52, 55, 4, 1],
                                  [4, 7, 100, 103, 4, 1],
                                  [9, 14, 57, 62, 6, 1],
                                  [9, 14, 105, 110, 6, 1],
                                  [63, 71, 111, 119, 9, 1],
                                  [4, 13, 52, 61, 10, 1],
                                  [4, 13, 100, 109, 10, 1],
                                  [4, 14, 52, 62, 11, 1],
                                  [4, 14, 100, 110, 11, 1],
                                  [52, 62, 100, 110, 11, 1],
                                  [57, 71, 105, 119, 15, 1],
                                  [26, 51, 74, 99, 26, 1],
                                  [26, 55, 74, 103, 30, 1],
                                  [26, 61, 74, 109, 36, 1],
                                  [26, 71, 74, 119, 46, 1]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test__find_add_rows(self):
        """
        Tests if __find_add_rows finds the correct pairs of repeated
        structures, represented as diagonals of a certain length, k.
        """

        # Test for pairs of repeated structures that start at the same time
        # step as previously found pairs of repeated structures of the same
        # length
        lst_no_anno_ep1 = np.array([[1, 15, 31, 45, 15],
                                    [1, 10, 46, 55, 10],
                                    [31, 40, 46, 55, 10],
                                    [10, 20, 40, 50, 11]])
        check_inds_ep1 = np.array([1, 31, 46])
        k_ep1 = 10

        output_ep1 = find_add_rows(lst_no_anno_ep1, check_inds_ep1, k_ep1)

        expect_output_ep1 = np.array([[1, 10, 31, 40, 10],
                                      [11, 15, 41, 45, 5],
                                      [1, 10, 31, 40, 10],
                                      [11, 15, 41, 45, 5]])

        # Test output type
        self.assertIs(type(output_ep1), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output_ep1), np.size(expect_output_ep1))
        # Test output result
        self.assertEqual(output_ep1.tolist(), expect_output_ep1.tolist())

        # Test for pairs of repeated structures that end at the same time step
        # as previously found pairs of repeated structures of the same length
        lst_no_anno_ep2 = np.array([[4, 4, 14, 14, 1],
                                    [4, 4, 56, 56, 1],
                                    [4, 4, 110, 110, 1],
                                    [14, 14, 56, 56, 1],
                                    [14, 14, 110, 110, 1],
                                    [56, 56, 110, 110, 1],
                                    [4, 14, 52, 62, 11]])
        check_inds_ep2 = np.array([4, 14, 56, 110])
        k_ep2 = 1

        output_ep2 = find_add_rows(lst_no_anno_ep2, check_inds_ep2, k_ep2)

        expect_output_ep2 = np.array([[4, 4, 52, 52, 1],
                                      [5, 14, 53, 62, 10],
                                      [4, 13, 52, 61, 10],
                                      [14, 14, 62, 62, 1],
                                      [4, 7, 52, 55, 4],
                                      [8, 8, 56, 56, 1],
                                      [9, 14, 57, 62, 6]])

        self.assertIs(type(output_ep2), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output_ep2), np.size(expect_output_ep2))
        # Test output result
        self.assertEqual(output_ep2.tolist(), expect_output_ep2.tolist())

        # Test for pairs of repeated structures that neither start nor end at
        # the same time step as previously found pairs of repeated structures
        # of the same length
        lst_no_anno_ep3 = np.array([[8, 8, 14, 14, 1],
                                    [14, 14, 56, 56, 1],
                                    [8, 8, 62, 62, 1],
                                    [56, 56, 62, 62, 1],
                                    [14, 14, 104, 104, 1],
                                    [62, 62, 104, 104, 1],
                                    [8, 8, 110, 110, 1],
                                    [56, 56, 110, 110, 1],
                                    [104, 104, 110, 110, 1],
                                    [4, 14, 52, 62, 11],
                                    [4, 14, 100, 110, 11],
                                    [26, 71, 74, 119, 46]])
        check_inds_ep3 = np.array([4, 52, 100])
        k = 11

        output_ep3 = find_add_rows(lst_no_anno_ep3, check_inds_ep3, k)

        expect_output_ep3 = np.array([[26, 51, 74, 99, 26],
                                      [52, 62, 100, 110, 11],
                                      [63, 71, 111, 119, 9],
                                      [26, 51, 74, 99, 26],
                                      [52, 62, 100, 110, 11],
                                      [63, 71, 111, 119, 9]])

        # Test output type
        self.assertIs(type(output_ep3), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output_ep3), np.size(expect_output_ep3))
        # Test output result
        self.assertEqual(output_ep3.tolist(), expect_output_ep3.tolist())

    def test_find_all_repeats(self):
        """
        Tests if find_all_repeats finds all the correct diagonals present
        in thresh_mat.
        """

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
                                  [1, 2, 3, 4, 2],
                                  [2, 3, 4, 5, 2],
                                  [2, 3, 4, 5, 2]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_find_complete_list_anno_only(self):
        """
        Tests if find_complete_list_anno_only finds all the correct annotations
        for all pairs of repeats found in find_all_repeats.
        """

        pair_list = np.array([[3, 3, 5, 5, 1],
                              [2, 2, 8, 8, 1],
                              [3, 3, 9, 9, 1],
                              [2, 2, 15, 15, 1],
                              [8, 8, 15, 15, 1],
                              [4, 4, 17, 17, 1],
                              [2, 3, 8, 9, 2],
                              [3, 4, 9, 10, 2],
                              [2, 3, 15, 16, 2],
                              [8, 9, 15, 16, 2],
                              [3, 4, 16, 17, 2],
                              [2, 4, 8, 10, 3],
                              [3, 5, 9, 11, 3],
                              [7, 9, 14, 16, 3],
                              [2, 4, 15, 17, 3],
                              [3, 5, 16, 18, 3],
                              [9, 11, 16, 18, 3],
                              [7, 10, 14, 17, 4],
                              [7, 11, 14, 18, 5],
                              [8, 12, 15, 19, 5],
                              [7, 12, 14, 19, 6]])

        song_length = 19

        output = find_complete_list_anno_only(pair_list, song_length)

        expect_output = np.array([[2, 2, 8, 8, 1, 1],
                                  [2, 2, 15, 15, 1, 1],
                                  [8, 8, 15, 15, 1, 1],
                                  [3, 3, 5, 5, 1, 2],
                                  [3, 3, 9, 9, 1, 2],
                                  [4, 4, 17, 17, 1, 3],
                                  [2, 3, 8, 9, 2, 1],
                                  [2, 3, 15, 16, 2, 1],
                                  [8, 9, 15, 16, 2, 1],
                                  [3, 4, 9, 10, 2, 2],
                                  [3, 4, 16, 17, 2, 2],
                                  [2, 4, 8, 10, 3, 1],
                                  [2, 4, 15, 17, 3, 1],
                                  [3, 5, 9, 11, 3, 2],
                                  [3, 5, 16, 18, 3, 2],
                                  [9, 11, 16, 18, 3, 2],
                                  [7, 9, 14, 16, 3, 3],
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