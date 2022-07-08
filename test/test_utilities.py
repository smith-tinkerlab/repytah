# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, utilities.py
"""

import unittest
import numpy as np
from repytah.utilities import *
from repytah.utilities import __find_song_pattern as find_song_pattern


class TestUtilities(unittest.TestCase):

    def test_create_sdm(self):
        """
        Tests if create_sdm creates the correct self-dissimilarity matrix
        given feature vectors.
        """

        my_data = np.array([[0, 0.5, 0, 0, 0, 1, 0, 0],
                            [0, 2, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 3, 0],
                            [0, 3, 0, 0, 2, 0, 0, 0],
                            [0, 1.5, 0, 0, 5, 0, 0, 0]])

        num_fv_per_shingle = 3
        output = create_sdm(my_data, num_fv_per_shingle)
        expect_output = np.array([
            [0.0, 1.0, 1.0, 0.3739524907237728, 0.9796637041304479, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.45092001152209327, 0.9598390335548751],
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            [0.3739524907237728, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.9796637041304479, 0.45092001152209327, 1.0, 1.0, 0.0, 1.0],
            [1.0, 0.9598390335548751, 1.0, 1.0, 1.0, 0.0]
        ])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_find_initial_repeats(self):
        """
        Tests if find_initial_repeats finds all the large repeated structures
        represented as diagonals in thresh_mat.
        """

        # Input with single row && bandwidth_vec > thresh_bw
        thresh_mat = np.array([[1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        expect_output = np.array([[1, 1, 1, 1, 1]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Input with single row && bandwidth_vec < thresh_bw
        thresh_mat = np.array([[1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 3
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        expect_output = np.array([])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # No diagonals of length band_width
        thresh_mat = np.array([[0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), 0)

        # Thresh_mat with middle overlaps
        thresh_mat = np.array([[1, 0, 0, 1],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [1, 0, 0, 1]])
        bandwidth_vec = np.array([1])
        thresh_bw = 0
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        expect_output = np.array([[1, 1, 1, 1, 1],
                                  [1, 1, 4, 4, 1],
                                  [2, 2, 2, 2, 1],
                                  [3, 3, 3, 3, 1],
                                  [4, 4, 4, 4, 1]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Big input without middle overlaps
        thresh_mat = np.array([[1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
                               [0, 0, 1, 0, 0, 1, 1, 0, 0, 1],
                               [1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                               [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
                               [0, 1, 1, 0, 0, 1, 0, 1, 1, 0],
                               [0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                               [1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
                               [0, 1, 0, 0, 0, 1, 1, 0, 1, 0],
                               [0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])

        bandwidth_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        thresh_bw = 0
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        expect_output = np.array([[6, 6, 9, 9, 1],
                                  [5, 6, 7, 8, 2],
                                  [7, 8, 9, 10, 2],
                                  [1, 3, 4, 6, 3],
                                  [1, 3, 8, 10, 3],
                                  [2, 4, 5, 7, 3],
                                  [2, 4, 6, 8, 3],
                                  [1, 10, 1, 10, 10]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Big input with middle overlaps
        thresh_mat = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                               [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                               [1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        bandwidth_vec = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        thresh_bw = 0
        output = find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

        expect_output = np.array([[1, 1, 10, 10, 1],
                                  [6, 6, 7, 7, 1],
                                  [7, 7, 11, 11, 1],
                                  [8, 8, 11, 11, 1],
                                  [3, 4, 9, 10, 2],
                                  [1, 13, 1, 13, 13]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_stretch_diags(self):
        """
        Tests if stretch_diags creates the correct binary matrix with full
        length diagonals from binary matrix of diagonal starts and length of
        diagonals.
        """

        thresh_diags = np.array([[0, 0, 1, 0, 0],
                                 [0, 1, 0, 0, 0],
                                 [0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]])
        band_width = 3
        output = stretch_diags(thresh_diags, band_width)

        expect_output = [[False, False,  True, False, False, False, False],
                         [False,  True, False,  True, False, False, False],
                         [False, False,  True, False,  True, False, False],
                         [False, False, False,  True, False, False, False],
                         [False, False, False, False,  True, False, False],
                         [False, False, False, False, False, False, False],
                         [False, False, False, False, False, False, False]]

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output)

    def test_add_annotations(self):
        """
        Tests if add_annotations adds the correct annotations to the pairs of
        repeats in input_mat.
        """

        # Input with annotations correctly marked
        input_mat = np.array([[1, 1, 2, 2, 1, 1],
                              [3, 6, 7, 10, 4, 2]])
        song_length = 16
        output = add_annotations(input_mat, song_length)
        expect_output = np.array([[1, 1, 2, 2, 1, 1],
                                  [3, 6, 7, 10, 4, 2]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Input with annotations wrongly marked
        input_mat = np.array([[1, 1, 2, 2, 1, 0],
                              [3, 6, 7, 10, 4, 0]])
        song_length = 16
        output = add_annotations(input_mat, song_length)
        expect_output = np.array([[1, 1, 2, 2, 1, 1],
                                  [3, 6, 7, 10, 4, 2]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_find_song_pattern(self):
        """
        Tests if __find_song_pattern correctly encodes the information in
        thresh_diags into a single-row song_pattern array.
        """

        thresh_diags = np.array([[1, 0, 0, 0, 0],
                                 [0, 1, 1, 1, 0],
                                 [0, 1, 1, 0, 0],
                                 [0, 1, 0, 1, 0],
                                 [0, 0, 0, 0, 1]])
        output = find_song_pattern(thresh_diags)

        expect_output = np.array([1, 2, 2, 2, 3])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_reconstruct_full_block(self):
        """
        Tests if reconstruct_full_block creates the correct full record
        of repeated structures, from the first beat of the song to the end.
        """

        # Input without overlaps
        pattern_mat = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])

        pattern_key = np.array([1, 2, 2, 3, 4])
        output = reconstruct_full_block(pattern_mat, pattern_key)

        expect_output = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                  [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
                                  [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                                  [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                                  [1, 1, 1, 1, 0, 0, 1, 1, 1, 1]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim, 2)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Input with overlaps
        pattern_mat = np.array([0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        pattern_key = np.array([3])
        output = reconstruct_full_block(pattern_mat, pattern_key)

        expect_output = np.array([[0, 1, 1, 1, 1, 2, 2, 1, 0, 0]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim, 2)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_get_annotation_lst(self):
        """
        Tests if get_annotation_lst creates the correct annotation marker
        vector given vector of repeat lengths key_lst.
        """

        # Input with small size, all length different
        key_lst = np.array([1, 2, 3, 4, 5])
        output = get_annotation_lst(key_lst)

        expect_output = np.array([1, 1, 1, 1, 1])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

        # Input with small size, all lengths equal
        key_lst = np.array([1, 1, 1, 1, 1])
        output = get_annotation_lst(key_lst)

        expect_output = np.array([1, 2, 3, 4, 5])

        # Input with big size
        key_lst = np.array([1, 1, 3, 4, 5, 5, 6, 6, 7, 7, 9, 11,
                            11, 11, 11, 11, 11, 11, 11, 12, 16, 16, 16, 16,
                            17, 17, 18, 20, 20, 20, 20])
        output = get_annotation_lst(key_lst)

        expect_output = np.array([1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 2, 3,
                                  4, 5, 6, 7, 8, 1, 1, 2, 3, 4, 1, 2, 1, 1,
                                  2, 3, 4])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output size
        self.assertEqual(np.size(output), np.size(expect_output))
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_get_y_labels(self):
        """
        Tests if get_y_labels generates the correct labels for a visualization
        with width_vec and anno_vec.
        """

        width_vec = np.array([[1], [1], [3], [4], [4], [5], [5], [6], [6]])
        anno_vec = np.array([1, 2, 1, 1, 2, 1, 1, 1, 2])
        expect_output = np.array(['0', 'w = 1, a = 1', 'w = 1, a = 2',
                                  'w = 3, a = 1', 'w = 4, a = 1',
                                  'w = 4, a = 2', 'w = 5, a = 1',
                                  'w = 5, a = 1', 'w = 6, a = 1',
                                  'w = 6, a = 2'])

        output = get_y_labels(width_vec, anno_vec)

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_reformat(self):
        """
        Tests if reformat generates the correct list of repeated structures
        that includes repeat lengths, where they start and end given a
        binary array with 1's where repeats start and 0's.
        """

        pattern_mat = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                                [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
                                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
        pattern_key = np.array([1, 2, 2, 3, 4])
        output = reformat(pattern_mat, pattern_key)

        expect_output = np.array([[5, 5, 10, 10, 1],
                                  [2, 3, 8, 9, 2],
                                  [3, 4, 9, 10, 2],
                                  [1, 3, 7, 9, 3],
                                  [1, 4, 7, 10, 4]])

        # Test output type
        self.assertIs(type(output), np.ndarray)
        # Test output format
        self.assertEqual(output.ndim, 2)
        # Test output result
        self.assertEqual(output.tolist(), expect_output.tolist())


if __name__ == '__main__':
    unittest.main()