#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, transform.py
"""

import unittest
import numpy as np

from repytah.transform import remove_overlaps
from repytah.transform import __create_anno_remove_overlaps \
    as create_anno_remove_overlaps
from repytah.transform import __separate_anno_markers \
    as separate_anno_markers


class TestTransform(unittest.TestCase):

    def test_create_anno_remove_overlaps_single_row_input(self):
        """
        Tests if __create_anno_remove_overlaps works with a single-row matrix.
        """

        input_mat = np.array([2, 2, 4, 4, 1, 1])
        song_length = 10
        band_width = 1

        expect_pattern_row = np.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        expect_k_lst_out = np.array([[2, 2, 4, 4, 1, 1]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_small_input_overlaps_only(self):
        """
        Tests if __create_anno_remove_overlaps works with a small matrix
        containing only overlaps.
        """

        input_mat = np.array([[1, 4, 11, 14, 4, 1],
                              [4, 7, 14, 17, 4, 1]])
        song_length = 20
        band_width = 4

        expect_pattern_row = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        )
        expect_k_lst_out = np.array([])
        expect_overlaps_lst = np.array([[1, 4, 11, 14, 4, 1],
                                        [4, 7, 14, 17, 4, 2]])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_large_input_no_overlaps(self):
        """
        Tests if __create_anno_remove_overlaps works with a large matrix
        containing no overlaps and having bandwidth larger than 1.
        """

        input_mat = np.array([[2, 3, 8, 9, 2, 1],
                              [2, 3, 15, 16, 2, 1],
                              [8, 9, 15, 16, 2, 1],
                              [3, 4, 9, 10, 2, 2],
                              [3, 4, 16, 17, 2, 2],
                              [9, 10, 16, 17, 2, 2],
                              [4, 5, 10, 11, 2, 3],
                              [4, 5, 17, 18, 2, 3],
                              [10, 11, 17, 18, 2, 3],
                              [7, 8, 14, 15, 2, 4],
                              [11, 12, 18, 19, 2, 5]])
        song_length = 19
        band_width = 2

        expect_pattern_row = np.array(
            [0, 1, 2, 3, 0, 0, 4, 1, 2, 3, 5, 0, 0, 4, 1, 2, 3, 5, 0]
        )
        expect_k_lst_out = np.array([[2, 3, 8, 9, 2, 1],
                                     [2, 3, 15, 16, 2, 1],
                                     [3, 4, 9, 10, 2, 2],
                                     [3, 4, 16, 17, 2, 2],
                                     [4, 5, 10, 11, 2, 3],
                                     [4, 5, 17, 18, 2, 3],
                                     [7, 8, 14, 15, 2, 4],
                                     [8, 9, 15, 16, 2, 1],
                                     [9, 10, 16, 17, 2, 2],
                                     [10, 11, 17, 18, 2, 3],
                                     [11, 12, 18, 19, 2, 5]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_large_input_no_overlaps_bw_1(self):
        """
        Tests if __create_anno_remove_overlaps works with a large matrix
        containing no overlaps and having bandwidth equal to 1.
        """

        input_mat = np.array([[8, 8, 14, 14, 1, 1],
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
                              [104, 104, 110, 110, 1, 1]])

        song_length = 119
        band_width = 1

        expect_pattern_row = np.array([
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ])

        expect_k_lst_out = np.array([[8, 8, 14, 14, 1, 1],
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
                                     [104, 104, 110, 110, 1, 1]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_wrong_bandwidth(self):
        """
        Tests if __create_anno_remove_overlaps works with a matrix
        when a non-existing bandwidth is given.
        """

        input_mat = np.array([[2, 3, 8, 9, 2, 1],
                              [2, 3, 15, 16, 2, 1],
                              [8, 9, 15, 16, 2, 1],
                              [3, 4, 9, 10, 2, 2],
                              [3, 4, 16, 17, 2, 2],
                              [9, 10, 16, 17, 2, 2],
                              [4, 5, 10, 11, 2, 3],
                              [4, 5, 17, 18, 2, 3],
                              [10, 11, 17, 18, 2, 3],
                              [7, 8, 14, 15, 2, 4],
                              [11, 12, 18, 19, 2, 5]])
        song_length = 19
        band_width = 3

        expect_pattern_row = np.array(
            [0, 1, 2, 3, 0, 0, 4, 1, 2, 3, 5, 0, 0, 4, 1, 2, 3, 5, 0]
        )
        expect_k_lst_out = np.array([[2, 3, 8, 9, 2, 1],
                                     [2, 3, 15, 16, 2, 1],
                                     [3, 4, 9, 10, 2, 2],
                                     [3, 4, 16, 17, 2, 2],
                                     [4, 5, 10, 11, 2, 3],
                                     [4, 5, 17, 18, 2, 3],
                                     [7, 8, 14, 15, 2, 4],
                                     [8, 9, 15, 16, 2, 1],
                                     [9, 10, 16, 17, 2, 2],
                                     [10, 11, 17, 18, 2, 3],
                                     [11, 12, 18, 19, 2, 5]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_some_overlaps(self):
        """
        Tests if __create_anno_remove_overlaps works with a matrix
        containing both overlapping and non-overlapping repeats.
        """

        input_mat = np.array([[2, 3, 8, 9, 2, 1],
                              [2, 3, 15, 16, 2, 1],
                              [8, 9, 15, 16, 2, 1],
                              [3, 4, 9, 10, 2, 1],
                              [3, 4, 16, 17, 2, 1],
                              [9, 10, 16, 17, 2, 1],
                              [4, 5, 10, 11, 2, 2],
                              [4, 5, 17, 18, 2, 2],
                              [10, 11, 17, 18, 2, 2],
                              [7, 8, 14, 15, 2, 3],
                              [11, 12, 18, 19, 2, 4]])
        song_length = 19
        band_width = 2

        expect_pattern_row = np.array(
            [0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 3, 0, 0, 2, 0, 0, 1, 3, 0]
        )
        expect_k_lst_out = np.array([[4, 5, 10, 11, 2, 1],
                                     [4, 5, 17, 18, 2, 1],
                                     [7, 8, 14, 15, 2, 2],
                                     [10, 11, 17, 18, 2, 1],
                                     [11, 12, 18, 19, 2, 3]])

        expect_overlaps_lst = np.array([[2, 3, 8, 9, 2, 1],
                                        [2, 3, 15, 16, 2, 1],
                                        [8, 9, 15, 16, 2, 1],
                                        [3, 4, 9, 10, 2, 2],
                                        [3, 4, 16, 17, 2, 2],
                                        [9, 10, 16, 17, 2, 2]])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_skipped_anno_small_input(self):
        """
        Tests that step 2 of __create_anno_remove_overlaps is able to check
        whether the annotation has a repeat associated to it for a small
        matrix.
        """

        input_mat = np.array([[2, 2, 8, 8, 1, 0],
                              [2, 2, 10, 10, 1, 1],
                              [3, 3, 4, 4, 1, 2],
                              [3, 3, 6, 6, 1, 2]])
        song_length = 10
        band_width = 1

        expect_pattern_row = np.array([0, 1, 2, 2, 0, 2, 0, 0, 0, 1])
        expect_k_lst_out = np.array([[2, 2, 8, 8, 1, 0],
                                     [2, 2, 10, 10, 1, 1],
                                     [3, 3, 4, 4, 1, 2],
                                     [3, 3, 6, 6, 1, 2]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_create_anno_remove_overlaps_skipped_anno_large_input(self):
        """
        Tests if step 2 of __create_anno_remove_overlaps is able to check
        whether the annotation has a repeat associated to it for a large
        matrix.
        """

        input_mat = np.array([[2, 2, 8, 8, 1, 1],
                              [2, 2, 15, 15, 1, 1],
                              [8, 8, 15, 15, 1, 1],
                              [3, 3, 5, 5, 1, 2],
                              [3, 3, 9, 9, 1, 2],
                              [5, 5, 9, 9, 1, 2],
                              [3, 3, 11, 11, 1, 2],
                              [5, 5, 11, 11, 1, 2],
                              [9, 9, 11, 11, 1, 2],
                              [3, 3, 16, 16, 1, 2],
                              [5, 5, 16, 16, 1, 2],
                              [9, 9, 16, 16, 1, 2],
                              [11, 11, 16, 16, 1, 2],
                              [3, 3, 18, 18, 1, 2],
                              [5, 5, 18, 18, 1, 2],
                              [9, 9, 18, 18, 1, 2],
                              [11, 11, 18, 18, 1, 2],
                              [16, 16, 18, 18, 1, 2],
                              [4, 4, 10, 10, 1, 3],
                              [4, 4, 17, 17, 1, 3],
                              [10, 10, 17, 17, 1, 3],
                              [7, 7, 14, 14, 1, 4],
                              [12, 12, 19, 19, 1, 6],
                              [2, 2, 12, 12, 1, 6]])
        song_length = 19
        band_width = 1

        expect_pattern_row = np.array(
            [0, 5, 2, 3, 2, 0, 4, 1, 2, 3, 2, 5, 0, 4, 1, 2, 3, 2, 5]
        )

        expect_k_lst_out = np.array([[2, 2, 8, 8, 1, 1],
                                     [2, 2, 12, 12, 1, 5],
                                     [2, 2, 15, 15, 1, 1],
                                     [3, 3, 5, 5, 1, 2],
                                     [3, 3, 9, 9, 1, 2],
                                     [3, 3, 11, 11, 1, 2],
                                     [3, 3, 16, 16, 1, 2],
                                     [3, 3, 18, 18, 1, 2],
                                     [4, 4, 10, 10, 1, 3],
                                     [4, 4, 17, 17, 1, 3],
                                     [5, 5, 9, 9, 1, 2],
                                     [5, 5, 11, 11, 1, 2],
                                     [5, 5, 16, 16, 1, 2],
                                     [5, 5, 18, 18, 1, 2],
                                     [7, 7, 14, 14, 1, 4],
                                     [8, 8, 15, 15, 1, 1],
                                     [9, 9, 11, 11, 1, 2],
                                     [9, 9, 16, 16, 1, 2],
                                     [9, 9, 18, 18, 1, 2],
                                     [10, 10, 17, 17, 1, 3],
                                     [11, 11, 16, 16, 1, 2],
                                     [11, 11, 18, 18, 1, 2],
                                     [12, 12, 19, 19, 1, 5],
                                     [16, 16, 18, 18, 1, 2]])
        expect_overlaps_lst = np.array([])

        pattern_row, k_lst_out, overlap_lst = create_anno_remove_overlaps(
            input_mat, song_length, band_width)

        self.assertTrue((pattern_row == expect_pattern_row).all())
        self.assertTrue((k_lst_out == expect_k_lst_out).all())
        self.assertTrue((overlap_lst == expect_overlaps_lst).all())

    def test_separate_anno_markers_single_row_input(self):
        """
        Tests if __separate_anno_markers works with a single-row matrix.
        """

        k_mat = np.array([[7, 12, 14, 19, 6, 1]])
        song_length = 19
        band_width = 6
        pattern_row = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        )

        expect_pattern_mat = np.array(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        )
        expect_pattern_key = np.array([6])
        expect_anno_id_lst = np.array([[1]])

        pattern_mat, pattern_key, anno_id_lst = separate_anno_markers(
            k_mat, song_length, band_width, pattern_row)

        self.assertTrue((pattern_mat == expect_pattern_mat).all())
        self.assertTrue((pattern_key == expect_pattern_key).all())
        self.assertTrue((anno_id_lst == expect_anno_id_lst).all())

    def test_separate_anno_markers_small_input(self):
        """
        Tests if __separate_anno_markers works with a small matrix.
        """

        k_mat = np.array([[3, 3, 9, 9, 1, 1],
                          [3, 3, 15, 15, 1, 1],
                          [5, 5, 12, 12, 1, 2]])
        song_length = 19
        band_width = 1
        pattern_row = np.array(
            [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0]
        )

        expect_pattern_mat = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        ])
        expect_pattern_key = np.array([[1],
                                       [1]])
        expect_anno_id_lst = np.array([[1],
                                       [2]])

        pattern_mat, pattern_key, anno_id_lst = separate_anno_markers(
            k_mat, song_length, band_width, pattern_row)

        self.assertTrue((pattern_mat == expect_pattern_mat).all())
        self.assertTrue((pattern_key == expect_pattern_key).all())
        self.assertTrue((anno_id_lst == expect_anno_id_lst).all())

    def test_separate_anno_markers_large_input(self):
        """
        Tests if __separate_anno_markers works with a large matrix.
        """

        k_mat = np.array([[2, 2, 8, 8, 1, 1],
                          [2, 2, 15, 15, 1, 1],
                          [3, 3, 5, 5, 1, 2],
                          [3, 3, 9, 9, 1, 2],
                          [3, 3, 11, 11, 1, 2],
                          [3, 3, 16, 16, 1, 2],
                          [3, 3, 18, 18, 1, 2],
                          [4, 4, 10, 10, 1, 3],
                          [4, 4, 17, 17, 1, 3],
                          [5, 5, 9, 9, 1, 2],
                          [5, 5, 11, 11, 1, 2],
                          [5, 5, 16, 16, 1, 2],
                          [5, 5, 18, 18, 1, 2],
                          [7, 7, 14, 14, 1, 4],
                          [8, 8, 15, 15, 1, 1],
                          [9, 9, 11, 11, 1, 2],
                          [9, 9, 16, 16, 1, 2],
                          [9, 9, 18, 18, 1, 2],
                          [10, 10, 17, 17, 1, 3],
                          [11, 11, 16, 16, 1, 2],
                          [11, 11, 18, 18, 1, 2],
                          [12, 12, 19, 19, 1, 5],
                          [16, 16, 18, 18, 1, 2]])
        song_length = 19
        band_width = 1
        pattern_row = np.array(
            [0, 1, 2, 3, 2, 0, 4, 1, 2, 3, 2, 5, 0, 4, 1, 2, 3, 2, 5]
        )

        expect_pattern_mat = np.array([
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
        ])

        expect_pattern_key = np.array([[1],
                                       [1],
                                       [1],
                                       [1],
                                       [1]])

        expect_anno_id_lst = np.array([[1],
                                       [2],
                                       [3],
                                       [4],
                                       [5]])

        pattern_mat, pattern_key, anno_id_lst = separate_anno_markers(
            k_mat, song_length, band_width, pattern_row)

        self.assertTrue((pattern_mat == expect_pattern_mat).all())
        self.assertTrue((pattern_key == expect_pattern_key).all())
        self.assertTrue((anno_id_lst == expect_anno_id_lst).all())

    def test_remove_overlaps_small_input_with_overlaps(self):
        """
        Tests if remove_overlaps works with a small matrix containing
        overlaps.
        """

        input_lst = np.array([[1, 4, 11, 14, 4, 1],
                              [4, 7, 14, 17, 4, 1],
                              [2, 3, 12, 13, 2, 1]])
        song_length = 20

        expect_lst_no_overlaps = np.array([[2, 3, 12, 13, 2, 1]])
        expect_matrix_no_overlaps = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        expect_key_no_overlaps = np.array([2])
        expect_annotations_no_overlaps = np.array([1])
        expect_all_overlap_lst = np.array([[1, 4, 11, 14, 4, 1],
                                           [4, 7, 14, 17, 4, 2]])

        lst_no_overlaps, matrix_no_overlaps, key_no_overlaps, \
            annotations_no_overlaps, all_overlap_lst = \
            remove_overlaps(input_lst, song_length)

        self.assertTrue((lst_no_overlaps == expect_lst_no_overlaps).all())
        self.assertTrue((matrix_no_overlaps == expect_matrix_no_overlaps).all())
        self.assertTrue((key_no_overlaps == expect_key_no_overlaps).all())
        self.assertTrue((annotations_no_overlaps ==
                         expect_annotations_no_overlaps).all())
        self.assertTrue((all_overlap_lst == expect_all_overlap_lst).all())

    def test_remove_overlaps_small_input_without_overlaps(self):
        """
        Tests if remove_overlaps works with a small matrix containing
        no overlaps.
        """

        input_lst = np.array([[1, 1, 10, 10, 1, 1],
                              [7, 7, 13, 13, 1, 1],
                              [3, 4, 17, 18, 2, 1]])
        song_length = 20

        expect_lst_no_overlaps = np.array([[1, 1, 10, 10, 1, 1],
                                           [7, 7, 13, 13, 1, 1],
                                           [3, 4, 17, 18, 2, 1]])
        expect_matrix_no_overlaps = np.array([
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        expect_key_no_overlaps = np.array([1, 2])
        expect_annotations_no_overlaps = np.array([1, 1])
        expect_all_overlap_lst = np.empty([0, 6])

        lst_no_overlaps, matrix_no_overlaps, key_no_overlaps,\
            annotations_no_overlaps, all_overlap_lst = \
            remove_overlaps(input_lst, song_length)

        self.assertTrue((lst_no_overlaps == expect_lst_no_overlaps).all())
        self.assertTrue((matrix_no_overlaps == expect_matrix_no_overlaps).all())
        self.assertTrue((key_no_overlaps == expect_key_no_overlaps).all())
        self.assertTrue((annotations_no_overlaps ==
                         expect_annotations_no_overlaps).all())
        self.assertTrue((all_overlap_lst == expect_all_overlap_lst).all())

    def test_remove_overlaps_large_input_with_overlaps(self):
        """
        Tests if remove_overlaps works with a large matrix containing
        overlaps.
        """

        input_lst = np.array([[1, 2, 8, 9, 2, 1],
                              [2, 3, 9, 10, 2, 1],
                              [1, 2, 2, 3, 2, 1],
                              [16, 17, 18, 19, 2, 1],
                              [16, 17, 9, 10, 2, 1],
                              [3, 4, 10, 11, 2, 2],
                              [4, 6, 11, 13, 3, 1],
                              [5, 7, 13, 15, 3, 1],
                              [5, 7, 11, 13, 3, 1]])
        song_length = 20

        expect_lst_no_overlaps = np.array([[3, 4, 10, 11, 2, 1]])
        expect_matrix_no_overlaps = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        expect_key_no_overlaps = np.array([2])
        expect_annotations_no_overlaps = np.array([1])
        expect_all_overlap_lst = np.array([[4, 6, 11, 13, 3, 1],
                                           [5, 7, 11, 13, 3, 1],
                                           [5, 7, 13, 15, 3, 1],
                                           [1, 2, 2, 3, 2, 1],
                                           [1, 2, 8, 9, 2, 1],
                                           [2, 3, 9, 10, 2, 1],
                                           [16, 17, 9, 10, 2, 1],
                                           [16, 17, 18, 19, 2, 1]])

        lst_no_overlaps, matrix_no_overlaps, key_no_overlaps,\
            annotations_no_overlaps, all_overlap_lst = \
            remove_overlaps(input_lst, song_length)

        self.assertTrue((lst_no_overlaps == expect_lst_no_overlaps).all())
        self.assertTrue((matrix_no_overlaps == expect_matrix_no_overlaps).all())
        self.assertTrue((key_no_overlaps == expect_key_no_overlaps).all())
        self.assertTrue((annotations_no_overlaps ==
                         expect_annotations_no_overlaps).all())
        self.assertTrue((all_overlap_lst == expect_all_overlap_lst).all())

    def test_remove_overlaps_large_input_without_overlaps(self):
        """
        Tests if remove_overlaps works with a large matrix containing
        no overlaps.
        """

        input_lst = np.array([[2, 2, 8, 8, 1, 1],
                              [2, 2, 15, 15, 1, 1],
                              [3, 3, 5, 5, 1, 2],
                              [3, 3, 9, 9, 1, 2],
                              [3, 3, 11, 11, 1, 2],
                              [3, 3, 16, 16, 1, 2],
                              [3, 3, 18, 18, 1, 2],
                              [4, 4, 10, 10, 1, 3],
                              [4, 4, 17, 17, 1, 3],
                              [5, 5, 9, 9, 1, 2],
                              [5, 5, 11, 11, 1, 2],
                              [5, 5, 16, 16, 1, 2],
                              [5, 5, 18, 18, 1, 2],
                              [7, 7, 14, 14, 1, 4],
                              [8, 8, 15, 15, 1, 1],
                              [9, 9, 11, 11, 1, 2],
                              [9, 9, 16, 16, 1, 2],
                              [9, 9, 18, 18, 1, 2],
                              [10, 10, 17, 17, 1, 3],
                              [11, 11, 16, 16, 1, 2],
                              [11, 11, 18, 18, 1, 2],
                              [12, 12, 19, 19, 1, 5],
                              [16, 16, 18, 18, 1, 2],
                              [2, 3, 8, 9, 2, 1],
                              [2, 3, 15, 16, 2, 1],
                              [3, 4, 9, 10, 2, 2],
                              [3, 4, 16, 17, 2, 2],
                              [4, 5, 10, 11, 2, 3],
                              [4, 5, 17, 18, 2, 3],
                              [7, 8, 14, 15, 2, 4],
                              [8, 9, 15, 16, 2, 1],
                              [9, 10, 16, 17, 2, 2],
                              [10, 11, 17, 18, 2, 3],
                              [11, 12, 18, 19, 2, 5],
                              [2, 4, 8, 10, 3, 1],
                              [2, 4, 15, 17, 3, 1],
                              [3, 5, 9, 11, 3, 2],
                              [3, 5, 16, 18, 3, 2],
                              [7, 9, 14, 16, 3, 3],
                              [8, 10, 15, 17, 3, 1],
                              [9, 11, 16, 18, 3, 2],
                              [10, 12, 17, 19, 3, 4],
                              [2, 5, 8, 11, 4, 1],
                              [2, 5, 15, 18, 4, 1],
                              [7, 10, 14, 17, 4, 2],
                              [8, 11, 15, 18, 4, 1],
                              [9, 12, 16, 19, 4, 3],
                              [7, 11, 14, 18, 5, 1],
                              [8, 12, 15, 19, 5, 2],
                              [7, 12, 14, 19, 6, 1]])

        song_length = 20

        expect_lst_no_overlaps = np.array([[2, 2, 8, 8, 1, 1],
                                           [2, 2, 15, 15, 1, 1],
                                           [3, 3, 5, 5, 1, 2],
                                           [3, 3, 9, 9, 1, 2],
                                           [3, 3, 11, 11, 1, 2],
                                           [3, 3, 16, 16, 1, 2],
                                           [3, 3, 18, 18, 1, 2],
                                           [4, 4, 10, 10, 1, 3],
                                           [4, 4, 17, 17, 1, 3],
                                           [5, 5, 9, 9, 1, 2],
                                           [5, 5, 11, 11, 1, 2],
                                           [5, 5, 16, 16, 1, 2],
                                           [5, 5, 18, 18, 1, 2],
                                           [7, 7, 14, 14, 1, 4],
                                           [8, 8, 15, 15, 1, 1],
                                           [9, 9, 11, 11, 1, 2],
                                           [9, 9, 16, 16, 1, 2],
                                           [9, 9, 18, 18, 1, 2],
                                           [10, 10, 17, 17, 1, 3],
                                           [11, 11, 16, 16, 1, 2],
                                           [11, 11, 18, 18, 1, 2],
                                           [12, 12, 19, 19, 1, 5],
                                           [16, 16, 18, 18, 1, 2],
                                           [2, 3, 8, 9, 2, 1],
                                           [2, 3, 15, 16, 2, 1],
                                           [3, 4, 9, 10, 2, 2],
                                           [3, 4, 16, 17, 2, 2],
                                           [4, 5, 10, 11, 2, 3],
                                           [4, 5, 17, 18, 2, 3],
                                           [7, 8, 14, 15, 2, 4],
                                           [8, 9, 15, 16, 2, 1],
                                           [9, 10, 16, 17, 2, 2],
                                           [10, 11, 17, 18, 2, 3],
                                           [11, 12, 18, 19, 2, 5],
                                           [2, 4, 8, 10, 3, 1],
                                           [2, 4, 15, 17, 3, 1],
                                           [3, 5, 9, 11, 3, 2],
                                           [3, 5, 16, 18, 3, 2],
                                           [7, 9, 14, 16, 3, 3],
                                           [8, 10, 15, 17, 3, 1],
                                           [9, 11, 16, 18, 3, 2],
                                           [10, 12, 17, 19, 3, 4],
                                           [2, 5, 8, 11, 4, 1],
                                           [2, 5, 15, 18, 4, 1],
                                           [7, 10, 14, 17, 4, 2],
                                           [8, 11, 15, 18, 4, 1],
                                           [9, 12, 16, 19, 4, 3],
                                           [7, 11, 14, 18, 5, 1],
                                           [8, 12, 15, 19, 5, 2],
                                           [7, 12, 14, 19, 6, 1]])

        expect_matrix_no_overlaps = np.array([
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        ])

        expect_key_no_overlaps = np.array(
            [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 6]
        )
        expect_annotations_no_overlaps = np.array(
            [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 1, 2, 1]
        )
        expect_all_overlap_lst = np.empty([0, 6])

        lst_no_overlaps, matrix_no_overlaps, key_no_overlaps,\
            annotations_no_overlaps, all_overlap_lst \
            = remove_overlaps(input_lst, song_length)

        self.assertTrue((lst_no_overlaps == expect_lst_no_overlaps).all())
        self.assertTrue((matrix_no_overlaps == expect_matrix_no_overlaps).all())
        self.assertTrue((key_no_overlaps == expect_key_no_overlaps).all())
        self.assertTrue((annotations_no_overlaps ==
                         expect_annotations_no_overlaps).all())
        self.assertTrue((all_overlap_lst == expect_all_overlap_lst).all())


if __name__ == '__main__':
    unittest.main()
