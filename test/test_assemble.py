# -*- coding: utf-8 -*-
"""
Unit tests for Aligned Hierarchies, assemble.py
"""

import unittest
import numpy as np

from repytah.assemble import breakup_overlaps_by_intersect
from repytah.assemble import check_overlaps
from repytah.assemble import __compare_and_cut as compare_and_cut
from repytah.assemble import __num_of_parts as num_of_parts
from repytah.assemble import __inds_to_rows as inds_to_rows
from repytah.assemble import __merge_based_on_length as merge_based_on_length
from repytah.assemble import __merge_rows as merge_rows
from repytah.assemble import hierarchical_structure


class TestAssemble(unittest.TestCase):

    def test_breakup_overlaps_by_intersect(self):
        """
        Tests if breakup_overlap_by_intersect gives the correct output 
        accessible via a tuple for an example.
        """

        input_pattern_obj = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            ])
        bw_vec = np.array([[3],
                           [5],
                           [8],
                           [8]])
        thresh_bw = 0

        output = breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, 
                                               thresh_bw)

        expect_output0 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            ])

        expect_output1 = np.array([[3],
                                   [5]])
        
        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1].tolist(), expect_output1.tolist())

    def test_check_overlaps(self):
        """
        Tests if check_overlaps gives the correct output with the correct data
        type and size for an example case.
        """

        input_mat = np.array([[1, 1, 0, 1, 0, 0],
                              [1, 1, 1, 0, 1, 0],
                              [0, 1, 1, 0, 0, 1],
                              [1, 0, 0, 1, 0, 0], 
                              [0, 1, 0, 0, 1, 0], 
                              [0, 0, 1, 0, 0, 1]])
        
        expect_output = np.array([[0, 1, 1, 1, 1, 0],
                                  [0, 0, 1, 1, 1, 1],
                                  [0, 0, 0, 0, 1, 1],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0]])
        
        output = check_overlaps(input_mat)

        self.assertIs(type(output), np.ndarray)
        self.assertEqual(np.size(output), np.size(expect_output))
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_compare_and_cut(self):
        """
        Tests if __compare_and_cut gives the correct output accessible via a
        tuple for an example.
        """

        red = np.array(
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            )
        red_len = np.array([5])
        blue = np.array(
            [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            )
        blue_len = np.array([3])

        output = compare_and_cut(red, red_len, blue, blue_len)

        expect_output0 = np.array([
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            ])

        expect_output1 = np.array([[1],
                                   [1],
                                   [2]])

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1].tolist(), expect_output1.tolist())

    def test_num_of_parts_if_statement(self):
        """
        Tests if __num_of_parts gives the correct output accessible via a tuple 
        for an example when the if clause is entered
        (i.e. if the input vector has no breaks).
        """

        input_vec = np.array([3, 4])
        input_start = np.array([0])
        input_all_starts = np.array([3, 7, 10])

        expect_output0 = np.array([6, 10, 13])
        expect_output1 = 2

        output = num_of_parts(input_vec, input_start, input_all_starts)

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1], expect_output1)

    def test_num_of_parts_else_statement(self):
        """
        Tests if __num_of_parts gives the correct output accessible via a tuple 
        for an example case when the else clause is entered 
        (i.e. if the input vector has a break).
        """

        input_vec = np.array([3, 5])
        input_start = np.array([3])
        input_all_starts = np.array([3, 7, 10])

        expect_output0 = np.array([[3, 7, 10],
                                   [5, 9, 12]])

        expect_output1 = np.array([[1],
                                   [1]])

        output = num_of_parts(input_vec, input_start, input_all_starts)

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1].tolist(), expect_output1.tolist())

    def test_inds_to_rows(self):
        """
        Tests if __inds_to_rows gives the correct output with the correct data
        type and size for an example case.
        """

        start_mat = np.array([0, 1, 6, 7])
        row_length = 10

        expect_output = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0]])

        output = inds_to_rows(start_mat, row_length)

        self.assertIs(type(output), np.ndarray)
        self.assertEqual(np.size(output), np.size(expect_output))
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_merge_based_on_length(self):
        """
        Tests if __merge_based_on_length gives the correct output accessible 
        via a tuple for an example case.
        """

        full_mat = np.array([
            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
            ])

        full_bw = np.array([[2],
                            [2]])

        target_bw = np.array([[2],
                              [2]])

        output = merge_based_on_length(full_mat, full_bw, target_bw)

        expect_output0 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0]
            ])
        expect_output1 = np.array([2])

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1].tolist(), expect_output1.tolist())

    def test_merge_rows(self):
        """
        Tests if __merge_rows gives the correct output with the correct data
        type and size for an example case.
        """

        input_mat = np.array([
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
            ])
        input_width = np.array([1])

        output = merge_rows(input_mat, input_width)

        expect_output = np.array([
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0]
            ])

        self.assertIs(type(output), np.ndarray)
        self.assertEqual(np.size(output), np.size(expect_output))
        self.assertEqual(output.tolist(), expect_output.tolist())

    def test_hierarchical_structure(self):
        """
        Tests if hierarchical_structure gives the correct output accessible via
        a tuple for an example case.
        """

        input_matrix_no = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        input_key_no = np.array([[5],
                                 [10]])
        input_sn = 20

        output = hierarchical_structure(input_matrix_no, input_key_no, 
                                        input_sn)

        expect_output0 = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
            ])
        expect_output1 = np.array([[5]])
        expect_output2 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            ])

        self.assertIs(type(output), tuple)
        self.assertEqual(output[0].tolist(), expect_output0.tolist())
        self.assertEqual(output[1].tolist(), expect_output1.tolist())
        self.assertEqual(output[2].tolist(), expect_output2.tolist())

    def test_hierarchical_structure_equal_with_boolean(self):
        """
        Tests if hierarchical_structure gives the same output for vis=True 
        and vis=False as visualizations are just shown.
        """

        input_matrix_no = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        input_key_no = np.array([[5],
                                 [10]])
        input_sn = 20

        output_false = hierarchical_structure(input_matrix_no, input_key_no, 
                                              input_sn)  # default vis=False
        output_true = hierarchical_structure(input_matrix_no, input_key_no, 
                                             input_sn, vis=True)

        self.assertEqual(output_false[0].tolist(), output_true[0].tolist())
        self.assertEqual(output_false[1].tolist(), output_true[1].tolist())
        self.assertEqual(output_false[2].tolist(), output_true[2].tolist())
        self.assertEqual(output_false[3].tolist(), output_true[3].tolist())


if __name__ == '__main__':
    unittest.main()