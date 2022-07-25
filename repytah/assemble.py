#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
assemble.py 

This module finds and forms essential structure components, which are the 
smallest building blocks that form every repeat in the song. 

These functions ensure that each time step of a song is contained in at most 
one of the song's essential structure components by checking that there are no 
overlapping repeats in time. When repeats overlap, they undergo a process 
where they are divided until there are only non-overlapping pieces left. 

The module contains the following functions:

    * breakup_overlaps_by_intersect
        Extracts repeats in input_pattern_obj that has the starting indices 
        of the repeats, into the essential structure components using bw_vec,
        that has the lengths of each repeat.
    
    * check_overlaps 
        Compares every pair of groups, determining if there are any repeats 
        in any pairs of the groups that overlap. 

    * __compare_and_cut 
        Compares two rows of repeats labeled RED and BLUE, and determines if
        there are any overlaps in time between them. If there are overlaps, 
        we cut the repeats in RED and BLUE into up to 3 pieces. 

    * __num_of_parts
        Determines the number of blocks of consecutive time steps in a list 
        of time steps. A block of consecutive time steps represents a 
        distilled section of a repeat.    

    * __inds_to_rows 
        Expands a vector containing the starting indices of a piece or two 
        of a repeat into a matrix representation recording when these pieces 
        occur in the song with 1's. All remaining entries are marked with 
        0's.

    * __merge_based_on_length
        Merges repeats that are the same length, as set by full_bandwidth, 
        and are repeats of the same piece of structure.

    * __merge_rows 
        Merges rows that have at least one common repeat. These common 
        repeat(s) must occur at the same time step and be of a common length.

    * hierarchical_structure
        Distills the repeats encoded in matrix_no_overlaps (and key_no_overlaps) 
        to the essential structure components and then builds the hierarchical 
        representation. Optionally outputs visualizations of the hierarchical 
        representations.
    
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

from .search import find_all_repeats, find_complete_list_anno_only
from .utilities import reconstruct_full_block, get_annotation_lst, get_y_labels
from .transform import remove_overlaps


def breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, thresh_bw):
    """
    Extracts repeats in input_pattern_obj that has the starting indices of the
    repeats, into the essential structure components using bw_vec, that has the
    lengths of each repeat. The essential structure components are the
    smallest building blocks that form every repeat in the song.

    Args:
        input_pattern_obj (np.ndarray):
            Binary matrix with 1's where repeats begin and 0's otherwise.

        bw_vec (np.ndarray):
            Vector containing the lengths of the repeats encoded in
            input_pattern_obj.

        thresh_bw (int):
            One less than the smallest allowable repeat length.

    Returns:
        A tuple (pattern_no_overlaps, pattern_no_overlaps_key) where all
        variables have data type np.ndarray.

        pattern_no_overlaps is a binary matrix with 1's where repeats
        of essential structure components begin.

        pattern_no_overlaps_key is a vector containing the lengths of the
        repeats of essential structure components in pattern_no_overlaps.
    """
    # Make sure bw_vec is a 2D vector
    if bw_vec.ndim == 1:
        # Convert a 1D array into 2D vector
        bw_vec = bw_vec[None, :].reshape(-1, 1)

    # Sort bw_vec and input_pattern_obj so that we process the
    # biggest pieces first

    # Sort the indices of bw_vec in descending order
    row_bw_inds = np.argsort(bw_vec, axis=None)[::-1]

    # Sort input_pattern_obj and bw_vec respectively
    input_pattern_obj = input_pattern_obj[row_bw_inds, :]
    desc_bw_vec = bw_vec[row_bw_inds, :]

    # Get the index of where bw_vec is equal to thresh_bw for
    # future removal process
    thresh_inds = np.nonzero(desc_bw_vec == thresh_bw)[0]
    if thresh_inds.size == 0:
        thresh_inds = max(bw_vec.shape)
    else:
        thresh_inds = thresh_inds[0]

    pno_block = reconstruct_full_block(input_pattern_obj, desc_bw_vec)

    # Check stopping condition -- Are there overlaps?
    while np.sum(np.sum(pno_block[:thresh_inds, :], axis=0) > 1) > 0:
        # Find all overlaps by comparing the rows of repeats pairwise
        overlaps_pno_block = check_overlaps(pno_block)

        # Remove the rows with bandwidth thresh_bw or less from consideration
        overlaps_pno_block[thresh_inds:, ] = 0
        overlaps_pno_block[:, thresh_inds:] = 0

        # Find the first two groups of repeats that overlap, calling one group
        # RED and the other group BLUE
        [ri, bi] = overlaps_pno_block.nonzero()
        ri = ri[0]
        bi = bi[0]

        # RED overlap
        red = input_pattern_obj[ri, :]
        RL = desc_bw_vec[ri, :]

        # BLUE overlap
        blue = input_pattern_obj[bi, :]
        BL = desc_bw_vec[bi, :]

        # Compare the repeats in RED and BLUE, cutting the repeats in those
        # groups into non-overlapping pieces
        union_mat, union_length = __compare_and_cut(red, RL, blue, BL)
        input_pattern_obj = np.delete(input_pattern_obj, [ri, bi], axis=0)
        bw_vec = np.delete(desc_bw_vec, [ri, bi], axis=0)

        # Stack the new repeats
        if union_mat.size != 0:
            input_pattern_obj = np.vstack((input_pattern_obj, union_mat))
            bw_vec = np.vstack((bw_vec, union_length))

        # Check if there are any repeats of length 1 that should be merged into
        # other groups of repeats of length 1 and merge them if necessary
        if sum(union_length == 1) > 0:

            input_pattern_obj, bw_vec = __merge_based_on_length(
                input_pattern_obj, bw_vec, np.array(1))

        # AGAIN, Sort bw_vec and input_pattern_obj in descending order
        # so that we process the biggest pieces first
        row_bw_inds = np.argsort(bw_vec, axis=None)[::-1]
        input_pattern_obj = input_pattern_obj[row_bw_inds, :]
        desc_bw_vec = bw_vec[row_bw_inds, :]

        # Find the first row that contains repeats of length less than T and
        # remove these rows from consideration during the next check of the
        # stopping condition
        thresh_inds = np.nonzero(desc_bw_vec == thresh_bw)[0]
        if thresh_inds.size == 0:
            thresh_inds = max(desc_bw_vec.shape)
        else:
            thresh_inds = thresh_inds[0]

        pno_block = reconstruct_full_block(input_pattern_obj, desc_bw_vec)

    # Sort the lengths in bw_vec in ascending order
    # and sort pattern_no_overlaps respectively
    bw_inds = np.argsort(desc_bw_vec, axis=None)
    pattern_no_overlaps_key = desc_bw_vec[bw_inds, ]
    pattern_no_overlaps = input_pattern_obj[bw_inds, ]

    return pattern_no_overlaps, pattern_no_overlaps_key


def check_overlaps(input_mat):
    """
    Compares every pair of repeat groups and determines if there are any repeats
    in any pairs of the groups that overlap.

    Args:
        input_mat (np.ndarray):
            Binary matrix with blocks of 1's equal to the length of repeats
            to be checked for overlaps.

    Returns:
        overlap_mat (np.ndarray):
            Logical array where (i,j) = 1 if row i of input_mat and row j
            of input_mat overlap and (i,j) = 0 elsewhere.
            
    """

    # Get the number of rows and columns
    rs = input_mat.shape[0]
    ws = input_mat.shape[1]

    # compare_left -- Every row of input_mat is repeated rs times to create 
    # a sub-matrix. We stack these sub-matrices on top of each other.
    compare_left = np.zeros(((rs * rs), ws))

    for i in range(rs):
        compare_add = input_mat[i, :]
        compare_add_mat = np.tile(compare_add, (rs, 1))
        a = i * rs
        b = (i + 1) * rs
        compare_left[a:b, :] = compare_add_mat

    # compare_right -- Stack rs copies of input_mat on top of itself
    compare_right = np.tile(input_mat, (rs, 1))

    # If input_mat is not binary, create binary temporary objects
    compare_left = compare_left > 0
    compare_right = compare_right > 0
    # Check every pair of rows to see which rows overlap with each other
    compare_all = np.sum(np.add(compare_left.astype(int),
                                compare_right.astype(int)) == 2,
                         axis=1) > 0
    # Convert compare_all to a 2D vector
    compare_all = compare_all[None, :].reshape(-1, 1)
    
    overlap_mat = np.reshape(compare_all, (rs, rs))

    # If overlap_mat is symmetric, only keep the upper-triangular portion. 
    # If not, keep all of overlap_mat.
    check_mat = np.allclose(overlap_mat, overlap_mat.T)

    if check_mat:
        overlap_mat = np.triu(overlap_mat, 1)

    return overlap_mat


def __compare_and_cut(red, red_len, blue, blue_len):
    """
    Compares two rows of repeats labeled RED and BLUE, and determines if there
    are any overlaps in time between them. If there is, then we cut the
    repeats in RED and BLUE into up to 3 pieces.

    Args:
        red (np.ndarray):
            Binary row vector encoding a set of repeats with 1's where each
            repeat starts and 0's otherwise.

        red_len (np.ndarray):
            Length of repeats encoded in red.

        blue (np.ndarray):
            Binary row vector encoding a set of repeats with 1's where each
            repeat starts and 0's otherwise.

        blue_len (np.ndarray):
            Length of repeats encoded in blue.

    Returns:
        A tuple (union_mat, union_length) where all variables have data type
        np.ndarray.

        union_mat is a binary matrix representation of up to three rows encoding
        non-overlapping repeats cut from red and blue.

        union_length is a vector containing the lengths of the repeats encoded
        in union_mat.
    """

    # Find the total time steps in red
    sn = red.shape[0]
    assert sn == blue.shape[0]

    # Find all starting indices in red and store them as a 2D array
    start_red = np.flatnonzero(red)
    start_red = start_red[None, :]

    # Find all starting indices in blue and store them as a 2D array
    start_blue = np.flatnonzero(blue)
    start_blue = start_blue[None, :]

    # Determine if the rows have any intersections
    red_block = reconstruct_full_block(red, red_len)
    blue_block = reconstruct_full_block(blue, blue_len)

    # Find the intersection of red and blue and call it purple
    red_block = red_block > 0
    blue_block = blue_block > 0
    purple_block = np.logical_and(red_block, blue_block)

    # If there is any intersection between the rows, then start comparing one
    # repeat in red to one repeat in blue
    if purple_block.sum() > 0:

        # Find the number of blocks in red and in blue
        lsr = max(start_red.shape)
        lsb = max(start_blue.shape)

        # Build the pairs of starting indices to search, where each pair
        # contains a starting index in red and a starting index in blue
        red_inds = np.tile(start_red.transpose(), (lsb, 1))
        blue_inds = np.tile(start_blue, (lsr, 1))
        tem_blue = blue_inds[:, 0].reshape(-1, 1)
        for i in range(1, blue_inds.shape[1]):
            col = blue_inds[:, i].reshape(-1,1)
            tem_blue = np.vstack((tem_blue, col))
        compare_inds = np.concatenate((tem_blue, red_inds), axis=1)

        # Initialize the output variables union_mat and union_length
        union_mat = np.array([])
        union_length = np.array([])

        # Loop over all pairs of starting indices
        for start_ind in range(0, lsr * lsb):

            # Isolate one repeat in red and one repeat in blue
            ri = compare_inds[start_ind, 1]
            bi = compare_inds[start_ind, 0]

            red_ri = np.arange(ri, ri + red_len)
            blue_bi = np.arange(bi, bi + blue_len)

            # Determine if the blocks intersect and call the intersection
            # purple
            purple = np.intersect1d(red_ri, blue_bi)

            if purple.shape[0] != 0:
                # Remove purple from red_ri, call it red_minus_purple
                red_minus_purple = np.setdiff1d(red_ri, purple)

                # If red_minus_purple is not empty, then see if there are one
                # or two parts in red_minus_purple.
                # Then cut purple out of all of the repeats in red.
                if red_minus_purple.size != 0:
                    # red_length_vec will have the length(s) of the parts in
                    # new_red
                    red_start_mat, red_length_vec = __num_of_parts(
                        red_minus_purple, ri, start_red
                    )

                    # If there are two parts left in red_minus_purple, then
                    # the new variable new_red, which holds the part(s) of
                    # red_minus_purple, should have two rows with 1's for the
                    # starting indices of the resulting pieces and 0's
                    # elsewhere.
                    new_red = __inds_to_rows(red_start_mat, sn)

                else:
                    # If red_minus_purple is empty, then set new_red and
                    # red_length_vec to empty
                    new_red = np.array([])
                    red_length_vec = np.array([])

                # Noting that purple is only one part and in both red_ri and
                # blue_bi, then we need to find where the purple starting
                # indices are in all the red_ri
                purple_in_red_mat, purple_length_vec = __num_of_parts(
                    purple, ri, start_red
                )
                blue_minus_purple = np.setdiff1d(blue_bi, purple)

                # If blue_minus_purple is not empty, then see if there are one
                # or two parts in blue_minus_purple. Then cut purple out of
                # all of the repeats in blue.
                if blue_minus_purple.size != 0:
                    blue_start_mat, blue_length_vec = __num_of_parts(
                        blue_minus_purple, bi, start_blue
                    )
                    new_blue = __inds_to_rows(blue_start_mat, sn)

                # If there are two parts left in blue_minus_purple, then the
                # new variable new_blue, which holds the part(s) of
                # blue_minus_purple, should have two rows with 1's for the
                # starting indices of the resulting pieces and 0's elsewhere.
                else:
                    # If blue_minus_purple is empty, then set new_blue and
                    # blue_length_vec to empty
                    new_blue = np.array([])
                    # Also blue_length_vec will have the length(s) of the
                    # parts in new_blue.
                    blue_length_vec = np.array([])

                # Recalling that purple is only one part and in both red_rd
                # and blue_bi, then we need to find where the purple starting
                # indices are in all the blue_ri
                purple_in_blue_mat, purple_length = __num_of_parts(
                    purple, bi, start_blue
                )

                # Union purple_in_red_mat and purple_in_blue_mat to get
                # purple_start, which stores all the purple indices
                purple_start = np.union1d(purple_in_red_mat[0], 
                                          purple_in_blue_mat[0])

                # Use purple_start to get new_purple with 1's where the repeats
                # in the purple rows start and 0 otherwise.
                new_purple = __inds_to_rows(purple_start, sn)
                if new_red.size != 0 or new_blue.size != 0:

                    # Form the outputs
                    # Use the condition check to avoid errors when stacking
                    # an empty array
                    if new_red.size != 0 and new_blue.size == 0:
                        union_mat = np.vstack((new_red, new_purple))
                        union_length = np.vstack((red_length_vec, 
                                                  purple_length))
                    elif new_red.size == 0 and new_blue.size != 0:
                        union_mat = np.vstack((new_blue, new_purple))
                        union_length = np.vstack((blue_length_vec, 
                                                  purple_length))
                    else:
                        union_mat = np.vstack((new_red, new_blue, new_purple))
                        union_length = np.vstack(
                            (red_length_vec, blue_length_vec, purple_length)
                        )

                    # Merge repeats that are the same length
                    union_mat, union_length = __merge_based_on_length(
                        union_mat, union_length, union_length
                    )

                    # When we find union_mat and union_length in this group,
                    # we break out of the for loop to add them to our final
                    # output
                    break

                elif new_red.size == 0 and new_blue.size == 0:
                    new_purple_block = reconstruct_full_block(
                        new_purple, np.array([purple_length])
                    )
                    # Only add the new repeat which has no overlaps
                    if max(new_purple_block[0]) < 2:
                        union_mat = new_purple
                        union_length = np.array([purple_length])
                        break

    # Check that there are no overlaps in each row of union_mat
    union_mat_add = np.empty((0, sn), int)
    union_mat_add_length = np.empty((0, 1), int)
    union_mat_rminds = np.empty((0, 1), int)

    # Isolate one row at a time, call it union_row
    for i in range(0, union_mat.shape[0]):
        union_row = union_mat[i, :]
        union_row_width = np.array([union_length[i]])
        union_row_block = reconstruct_full_block(union_row, union_row_width)

        # If there is at least one overlap, then compare and cut that row
        # until there are no overlaps
        if (np.sum(union_row_block[0] > 1)) > 0:
            union_mat_rminds = np.vstack((union_mat_rminds, i))
            union_row_new, union_row_new_length = __compare_and_cut(
                union_row, union_row_width, union_row, union_row_width
            )

            # Add union_row_new and union_row_new_length to union_mat_add and
            # union_mat_add_length, respectively
            union_mat_add = np.vstack((union_mat_add, union_row_new))
            union_mat_add_length = np.vstack(
                (union_mat_add_length, union_row_new_length)
            )

    # Remove the old rows from union_mat (as well as the old lengths from
    # union_length)
    if union_mat_rminds.size != 0:
        union_mat = np.delete(union_mat, union_mat_rminds, axis=0)
        union_length = np.delete(union_length, union_mat_rminds)

    # Add union_row_new and union_row_new_length to union_mat and
    # union_length, respectively, such that union_mat is in order by
    # lengths in union_length
    if union_mat_add.size != 0:
        union_mat = np.vstack((union_mat, union_mat_add))
    if union_mat_add_length.size != 0:
        union_length = np.vstack((np.array([union_length]).T, 
                                  union_mat_add_length))

    # Make sure union_length is a 2D vector
    if union_length.ndim == 1:
        union_length = np.array([union_length]).T
    if union_mat.size != 0:
        # Sort the union_mat and union_length in ascending order
        inds = np.argsort(union_length, axis=None)
        union_mat = union_mat[inds, :]
        union_length = union_length[inds, :]

    return union_mat, union_length


def __num_of_parts(input_vec, input_start, input_all_starts):
    """
    Determines the number of blocks of consecutive time steps in a list of 
    time steps. A block of consecutive time steps represents a distilled 
    section of a repeat. This distilled section will be replicated and the 
    starting indices of the repeats within it will be returned.

    Args:
        input_vec (np.ndarray):
            Vector that contains one or two parts of a repeat that are
            overlap(s) in time that may need to be replicated.

        input_start (np.ndarray):
            Starting index for the part to be replicated.

        input_all_starts (np.ndarray):
            Starting indices for replication.

    Returns:
        A tuple (start_mat, length_vec) where all variables have data type
        np.ndarray.

        start_mat is an array of one or two rows containing the starting indices
        of the replicated repeats.

        length_vec is a column vector containing the lengths of the replicated
        parts.
    """

    # Determine where input_vec has a break
    diff_vec = np.subtract(input_vec[1:], input_vec[:-1])
    diff_vec = np.insert(diff_vec, 0, 1)
    break_mark = np.where(diff_vec > 1)[0]

    # If input_vec is consecutive
    if sum(break_mark) == 0:
        # Initialize start_vec and end_vec
        start_vec = input_vec[0]
        end_vec = input_vec[-1]

        # Find the difference between the starts
        add_vec = start_vec - input_start

        # Find the new start of the distilled section
        start_mat = input_all_starts + add_vec

    # Else if input_vec has a break
    else:
        # Initialize start_vec and end_vec
        start_vec = np.zeros((2, 1))
        end_vec = np.zeros((2, 1))

        # Find the start and end time step of the first part
        start_vec[0] = input_vec[0]
        end_vec[0] = input_vec[break_mark - 1]

        # Find the start and end time step of the second part
        start_vec[1] = input_vec[break_mark]
        end_vec[1] = input_vec[-1]

        # Find the difference between the starts
        add_vec = np.array(start_vec - input_start).astype(int)

        # Make sure input_all_starts contains only integers
        input_all_starts = np.array(input_all_starts).astype(int)

        # Create start_mat with two parts
        start_mat = np.vstack(
            (input_all_starts + add_vec[0], input_all_starts + add_vec[1])
        )

    # Get the length of the new repeats
    length_vec = (end_vec - start_vec + 1).astype(int)

    return start_mat, length_vec


def __inds_to_rows(start_mat, row_length):
    """
    Expands a vector containing the starting indices of a piece or two of a
    repeat into a matrix representation recording when these pieces occur in
    the song with 1's. All remaining entries are marked with 0's.

    Args:
        start_mat (np.ndarray):
            Matrix of one or two rows, containing the starting indices.

        row_length (int):
            Length of the rows.

    Returns:
        new_mat (np.ndarray):
            Binary matrix of one or two rows, with 1's where the starting
            indices and 0's otherwise.
            
    """

    if start_mat.ndim == 1:
        # Convert a 1D array into 2D array
        start_mat = start_mat[None, :]

    # Initialize mat_rows and new_mat
    mat_rows = start_mat.shape[0]
    new_mat = np.zeros((mat_rows, row_length))

    for i in range(0, mat_rows):
        inds = start_mat[i, :]
        # Let the starting indices be 1
        new_mat[i, inds] = 1

    return new_mat.astype(int)


def __merge_based_on_length(full_mat, full_bw, target_bw):
    """
    Merges repeats that are the same length, as set by full_bw,
    and are repeats of the same piece of structure.

    Args:
        full_mat (np.ndarray):
            Binary matrix with ones where repeats start and zeroes otherwise.

        full_bw (np.ndarray):
            Length of repeats encoded in input_mat.

        target_bw (np.ndarray):
            Lengths of repeats that we seek to merge.

    Returns:
        A tuple (out_mat, one_length_vec) where all variables have data type
        np.ndarray.

        out_mat is a binary matrix with 1's where repeats start and 0's
        otherwise with rows of full_mat merged if appropriate.

        one_length_vec is a vector that contains the length of repeats encoded
        in out_mat.
    """

    # Sort full_bandwidth and full_mat
    bnds = np.argsort(full_bw, axis=None)
    temp_bandwidth = full_bw.flatten()[bnds]
    temp_mat = full_mat[bnds, :]

    # Find the unique elements of target_bandwidth
    target_bandwidth = np.unique(target_bw)

    # Number of columns
    target_size = target_bandwidth.shape[0]

    for i in range(0, target_size):
        test_bandwidth = target_bandwidth[i]

        # Check if temp_bandwidth is equal to test_bandwidth
        inds = (temp_bandwidth == test_bandwidth)

        # If the sum of all inds elements is greater than 1, then execute this
        # if statement
        if inds.sum() > 1:
            # Isolate rows that correspond to test_bandwidth and merge them
            merge_bw = temp_mat[inds, :]
            merged_mat = __merge_rows(merge_bw, np.array([test_bandwidth]))

            # Number of columns
            bandwidth_add_size = merged_mat.shape[0]
            bandwidth_add = test_bandwidth * np.ones((bandwidth_add_size, 
                                                      1)).astype(int)

            if np.any(inds):
                # Convert the boolean array inds into an array of integers
                inds = np.array(inds).astype(int)
                remove_inds = np.where(inds == 1)

                # Delete the rows that meet the condition set by remove_inds
                temp_mat = np.delete(temp_mat, remove_inds, axis=0)
                temp_bandwidth = np.delete(temp_bandwidth, remove_inds, axis=0)

            # Combine rows into a single matrix
            temp_mat = np.vstack((temp_mat, merged_mat))

            if temp_bandwidth.size == 0:
                temp_bandwidth = np.concatenate(bandwidth_add)
            elif temp_bandwidth.size > 0:
                temp_bandwidth = np.concatenate(
                    (temp_bandwidth, bandwidth_add.flatten())
                )

            # Return the indices that would sort temp_bandwidth
            bnds = np.argsort(temp_bandwidth)

            # Sort the elements of temp_bandwidth and temp_mat
            temp_bandwidth = temp_bandwidth[bnds]
            temp_mat = temp_mat[bnds, ]

    # Create output
    out_mat = temp_mat
    out_length_vec = temp_bandwidth

    if out_length_vec.size != 1:
        out_length_vec = out_length_vec.reshape(-1, 1)

    return out_mat, out_length_vec


def __merge_rows(input_mat, input_width):
    """
    Merges rows that have at least one common repeat; said common repeat(s)
    must occur at the same time step and be of common length.

    Args:
        input_mat (np.ndarray):
            Binary matrix with ones where repeats start and zeroes otherwise.

        input_width (int):
            Length of repeats encoded in input_mat.

    Returns:
        merge_mat (np.ndarray):
            Binary matrix with ones where repeats start and zeroes otherwise.
        
    """

    # Step 0: initialize temporary variables
    not_merge = input_mat
    merge_mat = np.empty((0, input_mat.shape[1]), int)
    merge_key = np.empty(1, int)
    rows = input_mat.shape[0]

    # Step 1: has every row been checked?
    while rows > 0:
        # Step 2: start merge process
        # Step 2a: choose first unmerged row
        row2check = not_merge[0, :]

        # Create a comparison matrix with copies of row2check stacked
        # so that r2c_mat is the same size as the set of rows waiting
        # to be merged
        r2c_mat = np.kron(np.ones((rows, 1)), row2check)

        # Step 2b: find indices of unmerged overlapping rows
        merge_inds = np.sum(((r2c_mat + not_merge) == 2), axis=1) > 0

        # Step 2c: union rows with starting indices in common with row2check
        # and remove those rows from input_mat
        union_merge = np.sum(not_merge[merge_inds, :], axis=0) > 0
        union_merge = union_merge.astype(int)
        not_merge = np.delete(not_merge, np.where(merge_inds == 1), 0)

        # Step 2d: check that newly merged rows do not cause overlaps within row
        # If there are conflicts, rerun compare_and_cut
        merge_block = reconstruct_full_block(union_merge, input_width)

        if np.max(merge_block) > 1:
            (union_merge, union_merge_key) = __compare_and_cut(
                union_merge, input_width, union_merge, input_width
            )
        else:
            union_merge_key = input_width

        # Step 2e: add unions to merge_mat and merge_key
        merge_mat = np.vstack((merge_mat, union_merge))
        merge_key = np.vstack((merge_key, union_merge_key))

        # Step 3: reinitialize rs for stopping condition
        rows = not_merge.shape[0]

    if np.ndim(merge_mat) == 1:
        # Make sure the output is a 2d array
        merge_mat = np.array([merge_mat])

    return merge_mat.astype(int)


def hierarchical_structure(matrix_no_overlaps, key_no_overlaps, sn, vis=False):
    """
    Distills the repeats encoded in matrix_no_overlaps (and key_no_overlaps) 
    to the essential structure components and then builds the hierarchical 
    representation. Optionally shows visualizations of the hierarchical
    structure via the vis argument.

    Args:
        matrix_no_overlaps (np.ndarray):
            Binary matrix with 1's where repeats begin and 0's otherwise.

        key_no_overlaps (np.ndarray):
            Vector containing the lengths of the repeats encoded
            in matrix_no_overlaps.

        sn (int):
        Song length, which is the number of audio shingles.

        vis (bool):
            Shows visualizations if True (default = False).

    Returns:
        A tuple (full_visualization, full_key, full_matrix_no_overlaps,
        full_anno_lst) where all variables have data type np.ndarray.

        full_visualization is a binary matrix representation for
        full_matrix_no_overlaps with blocks of 1's equal to the lengths
        prescribed in full_key.

        full_key is a vector containing the lengths of the hierarchical
        structure encoded in full_matrix_no_overlaps.

        full_matrix_no_overlaps is a binary matrix with 1's where hierarchical
        structure begins and 0's otherwise.

        full_anno_lst is a vector containing the annotation markers of the
        hierarchical structure encoded in each row of full_matrix_no_overlaps.

    """

    pno, pno_key = breakup_overlaps_by_intersect(matrix_no_overlaps,
                                                 key_no_overlaps, 0)

    # Using pno and pno_key, we build a vector that tells us the order of the
    # repeats of the essential structure components

    # Get the block representation for pno, called pno_block
    pno_block = reconstruct_full_block(pno, pno_key)

    if vis:
        # IMAGE 1 construction
        pno_anno = get_annotation_lst(pno_key)
        pno_y_labels = get_y_labels(pno_key, pno_anno)
        num_pno_rows = np.size(pno, axis=0)
        twos = np.full((num_pno_rows, sn), 2, dtype=int)
        # Visualization trick: 2s - white, 0s - black, 1s - gray
        vis_array = twos - (pno_block + pno)
        fig, ax = plt.subplots(1, 1, figsize=(10, 9))
        sdm = ax.imshow(vis_array, cmap="gray", aspect="auto")
        plt.title("Essential Structure Components")
        # Set the number of ticks and set tick intervals to be equal
        ax.set_yticks(np.arange(0, np.size(pno_y_labels) - 1))
        ax.set_yticklabels(pno_y_labels[1:])
        plt.show()

    # Assign a unique (nonzero) number for each row in PNO. We refer these
    # unique numbers COLORS.
    num_colors = pno.shape[0]
    num_timesteps = pno.shape[1]

    # Create unique color identifier for num_colors
    color_lst = np.arange(1, num_colors + 1)

    # Turn it into a column
    color_lst = color_lst.reshape(np.size(color_lst), 1)
    color_mat = np.tile(color_lst, (1, num_timesteps))

    # For each time step in row i that equals 1, change the value at that time
    # step to i
    pno_color = color_mat * pno
    pno_color_vec = pno_color.sum(axis=0)

    # Find where repeats exist in time, paying special attention to the starts
    # and ends of each repeat of an essential structure component
    # take sums down columns --- conv to logical
    pno_block_vec = (np.sum(pno_block, axis=0)) > 0
    pno_block_vec = pno_block_vec.astype(np.float32)
    one_vec = pno_block_vec[0: sn - 1] - pno_block_vec[1:sn]

    # Find all the blocks of consecutive time steps that are not contained in
    # any of the essential structure components
    # We call these blocks zero blocks
    # Shift pno_block_vec so that the zero blocks are marked at the correct
    # time steps with 1's
    if pno_block_vec[0] == 0:
        one_vec = np.insert(one_vec, 0, 1)
    elif pno_block_vec[0] == 1:
        one_vec = np.insert(one_vec, 0, 0)

    # Assign one new unique number to all the zero blocks
    pno_color_vec[one_vec == 1] = num_colors + 1

    # We are only concerned with the order that repeats of the essential
    # structure components occur in. So we create a vector that only contains
    # the starting indices for each repeat of the essential structure components

    # We isolate the starting index of each repeat of the essential structure
    # components and save a binary vector with 1 at a time step if a repeat of
    # any essential structure component occurs there
    non_zero_inds = (pno_color_vec > 0)
    num_nzi = non_zero_inds.sum(axis=0)
    pno_color_inds_only = pno_color_vec[non_zero_inds]

    # For indices that signals the start of a zero block, turn those indices
    # back to 0
    zero_inds_short = (pno_color_inds_only == (num_colors + 1))
    pno_color_inds_only[zero_inds_short] = 0

    # Create a binary matrix symm_pno_inds_only such that the (i,j) entry is 1
    # if the following three conditions are true:
    #     1) a repeat of an essential structure component is the i-th thing in
    #        the ordering
    #     2) a repeat of an essential structure component is the j-th thing in
    #        the ordering
    #     3) the repeat occurring in the i-th place of the ordering and the
    #        one occurring in the j-th place of the ordering are repeats of the
    #        same essential structure component.

    # If any of the above conditions are not true, then the (i,j) entry of
    # symm_pno_inds_only is 0.

    # Turn our pattern row into a square matrix by stacking that row the
    # number of times equal to the columns in that row
    pno_io_mat = np.tile(pno_color_inds_only, (num_nzi, 1))
    pno_io_mat = pno_io_mat.astype(np.float32)

    pno_io_mask = (
                    (pno_io_mat > 0).astype(np.float32)
                    + (pno_io_mat.transpose() > 0).astype(np.float32)
                  ) == 2
    symm_pno_inds_only = (
                          pno_io_mat.astype(np.float32) ==
                          pno_io_mat.transpose().astype(np.float32)
                         ) * pno_io_mask

    if vis:
        # IMAGE 2
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sdm = ax.imshow(symm_pno_inds_only, cmap="binary", aspect="auto")
        plt.title(
            "Threshold Self-dissimilarity Matrix of " +
            "the Ordering Essential Structure Components"
        )
        # this locator puts ticks at regular intervals
        loc = plticker.MultipleLocator(base=1.0)
        ax.yaxis.set_major_locator(loc)
        ax.xaxis.set_major_locator(loc)
        plt.show()

    # Extract all the diagonals in symm_pno_inds_only and get pairs of
    # repeated sublists in the order that repeats of essential structure
    # components.

    # These pairs of repeated sublists are the basis of our hierarchical
    # representation.

    nzi_lst = find_all_repeats(symm_pno_inds_only, np.arange(1, num_nzi + 1))
    remove_inds = (nzi_lst[:, 0] == nzi_lst[:, 2])

    # Remove any pairs of repeats that are two copies of the same repeat (i.e.
    # a pair (A,B) where A == B)
    if np.any(remove_inds):
        remove_inds = np.array(remove_inds).astype(int)
        remove = np.where(remove_inds == 1)
        nzi_lst = np.delete(nzi_lst, remove, axis=0)

    # Add the annotation markers to the pairs in nzi_lst
    nzi_lst_anno = find_complete_list_anno_only(nzi_lst, num_nzi)

    # Remove the overlaps
    nzi_matrix_no_overlaps, nzi_key_no_overlaps =\
        remove_overlaps(nzi_lst_anno, num_nzi)[1:3]

    # Reconstruct full block
    nzi_pattern_block = reconstruct_full_block(nzi_matrix_no_overlaps,
                                               nzi_key_no_overlaps)
    nzi_rows = nzi_pattern_block.shape[0]
    full_anno_lst = get_annotation_lst(nzi_key_no_overlaps)
    vis_y_labels = get_y_labels(nzi_key_no_overlaps[None, :].reshape(-1, 1),
                                full_anno_lst)
    if vis:
        # IMAGE 3
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sdm = ax.imshow(nzi_pattern_block, cmap="binary", aspect="auto")
        plt.title(
            "Repeated Ordered Sublists of the " +
            "Essential Structure Components"
        )
        # This locator puts ticks at regular intervals
        loc = plticker.MultipleLocator(1)
        # Set the number of ticks and set tick intervals to be equal
        ax.set_yticks(np.arange(0, np.size(vis_y_labels) - 1))
        # Set the ticklabels along the y axis and remove 0 in vis_y_labels
        ax.set_yticklabels(vis_y_labels[1:])
        ax.xaxis.set_major_locator(loc)
        plt.show()

        # IMAGE 4
        fig, ax = plt.subplots(1, 1, figsize=(10, 12))
        sdm = ax.imshow((nzi_pattern_block + nzi_matrix_no_overlaps),
                        cmap="binary", aspect="auto")
        plt.title(
            "Repeated Ordered Sublists of the " +
            "Essential Structure Components " +
            "with Leading Index Highlighted"
        )
        loc = plticker.MultipleLocator(
            base=1.0
        )  # This locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)
        # Set the number of ticks and set tick intervals to be equal
        ax.set_yticks(np.arange(0, np.size(vis_y_labels) - 1))
        # Set the ticklabels along the y axis and remove 0 in vis_y_labels
        ax.set_yticklabels(vis_y_labels[1:])
        plt.show()

    nzi_rows = nzi_pattern_block.shape[0]

    # Find where all blocks start and end
    pattern_starts = np.nonzero(non_zero_inds)[0]
    pattern_ends = np.array([pattern_starts[1:] - 1])
    pattern_ends = np.insert(pattern_ends, np.shape(pattern_ends)[1], sn - 1)
    pattern_lengths = np.array(pattern_ends - pattern_starts + 1)

    full_visualization = np.zeros((nzi_rows, sn), dtype=int)
    full_matrix_no_overlaps = np.zeros((nzi_rows, sn), dtype=int)

    for i in range(0, num_nzi):
        repeated_sect = nzi_pattern_block[:, i].reshape(
            np.shape(nzi_pattern_block)[0], 1
        )

        full_visualization[:,
        pattern_starts[i]: pattern_ends[i] + 1] = np.tile(
            repeated_sect, (1, pattern_lengths[i])
        )

        full_matrix_no_overlaps[:, pattern_starts[i]] =\
            nzi_matrix_no_overlaps[:, i]

    # Get full_key, the matching bandwidth key for full_matrix_no_overlaps
    full_key = np.zeros((nzi_rows, 1), dtype=int)
    find_key_mat = full_visualization + full_matrix_no_overlaps

    for i in range(0, nzi_rows):
        one_start = np.where(find_key_mat[i, :] == 2)[0][0]
        temp_row = find_key_mat[i, :]
        temp_row[0: one_start + 1] = 1

        if np.size(np.where(temp_row == 0)[0]) == 0:
            find_zero = sn
        else:
            find_zero = np.where(temp_row == 0)[0][0]

        if np.size(np.where(temp_row == 2)[0]) == 0:
            find_two = sn
        else:
            find_two = np.where(temp_row == 2)[0][0]

        one_end = np.minimum(find_zero, find_two)
        full_key[i] = one_end - one_start

    full_key_inds = np.argsort(full_key, axis=0)

    # Switch to row
    full_key_inds = full_key_inds[:, 0]
    full_key = np.sort(full_key, axis=0)
    full_visualization = full_visualization[full_key_inds, :]
    full_matrix_no_overlaps = full_matrix_no_overlaps[full_key_inds, :]

    # Remove rows of our hierarchical representation that contain only
    # one repeat
    inds_remove = np.where(np.sum(full_matrix_no_overlaps, 1) <= 1)
    full_key = np.delete(full_key, inds_remove, axis=0)

    full_matrix_no_overlaps = np.delete(full_matrix_no_overlaps, inds_remove,
                                        axis=0)
    full_visualization = np.delete(full_visualization, inds_remove, axis=0)

    full_anno_lst = get_annotation_lst(full_key)

    if vis:
        # IMAGE 5
        full_anno_lst = get_annotation_lst(full_key)
        vis_y_labels = get_y_labels(full_key, full_anno_lst)
        num_vis_rows = np.size(full_visualization, axis=0)
        twos = np.full((num_vis_rows, sn), 2, dtype=int)
        vis_array = twos - (full_visualization + full_matrix_no_overlaps)
        fig, ax = plt.subplots(1, 1, figsize=(10, 13))
        sdm = ax.imshow(vis_array, cmap="gray", aspect="auto")
        plt.title("Complete Aligned Hierarchies")
        # Set the number of ticks and set tick intervals to be equal
        ax.set_yticks(np.arange(0, np.size(vis_y_labels) - 1))
        # Set the ticklabels along the y axis and remove 0 in vis_y_labels
        ax.set_yticklabels(vis_y_labels[1:])
        plt.show()

    return full_visualization, full_key, full_matrix_no_overlaps, full_anno_lst