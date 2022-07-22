#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities.py

This module, when imported, allows search.py, transform.py and assemble.py
in the repytah package to run smoothly.

The module contains the following functions:

    * create_sdm
        Creates a self-dissimilarity matrix; this matrix is found
        by creating audio shingles from feature vectors, and finding cosine
        distance between shingles.

    * find_initial_repeats
        Finds all diagonals present in thresh_mat, removing each diagonal
        as it is found.

    * stretch_diags
        Fills out diagonals in binary self-dissimilarity matrix from diagonal
        starts and lengths.

    * add_annotations
        Adds annotations to each pair of repeated structures according to
        their length and order of occurrence.

    * __find_song_pattern
        Stitches information about repeat locations from thresh_diags matrix
        into a single row.

    * reconstruct_full_block
        Creates a record of when pairs of repeated structures occur, from
        the first beat in the song to the last beat of the song. Pairs of
        repeated structures are marked with 1's.

    * get_annotation_lst
        Gets one annotation marker vector, given vector of lengths key_lst.

    * get_y_labels
        Generates the labels for visualization.

    * reformat [Only used for creating test examples]
        Transforms a binary matrix representation of when repeats occur in
        a song into a list of repeated structures detailing the length and
        occurrence of each repeat.

"""

import numpy as np
import scipy.sparse as sps
import scipy.spatial.distance as spd
import cv2

def create_sdm(fv_mat, num_fv_per_shingle):
    """
    Creates self-dissimilarity matrix; this matrix is found by creating audio
    shingles from feature vectors, and finding the cosine distance between
    shingles.

    Args:
        fv_mat (np.ndarray):
            Matrix of feature vectors where each column is a time step and each
            row includes feature information i.e. an array of 144 columns/beats
            and 12 rows corresponding to chroma values.

        num_fv_per_shingle (int):
            Number of feature vectors per audio shingle.

    Returns:
        self_dissim_mat (np.ndarray):
            Self-dissimilarity matrix with paired cosine distances between
            shingles.
    """

    [num_rows, num_columns] = fv_mat.shape

    if num_fv_per_shingle == 1:
        mat_as = fv_mat
    else:
        mat_as = np.zeros(((num_rows * num_fv_per_shingle),
                           (num_columns - num_fv_per_shingle + 1)))

        for i in range(num_fv_per_shingle):
            # Use feature vectors to create an audio shingle
            # for each time step and represent these shingles
            # as vectors by stacking the relevant feature
            # vectors on top of each other
            mat_as[i * num_rows:(i + 1) * num_rows, :] = \
                fv_mat[:, i:(num_columns - num_fv_per_shingle + i + 1)]

    # Build the pairwise-cosine distance matrix between audio shingles
    sdm_row = spd.pdist(mat_as.T, 'cosine')

    # Build self-dissimilarity matrix by changing the condensed
    # pairwise-cosine distance matrix to a redundant matrix
    self_dissim_mat = spd.squareform(sdm_row)

    return self_dissim_mat


def find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw):
    """
    Looks for the largest repeated structures in thresh_mat. Finds all
    repeated structures, represented as diagonals present in thresh_mat,
    and then stores them with their start/end indices and lengths in a
    list. As each diagonal is found, they are removed to avoid identifying
    repeated sub-structures.


    Args:
        thresh_mat (np.ndarray[int]):
            Thresholded matrix that we extract diagonals from.

        bandwidth_vec (np.ndarray[1D,int]):
            Array of lengths of diagonals to be found. Should be
            1, 2, 3,..., n where n is the number of timesteps.

        thresh_bw (int):
            One less than smallest allowed repeat length.

    Returns:
        all_lst (np.ndarray[int]):
            List of pairs of repeats that correspond to diagonals in
            thresh_mat.
    """

    # Initialize the input and temporary variables
    thresh_temp = thresh_mat

    # Interval list for non-overlapping pairs
    int_all = np.empty((0, 5), int)

    # Interval list for the left side of the overlapping pairs
    sint_all = np.empty((0, 5), int)

    # Interval list for the right side of the overlapping pairs
    eint_all = np.empty((0, 5), int)

    # Interval list for the middle of the overlapping pairs if they exist
    mint_all = np.empty((0, 5), int)

    # Loop over all bandwidths from n to 1
    for bw in bandwidth_vec[::-1]:
        if bw > thresh_bw:

            # Use matrix correlation to find diagonals of length bw
            id_mat = np.identity(bw)

            # Search for diagonals of length bw
            
            # Use smallest datatype that can contain bw 
            if bw < 255:
                diagonal_mat = cv2.filter2D(thresh_temp.astype(np.uint8), -1,
                                            id_mat, anchor=(0, 0),
                                            borderType=cv2.BORDER_CONSTANT)
            elif bw <= 65535:
                diagonal_mat = cv2.filter2D(thresh_temp.astype(np.uint16), -1,
                                            id_mat, anchor=(0, 0),
                                            borderType=cv2.BORDER_CONSTANT)
            else:
                raise RuntimeError("Bandwidth value too large")

            # Splice away results from zero-padding
            outdims = (np.array(thresh_temp.shape[0])
                       - np.array(id_mat.shape[0])) + 1
            diagonal_mat = diagonal_mat[0:outdims, 0:outdims]

            # Mark where diagonals of length bw start
            diag_markers = (diagonal_mat == bw).astype(int)

            if diag_markers.any():

                # 1) Non-Overlaps: Search outside the overlapping shingles

                # Search for paired starts
                start_i, start_j = np.nonzero(np.triu(diag_markers, bw))
                start_i = start_i + 1
                start_j = start_j + 1

                num_ints = start_i.shape[0]


                # Find the matching ends for the previously found starts
                match_i = start_i + (bw - 1)
                match_j = start_j + (bw - 1)

                # List pairs of starts with their ends and the widths of the
                # non-overlapping interval

                int_lst = np.vstack((start_i, match_i, start_j, match_j,
                                     bw * np.ones((1, num_ints), int))).T

                # Add the new non-overlapping intervals to the full list of
                # non-overlapping intervals
                int_all = np.vstack((int_lst, int_all))

                # 2) Overlaps: Search only the overlaps in shingles

                # Search for paired starts

                start_i_over, start_j_over = np.nonzero((np.tril(
                    np.triu(diag_markers), (bw - 1))))
                start_i_over = start_i_over + 1  # row
                start_j_over = start_j_over + 1  # column
                num_overlaps = start_i_over.shape[0]

                # Check if overlap found is the whole data stream
                if num_overlaps == 1 and start_i_over == start_j_over:
                    sint_lst = np.vstack((start_i_over, start_i_over + (bw - 1),
                                          start_j_over, start_j_over + (bw - 1),
                                          bw)).T
                    sint_all = np.vstack((sint_all, sint_lst))

                elif num_overlaps > 0:
                    # Since you are checking the overlaps you need to cut these
                    # intervals into pieces: left, right, and middle.
                    # NOTE: the middle interval may NOT exist

                    # 2a) Left Overlap
                    # Width is the same for  left and right overlaps
                    K = start_j_over - start_i_over

                    smatch_i_over = start_j_over - 1
                    smatch_j_over = start_j_over + K - 1

                    # List pairs of starts with their ends and widths
                    sint_lst = np.vstack((start_i_over, smatch_i_over,
                                          start_j_over, smatch_j_over, K)).T

                    # Remove the pairs that fall below the bandwidth threshold
                    keep_s = sint_lst[:, 4] > thresh_bw
                    sint_lst = sint_lst[keep_s]

                    # Add the new left overlapping intervals to the full list
                    # of left overlapping intervals
                    sint_all = np.vstack((sint_all, sint_lst))

                    # 2b) Right Overlap

                    end_i_over = start_i_over + (bw - 1)
                    end_j_over = start_j_over + (bw - 1)

                    ematch_i_over = end_i_over - K + 1
                    ematch_j_over = end_i_over + 1

                    eint_lst = np.vstack((ematch_i_over, end_i_over,
                                          ematch_j_over, end_j_over, K)).T

                    # Remove the pairs that fall below the bandwidth threshold
                    keep_e = eint_lst[:, 4] > thresh_bw
                    eint_lst = eint_lst[keep_e]

                    # Add the new right overlapping intervals to the full list
                    # of right overlapping intervals
                    eint_all = np.vstack((eint_all, eint_lst))

                    # 2) Middle Overlap

                    mnds = (end_i_over - start_j_over - K + 1) > 0

                    if sum(mnds) > 0:
                        midd_int = np.vstack((
                            start_j_over, end_i_over - K, start_j_over + K,
                            end_i_over, end_i_over - start_j_over - K + 1)).T

                        # Remove the pairs that fall below the bandwidth
                        # threshold
                        mint_lst = midd_int[mnds]

                        # Add the new middle overlapping intervals to the
                        # full list of middle overlapping intervals
                        mint_all = np.vstack((mint_all, mint_lst))

            # Remove found diagonals of length BW from consideration
            SDM = stretch_diags(diag_markers, bw)
            thresh_temp = np.logical_xor(thresh_temp, SDM)

            if thresh_temp.sum() == 0:
                break

    # Combine all found pairs of repeats
    out_lst = np.vstack((sint_all, eint_all, mint_all))
    all_lst = np.vstack((int_all, out_lst))

    # Sort the output array first by repeat length, then by starts of i and
    # finally by j
    inds = np.lexsort((all_lst[:, 2], all_lst[:, 0], all_lst[:, 4]))
    all_lst = all_lst[inds]

    return all_lst.astype(int)


def stretch_diags(thresh_diags, band_width):
    """
    Creates a binary matrix with full length diagonals from a binary matrix of
    diagonal starts and length of diagonals.

    Args:
        thresh_diags (np.ndarray):
            Binary matrix where entries equal to 1 signals the existence
            of a diagonal.

        band_width (int):
            Length of encoded diagonals.

    Returns:
        stretch_diag_mat (np.ndarray[bool]):
            Logical matrix with diagonals of length band_width starting
            at each entry prescribed in thresh_diag.
    """

    # Create size of returned matrix
    n = thresh_diags.shape[0] + band_width - 1
    temp_song_marks_out = np.zeros(n)

    # find starting row, column indices of diagonals
    inds, jnds = thresh_diags.nonzero()

    subtemp = np.identity(band_width)

    # Expand each entry in thresh_diags into diagonal of
    # length band width
    for i in range(inds.shape[0]):
        tempmat = np.zeros((n, n))
        tempmat[inds[i]:(inds[i] + band_width),
        jnds[i]:(jnds[i] + band_width)] = subtemp
        temp_song_marks_out = temp_song_marks_out + tempmat

    # Ensure that stretch_diag_mat is a binary matrix
    stretch_diag_mat = (temp_song_marks_out > 0)

    return stretch_diag_mat


def add_annotations(input_mat, song_length):
    """
    Adds annotations to the pairs of repeats in input_mat.

    Args:
        input_mat (np.ndarray):
            List of pairs of repeats. The first two columns refer to the first
            repeat of the pair. The third and fourth columns refer to the
            second repeat of the pair. The fifth column refers to the repeat
            lengths. The sixth column contains any previous annotations, which
            will be removed.

        song_length (int):
            Number of audio shingles in the song.

    Returns:
        anno_list (np.ndarray):
            List of pairs of repeats with annotations marked.
    """

    num_rows = input_mat.shape[0]

    # Remove any already present annotation markers
    input_mat[:, 5] = 0

    # Find where repeats start
    s_one = input_mat[:, 0]
    s_two = input_mat[:, 2]

    # Create matrix of all repeats
    s_three = np.ones(num_rows, dtype=int)

    up_tri_mat = sps.coo_matrix((s_three, (s_one - 1, s_two - 1)),
                                shape=(song_length, song_length)).toarray()

    low_tri_mat = up_tri_mat.conj().transpose()

    full_mat = up_tri_mat + low_tri_mat

    # Stitch info from input_mat into a single row
    song_pattern = __find_song_pattern(full_mat)
    SPmax = max(song_pattern)

    # Add annotation markers to pairs of repeats
    for i in range(1, SPmax + 1):
        pinds = np.nonzero(song_pattern == i)

        # One if annotation not already marked, zero if it is
        check_inds = (input_mat[:, 5] == 0)

        for j in pinds[0]:
            # Find all starting pairs that contain time step j
            # and DO NOT have an annotation
            mark_inds = (s_one == j + 1) + (s_two == j + 1)
            mark_inds = (mark_inds > 0)
            mark_inds = check_inds * mark_inds

            # Add found annotations to the relevant time steps
            input_mat[:, 5] = (input_mat[:, 5] + i * mark_inds)

            # Remove pairs of repeats with annotations from consideration
            check_inds = check_inds ^ mark_inds

    temp_inds = np.argsort(input_mat[:, 5])

    # Create list of annotations
    anno_list = input_mat[temp_inds]

    return anno_list


def __find_song_pattern(thresh_diags):
    """
    Stitches information from thresh_diags matrix into a single
    row, song_pattern, that shows the time steps containing repeats;
    From the full matrix that decodes repeat beginnings (thresh_diags),
    the locations, or beats, where these repeats start are found and
    encoded into the song_pattern array.

    Args:
        thresh_diags (np.ndarray):
            Binary matrix with 1 at the start of each repeat pair (SI,SJ)
            and 0 elsewhere.
            WARNING: Must be symmetric.

    Returns:
        song_pattern (np.ndarray):
            Row where each entry represents a time step and the group that
            time step is a member of.
    """

    song_length = thresh_diags.shape[0]

    # Initialize song pattern base
    pattern_base = np.zeros(song_length, dtype=int)

    # Initialize group number
    pattern_num = 1

    col_sum = thresh_diags.sum(axis=0)
    check_inds = col_sum.nonzero()
    check_inds = check_inds[0]

    # Create vector of song length
    pattern_mask = np.ones(song_length)
    pattern_out = (col_sum == 0)
    pattern_mask = (pattern_mask - pattern_out).astype(int)

    while np.size(check_inds) != 0:
        # Take first entry in check_inds
        i = check_inds[0]

        # Take the corresponding row from thresh_diags
        temp_row = thresh_diags[i, :]

        # Find all time steps that i is close to
        inds = temp_row.nonzero()

        if np.size(inds) != 0:
            while np.size(inds) != 0:
                # Take sum of rows corresponding to inds and
                # multiplies the sums against p_mask
                c_mat = np.sum(thresh_diags[inds, :], axis=1).flatten()
                c_mat = c_mat * pattern_mask

                # Find nonzero entries of c_mat
                c_inds = c_mat.nonzero()

                # Give all elements of c_inds the same grouping
                # number as i
                pattern_base[c_inds] = pattern_num

                # Remove all used elements of c_inds from
                # check_inds and p_mask
                check_inds = np.setdiff1d(check_inds, c_inds)
                pattern_mask[c_inds] = 0

                # Reset inds to c_inds with inds removed
                inds = np.setdiff1d(c_inds, inds)
                inds = np.array([inds])

            # Update grouping number to prepare for next group
            pattern_num = pattern_num + 1

        # Remove i from check_inds
        check_inds = np.setdiff1d(check_inds, i)

    song_pattern = pattern_base

    return song_pattern


def reconstruct_full_block(pattern_mat, pattern_key):
    """
    Creates a record of when pairs of repeated structures occur, from the
    first beat in the song to the end. This record is a binary matrix with a
    block of 1's for each repeat encoded in pattern_mat whose length is
    encoded in pattern_key.
    
    Args:
        pattern_mat (np.ndarray):
            Binary matrix with 1's where repeats begin and 0's otherwise.

        pattern_key (np.ndarray):
            Vector containing the lengths of the repeats encoded in
            each row of pattern_mat.

    Returns:
        pattern_block (np.ndarray):
            Binary matrix representation for pattern_mat with blocks
            of 1's equal to the length's prescribed in pattern_key.
    """

    # First, find number of beats (columns) in pattern_mat:
    # Check size of pattern_mat (in cases where there is only 1 pair of
    # repeated structures)
    if pattern_mat.ndim == 1:
        # Convert a 1D array into 2D array
        pattern_mat = pattern_mat[None, :]
        # Assign number of beats to sn
        sn = pattern_mat.shape[1]
    else:
        # Assign number of beats to sn
        sn = pattern_mat.shape[1]

    # Assign number of repeated structures (rows) in pattern_mat to sb
    sb = pattern_mat.shape[0]

    # Pre-allocating a sn by sb array of zeros
    pattern_block = np.zeros((sb, sn)).astype(int)

    # Check if pattern_key is in vector row
    if pattern_key.ndim != 1:

        # Convert pattern_key into a vector row
        length_vec = pattern_key.flatten()

    else:
        length_vec = pattern_key

    for i in range(sb):
        # Retrieve all of row i of pattern_mat
        repeated_struct = pattern_mat[i, :]

        # Retrieve the length of the repeats encoded in row i of pattern_mat
        length = length_vec[i]

        # Pre-allocate a section of size length x sn for pattern_block
        sub_section = np.zeros((length, sn))

        # Replace first row in sub_section with repeated_struct
        sub_section[0, :] = repeated_struct

        # Create pattern_block: Sums up each column after sliding repeated
        # structure i to the right bw - 1 times
        for b in range(1, length):
            # Retrieve repeated structure i up to its (1 - b) position
            sub_struct_a = repeated_struct[:-b]

            # Row vector with number of entries not included in sub_struct_a
            sub_struct_b = np.zeros(b)

            # Append sub_struct_b in front of sub_struct_a
            new_struct = np.append(sub_struct_b, sub_struct_a)

            # Replace part of sub_section with new_struct
            sub_section[b, :] = new_struct

        # Replace part of pattern_block with the sums of each column in
        # sub_section
        pattern_block[i, :] = np.sum(sub_section, axis=0)

    return pattern_block


def get_annotation_lst(key_lst):
    """
    Creates one annotation marker vector, given vector of lengths key_lst.

    Args:
        key_lst (np.ndarray[int]):
            Array of lengths in ascending order.

    Returns:
        anno_lst_out (np.ndarray[int]):
            Array of one possible set of annotation markers for key_lst.
    """

    # Initialize the temporary variable
    num_rows = np.size(key_lst)
    full_anno_lst = np.zeros(num_rows)

    # Find the first instance of each length and give it 1 as an annotation
    # marker
    unique_keys = np.unique(key_lst, return_index=True)
    full_anno_lst[unique_keys[1]] = 1

    # Add remaining annotations to annotation list
    for i in range(0, np.size(full_anno_lst)):
        if full_anno_lst[i] == 0:
            full_anno_lst[i] = full_anno_lst[i - 1] + 1

    return full_anno_lst.astype(int)


def get_y_labels(width_vec, anno_vec):
    """
    Generates the labels for visualization with width_vec and anno_vec.

    Args:
        width_vec (np.ndarray[int]):
            Vector of widths for a visualization.

        anno_vec (np.ndarray[int]):
            Array of annotations for a visualization.

    Returns:
        y_labels (np.ndarray[str]):
            Labels for the y-axis of a visualization. Each label contains the
            width and annotation number of an essential structure component.
    """

    # Determine number of rows to label
    num_rows = np.size(width_vec)

    # Make sure the sizes of width_vec and anno_vec are the same
    assert (num_rows == np.size(anno_vec))

    # Initialize the array with 0 as the origin
    y_labels = np.array([0])

    # Loop over the array adding labels
    for i in range(0, num_rows):
        label = ('w = ' + str(width_vec[i][0].astype(int)) +
                 ', a = ' + str(anno_vec[i]))
        y_labels = np.append(y_labels, label)

    return y_labels


def reformat(pattern_mat, pattern_key):
    """
    Transforms a binary array with 1's where repeats start and 0's
    otherwise into a list of repeated structures. This list consists of
    information about the repeats including length, when they occur and when
    they end.

    Every row has a pair of repeated structure. The first two columns are
    the time steps of when the first repeat of a repeated structure start and
    end. Similarly, the second two columns are the time steps of when the
    second repeat of a repeated structure start and end. The fifth column is
    the length of the repeated structure.

    Reformat is not used in the main process for creating the
    aligned hierarchies. It is helpful when writing example inputs for
    the tests.

    Args:
        pattern_mat (np.ndarray):
            Binary array with 1's where repeats start and 0's otherwise.

        pattern_key (np.ndarray):
            Array with the lengths of each repeated structure in pattern_mat.

    Returns:
        info_mat (np.ndarray):
            Array with the time steps of when the pairs of repeated structures
            start and end organized.
    """

    # Pre-allocate output array with zeros
    info_mat = np.zeros((pattern_mat.shape[0], 5))

    # Retrieve the index values of the repeats in pattern_mat
    results = np.where(pattern_mat == 1)

    for x, j in zip(range(pattern_mat.shape[0]), (range(0,
                                                        results[0].size - 1,
                                                        2))):
        # Assign the time steps of the repeated structures into info_mat
        info_mat[x, 0] = results[1][j] + 1
        info_mat[x, 1] = info_mat[x, 0] + pattern_key[x] - 1
        info_mat[x, 2] = results[1][j + 1] + 1
        info_mat[x, 3] = info_mat[x, 2] + pattern_key[x] - 1
        info_mat[x, 4] = pattern_key[x]

    return info_mat.astype(int)
