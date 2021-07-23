Utilities
=========

The module ``utilities.py``, when imported, allows ``search.py``, ``transform.py`` and 
``assemble.py`` in the ``repytah`` package to run smoothly. 

This module contains the following functions:

.. function:: create_sdm(fv_mat, num_fv_per_shingle)

    Creates self-dissimilarity matrix; this matrix is found by creating audio
    shingles from feature vectors, and finding the cosine distance between
    shingles.

    :parameters:

        fv_mat : np.ndarray
            Matrix of feature vectors where each column is a time step and each
            row includes feature information i.e. an array of 144 columns/beats
            and 12 rows corresponding to chroma values.

        num_fv_per_shingle : int
            Number of feature vectors per audio shingle.

    :returns:

        self_dissim_mat : np.ndarray
            Self-dissimilarity matrix with paired cosine distances between shingles.

.. function:: find_initial_repeats(thresh_mat, bandwidth_vec, thresh_bw)

    Looks for the largest repeated structures in thresh_mat. Finds all
    repeated structures, represented as diagonals present in thresh_mat,
    and then stores them with their start/end indices and lengths in a
    list. As each diagonal is found, they are removed to avoid identifying
    repeated sub-structures.

    :parameters:

        thresh_mat : np.ndarray[int]
            Thresholded matrix that we extract diagonals from.

        bandwidth_vec : np.ndarray[1D,int]
            Array of lengths of diagonals to be found. Should be
            1, 2, 3, ..., n where n is the number of timesteps.

        thresh_bw : int
            Smallest allowed diagonal length.

    :returns:

        all_lst : np.ndarray[int]
            List of pairs of repeats that correspond to diagonals in
            thresh_mat.

.. function:: stretch_diags(thresh_diags, band_width)

    Creates a binary matrix with full length diagonals from a binary matrix of
    diagonal starts and length of diagonals.

    :parameters:

        thresh_diags : np.ndarray
            Binary matrix where entries equal to 1 signals the existence
            of a diagonal.

        band_width : int
            Length of encoded diagonals.

    :returns:

        stretch_diag_mat : np.ndarray[bool]
            Logical matrix with diagonals of length band_width starting
            at each entry prescribed in thresh_diag.

.. function:: add_annotations(input_mat, song_length)

    Adds annotations to the pairs of repeats in input_mat.

    :parameters:

        input_mat : np.ndarray
            List of pairs of repeats. The first two columns refer to
            the first repeat of the pair. The third and fourth columns
            refer to the second repeat of the pair. The fifth column
            refers to the repeat lengths. The sixth column contains any
            previous annotations, which will be removed.

        song_length : int
            Number of audio shingles in the song.

    :returns:

        anno_list : np.ndarray
            List of pairs of repeats with annotations marked.

.. function:: reconstruct_full_block(pattern_mat, pattern_key)

    Creates a record of when pairs of repeated structures occur, from the
    first beat in the song to the end. This record is a binary matrix with a
    block of 1's for each repeat encoded in pattern_mat whose length is
    encoded in pattern_key.

    :parameters:

        pattern_mat : np.ndarray
            Binary matrix with 1's where repeats begin and 0's otherwise.

        pattern_key : np.ndarray
            Vector containing the lengths of the repeats encoded in
            each row of pattern_mat.

    :returns:

        pattern_block : np.ndarray
            Binary matrix representation for pattern_mat with blocks
            of 1's equal to the length's prescribed in pattern_key.

.. function:: get_annotation_lst(key_lst)

    Creates one annotation marker vector, given vector of lengths key_lst.

    :parameters:

        key_lst : np.ndarray[int]
            Array of lengths in ascending order.

    :returns:

        anno_lst_out : np.ndarray[int]
            Array of one possible set of annotation markers for key_lst.

.. function:: get_y_labels(width_vec, anno_vec)

    Generates the labels for visualization with width_vec and anno_vec.

    :parameters:

        width_vec : np.ndarray[int]
            Vector of widths for a visualization.

        anno_vec : np.ndarray[int]
            Array of annotations for a visualization.

    :returns:

        y_labels : np.ndarray[str]
            Labels for the y-axis of a visualization.

.. function:: reformat(pattern_mat, pattern_key)

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
    aligned-hierarchies. It is helpful when writing example inputs for
    the tests.

    :parameters:

        pattern_mat : np.ndarray
            Binary array with 1's where repeats start and 0's otherwise.

        pattern_key : np.ndarray
            Array with the lengths of each repeated structure in pattern_mat.

    :returns:

        info_mat : np.ndarray
            Array with the time steps of when the pairs of repeated structures
            start and end organized.

