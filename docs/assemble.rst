Assemble
========

The module ``assemble.py`` finds and forms essential structure components, which are 
the smallest building blocks that form every repeat in the song. These functions ensure 
that each time step of a song is contained in at most one of the song's essential 
structure components by checking that there are no overlapping repeats in time. 
When repeats overlap, they undergo a process where they are divided until there are 
only non-overlapping pieces left. 

This module contains the following functions:

.. function:: breakup_overlaps_by_intersect(input_pattern_obj, bw_vec, thresh_bw)

    Extracts repeats in input_pattern_obj that has the starting indices of the
    repeats, into the essential structure components using bw_vec, that has the
    lengths of each repeat. The essential structure components are the
    smallest building blocks that form every repeat in the song.

    :parameters:

        input_pattern_obj : np.ndarray
            Binary matrix with 1's where repeats begin
            and 0's otherwise.

        bw_vec : np.ndarray
            Vector containing the lengths of the repeats
            encoded in input_pattern_obj.

        thresh_bw : int
            Smallest allowable repeat length.

    :returns:

        pattern_no_overlaps : np.ndrray
            Binary matrix with 1's where repeats of
            essential structure components begin.

        pattern_no_overlaps_key : np.ndarray
            Vector containing the lengths of the repeats
            of essential structure components in
            pattern_no_overlaps.

.. function:: check_overlaps(input_mat)

    Compares every pair of groups and determines if there are any repeats in
    any pairs of the groups that overlap.

    :parameters:

        input_mat : np.array[int]
            Matrix to be checked for overlaps.

    :returns:

        overlaps_yn : np.array[bool]
            Logical array where (i,j) = 1 if row i of input matrix and row j
            of input matrix overlap and (i,j) = 0 elsewhere.

.. function:: hierarchical_structure(matrix_no_overlaps, key_no_overlaps, sn, vis=False)

    Distills the repeats encoded in matrix_no_overlaps (and key_no_overlaps)
    to the essential structure components and then builds the hierarchical
    representation. Optionally shows visualizations of the hierarchical structure
    via the vis argument.

    :parameters:

        matrix_no_overlaps : np.ndarray[int]
            Binary matrix with 1's where repeats begin and 0's otherwise.

        key_no_overlaps : np.ndarray[int]
            Vector containing the lengths of the repeats encoded in matrix_no_overlaps.

        sn : int
            Song length, which is the number of audio shingles.

        vis : bool
            Shows visualizations if True (default = False).

    :returns:

        full_visualization : np.ndarray[int]
            Binary matrix representation for full_matrix_no_overlaps
            with blocks of 1's equal to the length's prescribed
            in full_key.

        full_key : np.ndarray[int]
            Vector containing the lengths of the hierarchical
            structure encoded in full_matrix_no_overlaps.

        full_matrix_no_overlaps : np.ndarray[int]
            Binary matrix with 1's where hierarchical
            structure begins and 0's otherwise.

        full_anno_lst : np.ndarray[int]
            Vector containing the annotation markers of the
            hierarchical structure encoded in each row of
            full_matrix_no_overlaps.

