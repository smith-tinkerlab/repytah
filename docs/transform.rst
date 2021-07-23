Transform
=========

The module ``transform.py`` contains functions that transform matrix inputs 
into different forms that are of use in bigger functions where they are called. 
These functions focus mainly on overlapping repeated structures and annotation 
markers.

This module contains the following functions:

.. function:: remove_overlaps(input_mat, song_length)

    Removes any pairs of repeat length and specific annotation marker 
    where there exists at least one pair of repeats that overlap in time.

    :parameters:

        input_mat : np.ndarray[int]
            List of pairs of repeats with annotations marked. The first 
            two columns refer to the first repeat or the pair, the second 
            two refer to the second repeat of the pair, the fifth column 
            refers to the length of the repeats, and the sixth column 
            contains the annotation markers.
            
        song_length : int
            Number of audio shingles.
 
    :returns:

        lst_no_overlaps : np.ndarray[int]
            List of pairs of repeats with annotations marked. All the 
            repeats of a given length and with a specific annotation 
            marker do not overlap in time.
            
        matrix_no_overlaps : np.ndarray[int]
            Matrix representation of lst_no_overlaps with one row for 
            each group of repeats.
            
        key_no_overlaps : np.ndarray[int]
            Vector containing the lengths of the repeats encoded in 
            each row of matrix_no_overlaps.
            
        annotations_no_overlaps : np.ndarray[int]
            Vector containing the annotation markers of the repeats 
            encoded in each row of matrix_no_overlaps.
            
        all_overlap_lst : np.ndarray[int]
            List of pairs of repeats with annotations marked removed 
            from input_mat. For each pair of repeat length and specific 
            annotation marker, there exist at least one pair of repeats 
            that do overlap in time.