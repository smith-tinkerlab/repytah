Search
======

The module ``search.py`` holds functions used to find and record the diagonals 
in the thresholded matrix, T. These functions prepare the diagonals found for
transformation and assembling later. 

This module contains the following functions:

.. function:: find_complete_list(pair_list, song_length)

    Finds all smaller diagonals (and the associated pairs of repeats) that are
    contained in pair_list, which is composed of larger diagonals found in 
    find_initial_repeats.
        
    :parameters:

        pair_list : np.ndarray
            List of pairs of repeats found in earlier steps
            (bandwidths MUST be in ascending order). If you have
            run find_initial_repeats before this script,
            then pair_list will be ordered correctly. 
            
        song_length : int
            Song length, which is the number of audio shingles.
   
    :returns:  

        lst_out : np.ndarray 
            List of pairs of repeats with smaller repeats added.

.. function:: find_all_repeats(thresh_mat, bw_vec)

    Finds all the diagonals present in thresh_mat. This function is nearly 
    identical to find_initial_repeats, with two crucial differences. 
    First, we do not remove diagonals after we find them. Second, 
    there is no smallest bandwidth size as we are looking for all diagonals.
        
    :parameters:

        thresh_mat : np.ndarray
            Thresholded matrix that we extract diagonals from.
        
        bw_vec : np.ndarray
            Vector of lengths of diagonals to be found.
            Should be 1, 2, 3, ..., n where n is the number of timesteps. 
        
    :returns:

        all_lst : np.array
            Pairs of repeats that correspond to diagonals in thresh_mat.

.. function:: find_complete_list_anno_only(pair_list, song_length)

    Finds annotations for all pairs of repeats found in find_all_repeats. 
    This list contains all the pairs of repeated structures with their 
    starting/ending indices and lengths.
    
    :parameters:

        pair_list : np.ndarray
            List of pairs of repeats.
            WARNING: Bandwidths must be in ascending order.
            
        song_length : int
            Number of audio shingles in song.
        
    :returns:

        out_lst : np.ndarray
            List of pairs of repeats with smaller repeats added and with
            annotation markers.