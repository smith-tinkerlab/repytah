#line 11 https://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
import numpy as np

def check_overlaps(input_mat):
    """
    Compares every pair of rows from input matrix and checks for overlaps
        between those pairs
    
    Args
    ----
    input_mat: np.array(int)
        matrix to be checked for overlaps
    
    Returns
    -------
    overlaps_yn: np.array(bool)
        logical array where (i,j) = 1 if row i of input matrix and row j
        of input matrix overlap and (i,j) = 0 elsewhere
    """
    # Get number of rows and columns
    rs = input_mat.shape[0]
    ws = input_mat.shape[1]

    # R_LEFT -- Every row of INPUT_MAT is repeated RS times to create a 
    # submatrix. We stack these submatrices on top of each other.
    compare_left = np.zeros(((rs*rs), ws))
    for i in range(rs):
        compare_add = input_mat[i,:]
        compare_add_mat = np.tile(compare_add, (rs,1))
        a = (i)*rs
        #python is exclusive... will return start_index to end_
        #index-1
        b = ((i+1)*rs)    
        compare_left[a:b, :] = compare_add_mat
        #endfor

    # R_RIGHT -- Stack RS copies of INPUT_MAT on top of itself
    compare_right = np.tile(input_mat, (rs,1))

    # If INPUT_MAT is not binary, create binary temporary objects
    compare_left = compare_left > 0
    compare_right = compare_right > 0


    # Empty matrix to store overlaps
    compare_all = np.zeros((compare_left.shape[0], 1))
    # For each row
    for i in range(compare_left.shape[0]):
        # Create new counter
        num_overlaps = 0
        for j in range(compare_left.shape[1]):
            if compare_left[i,j] ==1  and compare_right[i,j] == 1:
                #inc count
                num_overlaps = num_overlaps+1
            #endif
        #endinnerFor and now append num_overlaps to matrix
        compare_all[i,0] = num_overlaps

    compare_all = (compare_all > 0)
    overlap_mat = np.reshape(compare_all, (rs, rs))
    #print("overla-Mst: \n", overlap_mat)
    # If OVERLAP_MAT is symmetric, only keep the upper-triangular portion. If
    # not, keep all of OVERLAP_MAT.
    check_mat = np.allclose(overlap_mat, overlap_mat.T)
    if(check_mat):
        overlap_mat = np.triu(overlap_mat,1)

    # endif
    overlaps_yn = overlap_mat
    
    return overlaps_yn