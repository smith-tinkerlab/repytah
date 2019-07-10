import numpy as np 

def merge_based_on_length(full_mat,full_bandwidth,target_bandwidth):
    """
    Merges rows of full_mat that contain repeats that are the same 
        length and are repeats of the same piece of structure

    Args
    ----
    full_mat: np.array
        binary matrix with ones where repeats start and zeroes otherwise
        
    full_bw: np.array
        length of repeats encoded in input_mat
    
    target_bw: np.array
        lengths of repeats that we seek to merge
        
    Returns
    -------    
    out_mat: np.array
        binary matrix with ones where repeats start and zeros otherwise
        with rows of FULL_MAT merged if appropriate
        
    one_length_vec: np.array
        length of the repeats encoded in out_mat
    """
    temp_bandwidth = np.sort(full_bandwidth,axis=None)
    bnds = np.argsort(full_bandwidth,axis=None)
    temp_mat = full_mat[bnds,:] 
    
    # Find the list of unique lengths that you would like to search
    target_bandwidth = np.unique(target_bandwidth)
    T = target_bandwidth.shape[0] 
    
    for i in range(1,T+1):
        test_bandwidth = target_bandwidth[i-1]
        inds = (temp_bandwidth == test_bandwidth) # Check if temp_bandwidth is equal to test_bandwidth
        # If the sum of all inds is greater than 1, then execute the if statement
        if inds.sum() > 1:
            # Isolate rows that correspond to TEST_BW and merge them
            toBmerged = temp_mat[inds,:]
            # Function call - merge_rows
            merged_mat = merge_rows(toBmerged, test_bandwidth)

            bandwidth_add_size = merged_mat.shape[0] # Number of columns
            bandwidth_add = test_bandwidth * np.ones((bandwidth_add_size,1))

            temp_mat[inds,:] = np.array([])
            temp_bandwidth[inds,:] = np.array([])

            temp_mat = np.concatenate((temp_mat,merged_mat))
            temp_bandwidth = np.concatenate((temp_bandwidth,bandwidth_add))

            temp_bandwidth = np.sort(temp_bandwidth,axis=None)
            bnds = np.argsort(temp_bandwidth,axis=None)
            temp_mat = temp_mat[bnds,:]

    out_mat = temp_mat
    out_length_vec = temp_bandwidth
    
    return out_mat,out_length_vec
