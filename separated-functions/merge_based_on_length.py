import numpy as np

def _merge_based_on_length(full_mat,full_bw,target_bw):
    """
    Merges rows of full_mat that contain repeats that are the same 
        length (as set by full_bandwidth) and are repeats of the 
        same piece of structure.
        
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
        with rows of full_mat merged if appropriate
        
    one_length_vec: np.array
        length of the repeats encoded in out_mat
    """
    temp_bandwidth = np.sort(full_bandwidth,axis=None) # Sort the elements of full_bandwidth
    bnds = np.argsort(full_bandwidth,axis=None) # Return the indices that would sort full_bandwidth
    temp_mat = full_mat[bnds,:] 
    
    target_bandwidth = np.unique(target_bandwidth) # Find the unique elements of target_bandwidth
    target_size = target_bandwidth.shape[0] # Number of columns 
    
    for i in range(1,target_size+1):
        test_bandwidth = target_bandwidth[i-1]
        inds = (temp_bandwidth == test_bandwidth) # Check if temp_bandwidth is equal to test_bandwidth
        
        # If the sum of all inds elements is greater than 1, then execute this if statement
        if inds.sum() > 1:
            # Isolate rows that correspond to test_bandwidth and merge them
            merge_bw = temp_mat[inds,:]
            merged_mat = _merge_rows(merge_bw,test_bandwidth)
        
            bandwidth_add_size = merged_mat.shape[0] # Number of columns
            bandwidth_add = test_bandwidth * np.ones((bandwidth_add_size,1)).astype(int)
         
            if np.any(inds == True):
                # Convert the boolean array inds into an array of integers
                inds = np.array(inds).astype(int)
                remove_inds = np.where(inds == 1)
                # Delete the rows that meet the condition set by remove_inds
                temp_mat = np.delete(temp_mat,remove_inds,axis=0)
                temp_bandwidth = np.delete(temp_bandwidth,remove_inds,axis=0)
    
            # Combine rows into a single matrix
            bind_rows = [temp_mat,merged_mat]
            temp_mat = np.concatenate(bind_rows)

            if temp_bandwidth.size == 0: # Indicates temp_bandwidth is an empty array
                temp_bandwidth = np.concatenate(bandwidth_add)
            elif temp_bandwidth.size > 0: # Indicates temp_bandwidth is not an empty array
                bind_bw = [temp_bandwidth,bandwidth_add]
                temp_bandwidth = np.concatenate(bind_bw)

            temp_bandwidth = np.sort(temp_bandwidth) # Sort the elements of temp_bandwidth
            bnds = np.argsort(temp_bandwidth) # Return the indices that would sort temp_bandwidth
            temp_mat = temp_mat[bnds,]

    out_mat = temp_mat
    out_length_vec = temp_bandwidth
    
    output = (out_mat,out_length_vec)
    
    return output
