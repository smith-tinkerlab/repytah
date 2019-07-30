import numpy as np
import scipy.spatial.distance as spd

def create_sdm(fv_mat, num_fv_per_shingle):
    """
    Creates self-dissimilarity matrix; this matrix is found by creating audio 
    shingles from feature vectors, and finding cosine distance between 
    shingles
    
    Args
    ----
    fv_mat: np.array
        matrix of feature vectors where each column is a timestep and each row
        includes feature information i.e. an array of 144 columns/beats and 12
        rows corresponding to chroma values
        
    num_fv_per_shingle: int
        number of feature vectors per audio shingle
    
    Returns
    -------
    self_dissim_mat: np.array 
        self dissimilarity matrix with paired cosine distances between 
        shingles
    """
    [num_rows, num_columns] = fv_mat.shape
    if num_fv_per_shingle == 1:
        mat_as = fv_mat
    else:
        mat_as = np.zeros(((num_rows * num_fv_per_shingle),
                           (num_columns - num_fv_per_shingle + 1)))
        for i in range(1, num_fv_per_shingle+1):
            # Use feature vectors to create an audio shingle
            # for each time step and represent these shingles
            # as vectors by stacking the relevant feature
            # vectors on top of each other
            mat_as[((i-1)*num_rows+1)-1:(i*num_rows), : ] = 
	    fv_mat[ : , i-1 :(num_columns - num_fv_per_shingle + i)]

    sdm_row = spd.pdist(mat_as.T, 'cosine')
    self_dissim_mat = spd.squareform(sdm_row)
    return self_dissim_mat
