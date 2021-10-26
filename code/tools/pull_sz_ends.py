import numpy as np

def pull_sz_ends(patient, metadata):
    assert(patient in metadata)
    sz_names = metadata[patient]["Events"]["Ictal"]
    n_sz = len(sz_names)

    sz_ends = np.zeros(n_sz, dtype=np.int64)
    for i_sz, sz_name in enumerate(sz_names):
        sz_ends[i_sz] = sz_names[sz_name]["SeizureEnd"]  
    
    return np.unique(sz_ends)
