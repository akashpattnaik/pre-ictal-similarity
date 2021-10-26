import numpy as np

def pull_sz_starts(patient, metadata):
    assert(patient in metadata)
    sz_names = metadata[patient]["Events"]["Ictal"]
    n_sz = len(sz_names)

    sz_starts = np.zeros(n_sz, dtype=np.int64)
    for i_sz, sz_name in enumerate(sz_names):
        sz_starts[i_sz] = sz_names[sz_name]["SeizureEEC"]  
    
    return np.unique(sz_starts)
