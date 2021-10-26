import numpy as np

def time2ind(time, time_arr):
    if isinstance(time, np.ndarray):
        inds = [np.int64(np.argmax(time_arr > sz_time) - 1) for sz_time in time]
        return np.array(inds)
    else:
        ind = np.int64(np.argmax(time_arr > time) - 1)
        return ind
