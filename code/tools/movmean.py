import numpy as np

def movmean(x, k):
    if x.ndim == 1:
        # return np.convolve(np.pad(x, (k - 1, 0), mode='edge'), np.ones(k)/k, mode='valid')
        return np.convolve(x, np.ones(k)/k, mode='same')
    else:
        avgd_x = np.zeros(x.shape)
        for i, row in enumerate(x):
            avgd_x[i, :] = np.convolve(row, np.ones(k)/k, mode='same')
        return avgd_x

