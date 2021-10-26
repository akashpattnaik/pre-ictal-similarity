import scipy

def bandpower(x, fs, frange):
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > frange[0]) - 1
    ind_max = scipy.argmax(f > frange[1]) - 1
    return scipy.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])
