import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
import numpy as np

def plot_spectrogram(heatmap, start_time, end_time):
    '''
    heatmap should be time by features
    
    '''
    n_features = heatmap.shape[-1]
    n_channels = int(n_features / 6)

    fig, ax = plt.subplots()

    # for colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    # ax.set_title("Patient: {} Seizure #: {}".format(patient, i_sz))
    ax.set_xlabel("Time from seizure onset (sec)")
    cax.set_ylabel("Bandpower (dB)", labelpad=20)
    ax.axvline(0, c='k', ls='--')

    # y_axis formatting
    ax.set_yticks(np.arange(6)*n_channels)
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.arange(6)*n_channels + n_channels/2))
    ax.yaxis.set_minor_formatter(ticker.FixedFormatter([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'low-$\gamma$', r'high-$\gamma$']))

    im = ax.imshow(
        heatmap.T, 
        aspect='auto', 
        interpolation='none',
        extent=[start_time, end_time, 0, n_features],
        origin='lower',
        cmap='BuPu')
    
    fig.colorbar(im, cax=cax, orientation='vertical')

    return ax