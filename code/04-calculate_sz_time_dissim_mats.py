'''
This script uses dynamic time warping and euclidean distance to crease dissimilarity matrices between pairs of seizures.
In addition, this script creates temporal and circadian dissimilarity matrices.
'''
# %%
# Imports and environment setup
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy.io import loadmat
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from os.path import join as ospj

sys.path.append('tools')

from pull_sz_starts import pull_sz_starts

from mpl_toolkits.axes_grid1 import make_axes_locatable

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

PLOT = True

# Make patient directories
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    if not os.path.exists(pt_data_path):
        os.makedirs(pt_data_path)
    if not os.path.exists(pt_figure_path):
        os.makedirs(pt_figure_path)

# %% Function for plotting dissimilarity matrices
def plot_dissim_mat(dissim_mat, cbar_label, fig_name, title=None, cmap="BuPu"):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    im = ax.imshow(dissim_mat, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel(cbar_label, labelpad=15, color=palette['1'])
    cax.yaxis.set_tick_params(color=palette['1'], labelcolor=palette['1'])

    ax.tick_params(axis='x', colors=palette['1'], which='both')
    ax.tick_params(axis='y', colors=palette['1'], which='both')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(range(n_sz))
    ax.set_xticklabels(np.arange(n_sz, dtype=int) + 1, rotation=90)
    ax.set_yticks(range(n_sz))
    ax.set_yticklabels(np.arange(n_sz, dtype=int) + 1)
    ax.set_xlabel("Seizure", color=palette['1'])
    ax.set_ylabel("Seizure", color=palette['1'])

    if title:
        ax.set_title(title, color=palette['1'])

    plt.savefig(ospj(pt_figure_path, "{}.svg".format(fig_name)), transparent=True, bbox_inches='tight')
    plt.savefig(ospj(pt_figure_path, "{}.png".format(fig_name)), transparent=True, bbox_inches='tight')
    plt.close()

    return ax
# %% Calcualate seizure dissimilarities
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print(pt)

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    bandpower_mat_data = loadmat(ospj(pt_data_path, "bandpower-windows-sz.mat"))
    bandpower_data = 10*np.log10(bandpower_mat_data['allFeats'])
    t_sec = np.squeeze(bandpower_mat_data['entireT']) / 1e6
    sz_id = np.squeeze(bandpower_mat_data['szID'])

    n_sz = np.size(np.unique(sz_id))

    # Seizure dissimilarity
    sz_dissim_mat = np.zeros((n_sz, n_sz))
    for i_sz in range(1, n_sz + 1):
        for j_sz in range(1, n_sz + 1):
            if i_sz != j_sz:
                distance, path = fastdtw(bandpower_data[sz_id == i_sz, :], bandpower_data[sz_id == j_sz, :], dist=euclidean)
                sz_dissim_mat[i_sz - 1, j_sz - 1] = distance

    np.save(ospj(pt_data_path, "sz_dissim_mat.npy"), sz_dissim_mat)

    if PLOT:
        plot_dissim_mat(sz_dissim_mat, "Dissimilarity", "sz_dissim_mat", "Seizure dissimilarity", cmap="BuPu")

    break
# %% time and circadian difference matrix
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    print(pt)

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)
    
    sz_starts = pull_sz_starts(pt, metadata)

    # time dissimilarity
    time_dissim_mat = np.abs(sz_starts[:, None] - sz_starts[None, :]) / 60 / 60

    np.save(ospj(pt_data_path, "time_dissim_mat.npy"), time_dissim_mat)
    if PLOT:
        plot_dissim_mat(time_dissim_mat, "Time Difference (hrs)", "time_dissim_mat", "Temporal dissimilarity", cmap="BuPu")

    # circadian dissimilarity
    circadian_dissim_mat = np.abs(sz_starts[:, None] % (60 * 60 * 24) - sz_starts[None, :] % (60 * 60 * 24)) / 60 / 60

    np.save(ospj(pt_data_path, "circadian_dissim_mat.npy"), circadian_dissim_mat)
    if PLOT:
        plot_dissim_mat(circadian_dissim_mat, "Time of Day Difference (hrs)", "circadian_dissim_mat", "Circadian dissimilarity", cmap="BuPu")

# %%
