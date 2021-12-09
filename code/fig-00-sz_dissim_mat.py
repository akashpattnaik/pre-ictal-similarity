# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from os.path import join as ospj
import json

# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']
DTW_FLAG = config['flags']["DTW_FLAG"]
electrodes_opt = config['electrodes']
band_opt = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

metadata_fname = ospj(metadata_path, "DATA_MASTER.json")
with open(metadata_fname) as f:
    metadata = json.load(f)['PATIENTS']

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

# %% Function for plotting dissimilarity matrices
def fig_sz_dissim_mat(dissim_mat, cbar_label, palette, title=None, cmap="BuPu", savepath=None):
    n_sz = dissim_mat.shape[0]

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

    if savepath is not None:
        plt.savefig("{}.png".format(savepath), transparent=True, bbox_inches='tight')
        plt.savefig("{}.svg".format(savepath), transparent=True, bbox_inches='tight')
        plt.close()

    return fig, ax

# if __name__ == "__main__":
patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))
mode = "all"

for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    print("Making seizure and time dissimilarity figures for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    if DTW_FLAG:
        sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_dtw_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
        savepath = ospj(pt_figure_path, "sz_dissim_mat_dtw_band-{}_elec-{}".format(band_opt, electrodes_opt))
    else:
        sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
        savepath = ospj(pt_figure_path, "sz_dissim_mat_band-{}_elec-{}".format(band_opt, electrodes_opt))
    fig_sz_dissim_mat(
        sz_dissim_mat, 
        "Dissimilarity", 
        palette, 
        title="Seizure dissimilarity", 
        cmap="BuPu", 
        savepath=savepath
        )
    print("\tSeizure dissimilarity figure saved at {}".format(savepath))

    time_dissim_mat = np.load(ospj(pt_data_path, "time_dissim_mat.npy"))
    savepath = ospj(pt_figure_path, "time_dissim_mat")
    fig_sz_dissim_mat(
        time_dissim_mat, 
        "Time Difference (hrs)", 
        palette, 
        title="Temporal dissimilarity", 
        cmap="BuPu", 
        savepath=savepath
        )
    print("\tTime dissimilarity figure saved at {}".format(savepath))

    circadian_dissim_mat = np.load(ospj(pt_data_path, "circadian_dissim_mat.npy"))
    savepath = ospj(pt_figure_path, "circadian_dissim_mat")
    fig_sz_dissim_mat(
        circadian_dissim_mat, 
        "Time of Day Difference (hrs)", 
        palette, 
        title="Circadian dissimilarity", 
        cmap="BuPu", 
        savepath=savepath
        )
    print("\tCircadian dissimilarity figure saved at {}".format(savepath))

# %%
