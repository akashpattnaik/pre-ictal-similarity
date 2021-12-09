# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os.path import join as ospj
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

# %%
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    soz_subgraph_dissim_mat = np.load(ospj(pt_data_path, "soz_subgraph_dissim_mat_{}_{}.npy".format(electrodes_opt, band_opt)))
    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))

    n_sz = np.size(remaining_sz_ids)
    cmap='BuPu'
    title='SOZ Subgraph Dissimilarity'

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(soz_subgraph_dissim_mat, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel("SOZ Subgraph Expression Dissimilarity (a.u.)", labelpad=15, color=palette['1'])
    cax.yaxis.set_tick_params(color=palette['1'], labelcolor=palette['1'])

    ax.tick_params(axis='x', colors=palette['1'], which='both')
    ax.tick_params(axis='y', colors=palette['1'], which='both')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(range(n_sz))
    ax.set_xticklabels(remaining_sz_ids, rotation=90)
    ax.set_yticks(range(n_sz))
    ax.set_yticklabels(remaining_sz_ids)
    ax.set_xlabel("Seizure", color=palette['1'])
    ax.set_ylabel("Seizure", color=palette['1'])

    if title:
        ax.set_title(title, color=palette['1'])
    plt.savefig(ospj(pt_figure_path, "soz_subgraph_dissim_mat_{}_{}.svg".format(electrodes_opt, band_opt)), bbox_inches='tight', transparent='true')
    plt.savefig(ospj(pt_figure_path, "soz_subgraph_dissim_mat_{}_{}.png".format(electrodes_opt, band_opt)), bbox_inches='tight', transparent='true')
    plt.close(fig)


    sz_dissim_mat = np.load(ospj(pt_data_path, "sz_dissim_mat_dtw_{}_{}.npy".format(electrodes_opt, band_opt)))

    remaining_sz_dissim_mat = sz_dissim_mat[remaining_sz_ids[:, None] - 1, remaining_sz_ids - 1]
    title='Seizure Dissimilarity'

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    
    im = ax.imshow(remaining_sz_dissim_mat, cmap=cmap)
    fig.colorbar(im, cax=cax, orientation='vertical')
    cax.set_ylabel("Seizure Dissimilarity (a.u.)", labelpad=15, color=palette['1'])
    cax.yaxis.set_tick_params(color=palette['1'], labelcolor=palette['1'])

    ax.tick_params(axis='x', colors=palette['1'], which='both')
    ax.tick_params(axis='y', colors=palette['1'], which='both')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks(range(n_sz))
    ax.set_xticklabels(remaining_sz_ids, rotation=90)
    ax.set_yticks(range(n_sz))
    ax.set_yticklabels(remaining_sz_ids)
    ax.set_xlabel("Seizure", color=palette['1'])
    ax.set_ylabel("Seizure", color=palette['1'])

    if title:
        ax.set_title(title, color=palette['1'])
    plt.savefig(ospj(pt_figure_path, "remaining_sz_dissim_mat_{}_{}.svg".format(electrodes_opt, band_opt)), bbox_inches='tight', transparent='true')
    plt.savefig(ospj(pt_figure_path, "remaining_sz_dissim_mat_{}_{}.png".format(electrodes_opt, band_opt)), bbox_inches='tight', transparent='true')
    plt.close(fig)

    break
# %%
