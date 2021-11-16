# %%
import numpy as np
from os.path import join as ospj
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
sys.path.append('tools')

from movmean import movmean
from pull_sz_starts import pull_sz_starts
from plot_spectrogram import plot_spectrogram


# Get paths from config file and metadata
with open("config.json") as f:
    config = json.load(f)
repo_path = config['repositoryPath']
metadata_path = config['metadataPath']
palette = config['lightColors']
DTW_FLAG = config['flags']["DTW_FLAG"]
mode = config['mode']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# %%
# Plot the NMF subgraphs and expression
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    pt = "HUP130"
    print("Making NMF figures for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_{}.npy".format(mode)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_{}.npy".format(mode)))
    W = np.load(ospj(pt_data_path, "nmf_expression_{}.npy".format(mode)))
    H = np.load(ospj(pt_data_path, "nmf_coefficients_{}.npy".format(mode)))

    n_components = H.shape[0]
    
    for i in remaining_sz_ids:
        fig, ax = plt.subplots()
        t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60

        ax.plot(t_arr_min, movmean(W[sz_id == i, 1:].T, k=1, mode='same').T)

        ax.set_xlabel("Time from seizure onset (min)")
        ax.set_ylabel("Subgraph coefficient")
        ax.set_title("Seizure {}".format(i))
        ax.legend(np.arange(n_components - 1) + 2, title="Component")

        plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}_{}.svg".format(i, mode)), bbox_inches='tight', transparent='true')
        plt.savefig(ospj(pt_figure_path, "subgraph_expression_sz_{}_{}.png".format(i, mode)), bbox_inches='tight', transparent='true')
        # plt.close()

    ax = plot_spectrogram(H, start_time=0, end_time=n_components)
    ax.set_title("{}".format(pt))
    ax.set_xlabel("Component")

    plt.savefig(ospj(pt_figure_path, "subgraphs_{}.svg".format(mode)), bbox_inches='tight', transparent='true')
    plt.savefig(ospj(pt_figure_path, "subgraphs_{}.png".format(mode)), bbox_inches='tight', transparent='true')
    # plt.close()


    break
# %%
