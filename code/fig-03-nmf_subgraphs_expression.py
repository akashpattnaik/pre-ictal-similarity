# %%
import numpy as np
from os.path import join as ospj
import pandas as pd
import sys
import json
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

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
electrodes_opt = config['electrodes']
band_opt = config['bands']

data_path = ospj(repo_path, 'data')
figure_path = ospj(repo_path, 'figures')

patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort_test.xlsx"))

# %%
# Plot the NMF subgraphs and expression
for index, row in patient_cohort.iterrows():
    pt = row["Patient"]
    print("Making NMF figures for {}".format(pt))

    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    H = np.load(ospj(pt_data_path, "nmf_coefficients_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))

    sz_id = np.squeeze(sz_id)
    n_components = H.shape[0]
    
    for i in remaining_sz_ids:
        fig, ax = plt.subplots()
        t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60

        W_norm = normalize(W, norm='l1')
        ax.plot(t_arr_min, movmean(W_norm[sz_id == i, 1:].T, k=100).T)

        ax.set_xlabel("Time from seizure onset (min)")
        ax.set_ylabel("Subgraph coefficient")
        ax.set_title("Seizure {}".format(i))
        ax.legend(np.arange(n_components - 1) + 2, title="Component")

        plt.savefig(ospj(pt_figure_path, "expression_band-{}_elec-{}_sz-{}.svg".format(band_opt, electrodes_opt, i)), bbox_inches='tight', transparent='true')
        plt.savefig(ospj(pt_figure_path, "expression_band-{}_elec-{}_sz-{}.png".format(band_opt, electrodes_opt, i)), bbox_inches='tight', transparent='true')
        # plt.close()

    # ax = plot_spectrogram(H, start_time=0, end_time=n_components)
    # ax.set_title("{}".format(pt))
    # ax.set_xlabel("Component")

    # plt.savefig(ospj(pt_figure_path, "subgraphs_band-{}_elec-{}_sz-{}.svg".format(band_opt, electrodes_opt, i)), bbox_inches='tight', transparent='true')
    # plt.savefig(ospj(pt_figure_path, "subgraphs_band-{}_elec-{}_sz-{}.svg".format(band_opt, electrodes_opt, i)), bbox_inches='tight', transparent='true')
    # plt.close()


# %%
