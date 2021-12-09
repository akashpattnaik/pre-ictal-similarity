# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join as ospj
import pandas as pd
import json
from sklearn.preprocessing import normalize
import sys
sys.path.append('tools')

from movmean import movmean

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

# %%

for index, row in patient_cohort.iterrows():
    if row['Ignore']:
        continue

    pt = row["Patient"]
    print("Making soz subgraph figures for {}".format(pt))
    pt_data_path = ospj(data_path, pt)
    pt_figure_path = ospj(figure_path, pt)

    remaining_sz_ids = np.load(ospj(pt_data_path, "remaining_sz_ids.npy"))
    t_sec = np.load(ospj(pt_data_path, "lead_sz_t_sec_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    sz_id = np.load(ospj(pt_data_path, "lead_sz_sz_id_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    W = np.load(ospj(pt_data_path, "nmf_expression_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))
    H = np.load(ospj(pt_data_path, "nmf_coefficients_band-{}_elec-{}.npy".format(band_opt, electrodes_opt)))

    sz_id = np.squeeze(sz_id)
    soz_electrodes = np.load(ospj(pt_data_path, "soz_electrodes_{}_{}.npy".format(electrodes_opt, band_opt)))
    pt_soz_state = np.load(ospj(pt_data_path, "pt_soz_state_{}_{}.npy".format(electrodes_opt, band_opt)))

    print("\tPatient's seizure onset state is {}".format(pt_soz_state))
    

    n_components = H.shape[0]

    n_electrodes = soz_electrodes.shape[0] 

    # plot all states
    component_arr = np.reshape(H, (n_components, -1, n_electrodes))
    # component_z = np.zeros(component_arr.shape)
    # for i_comp in range(n_components):
    #     component_z[i_comp, :, :] = zscore(component_arr[i_comp, :, :], axis=1)
    n_bands = component_arr.shape[1]
    # sort to put non-soz first
    sort_soz_inds = np.argsort(soz_electrodes)
    n_soz = np.sum(soz_electrodes)
    n_non_soz = n_electrodes - n_soz

    for i_comp in range(n_components):
        fig, ax = plt.subplots()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        im = ax.imshow(
            component_arr[i_comp, :, sort_soz_inds].T, 
            aspect='auto', 
            interpolation='none',
            origin='lower')

        ax.axvline(n_non_soz - 0.5, c='r', lw=2)
        ax.set_title("Subgraph {}, {}".format(i_comp, pt))
        
        ax.set_yticks(np.arange(n_bands))
        if band_opt == 'all':
            ax.set_yticklabels([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$', r'low-$\gamma$', r'high-$\gamma$'])
        elif band_opt == 'broad':
            ax.set_yticklabels(["broadband"])

        ax.set_xticks(np.arange(n_electrodes))
        ax.set_xticks([n_non_soz / 2, n_non_soz + n_soz / 2])
        ax.set_xticklabels(["Non SOZ", "SOZ"])
        
        ax.set_xlabel("Electrodes")
        ax.set_ylabel("Frequency band")

        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        cbar.ax.set_ylabel('Power (dB)', rotation=90)

        plt.savefig(ospj(pt_figure_path, "soz_heatmap_band-{}_elec-{}_subgraph-{}.svg".format(band_opt, electrodes_opt, i_comp)), bbox_inches='tight', transparent='true')
        plt.savefig(ospj(pt_figure_path, "soz_heatmap_band-{}_elec-{}_subgraph-{}.png".format(band_opt, electrodes_opt, i_comp)), bbox_inches='tight', transparent='true')
        # plt.close(fig)

    for i in remaining_sz_ids:
        k = 100
        fig, ax = plt.subplots()
        t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60
        W_norm = normalize(W, norm='l1')
        ax.plot(t_arr_min, movmean(W_norm[sz_id == i, pt_soz_state].T, k=k).T)

        ax.set_xlabel("Time from seizure onset (min)")
        ax.set_ylabel("Subgraph coefficient")
        ax.set_title("Seizure {}".format(i))
        # ax.set_xlim([-200, 0])

        plt.savefig(ospj(pt_figure_path, "soz_expression_band-{}_elec-{}_sz-{}_k-{}.svg".format(band_opt, electrodes_opt, i, k)), bbox_inches='tight', transparent='true')
        plt.savefig(ospj(pt_figure_path, "soz_expression_band-{}_elec-{}_sz-{}_k-{}.png".format(band_opt, electrodes_opt, i, k)), bbox_inches='tight', transparent='true')
        # plt.close(fig)


# patient_cohort.to_csv(ospj(data_path, "patient_cohort_with_soz_states.csv"))

# # %%
# for i in remaining_sz_ids:
#     fig, ax = plt.subplots()
#     t_arr_min = (t_sec[sz_id == i] - t_sec[sz_id == i][-1]) / 60
#     W_norm = normalize(W, norm='l1')
#     ax.plot(t_arr_min, movmean(W_norm[sz_id == i, pt_soz_state].T, k=500).T)

#     ax.set_xlabel("Time from seizure onset (min)")
#     ax.set_ylabel("Subgraph coefficient")
#     ax.set_title("Seizure {}".format(i))
#     # ax.set_xlim([-200, 0])

# %%
