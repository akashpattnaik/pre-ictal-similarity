'''
This script moves the list of files from the data directory of the hmm-emu-state-space repository to the data directory of this repository
'''
# %%
import shutil
import pandas as pd
from os.path import join as ospj
import os

root_path = os.path.dirname(os.path.realpath(__file__))
root_path = "/".join(root_path.split("/")[:-1])
data_path = ospj(root_path, 'data')


patient_cohort = pd.read_excel(ospj(data_path, "patient_cohort.xlsx"))

# %%
old_data_path_root = "/gdrive/public/USERS/pattnaik/hmm-emu-state-space/data"
new_data_path_root = "../data"

files = [
    'target-electrodes-regions.mat',
    'target-electrodes-all.mat'
    ]
    
for pt in patient_cohort["Patient"]:
    print(pt)
    for file in files:
        # check if directory exists
        if not os.path.exists(ospj(new_data_path_root, pt)):
            os.makedirs(ospj(new_data_path_root, pt))

        old_data_path = ospj(old_data_path_root, pt, file)
        new_data_path = ospj(new_data_path_root, pt, file)

        if os.path.exists(old_data_path):
            shutil.copy(old_data_path, new_data_path)
        else:
            print("{} does not exist".format(old_data_path))
# %%
