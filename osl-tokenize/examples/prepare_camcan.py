# to be run on BMRC cluster 

from glob import glob
#from osl_dynamics.data import Data
import numpy as np
import os
import os.path as op
import pandas as pd

NUM_SUBJECTS = 500
files = sorted(glob('/well/woolrich/projects/camcan/spring23/src/*/sflip_parc-raw.fif'))
files = files[:NUM_SUBJECTS]

base_dir = f'/well/woolrich/users/vxw496/results/tokenize/camcan'
if not op.exists(base_dir):
    os.makedirs(base_dir)

if False:

    data_dir = f'{base_dir}/data'
    data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=8)
    data.save(data_dir)
    data.delete_dir()

    # test by loading data
    sflip_parc_files = sorted(glob(f'{data_dir}/array*.npy'))
    sflip_parc_files = sflip_parc_files[:NUM_SUBJECTS]

    for sessdatafile in sflip_parc_files:
        print(sessdatafile)

        # load session data
        sessdata = np.load(sessdatafile) # ntpts x nchans

        print(f"ntpts, nchans = {sessdata.shape}")
    

# Subject IDs
subjects = [file.split("/")[-2] for file in files]

# Get demographics
participants = pd.read_csv("/well/woolrich/projects/camcan/participants.tsv", sep="\t")

# Get ages
ages = []
for id in subjects:
    age = participants.loc[participants["participant_id"] == id]["age"].values[0]
    ages.append(age)

# Save
np.save(f"{base_dir}/age.npy", ages)
