import sys
import os
from os import path as op
import numpy as np
import shutil

sys.path.append('../')
from osl_tokenize.models import conv as tokenize
from osl_tokenize.simulation import bursts as burst_sims
from glob import glob


def run_tokenizer(dataset_name, SIM_SNR=3, USING_BMRC=False, VOCAB_SIZE=128, load_dataset_dir=""):

    '''
    Run tokenizer on real data (e.g. Camcan), or simulated data and run tokenizer on that

    Parameters
    ----------
    dataset_name : str
        Name of dataset
    SIM_SNR : float
        SNR to use for simulated data
    USING_BMRC : bool
        Using BMRC cluster
    VOCAB_SIZE : int
        Max size of vocabulary
    
    Returns
    -------
    None

    Examples
    --------

    >>> run_tokenizer('gaussian_sim', SIM_SNR=3)
    >>> run_tokenizer('burst_sim_large', SIM_SNR=3)

    Or from the command line:

    python ~/dev/projects/osl-tokenize/examples/run_tokenizer.py gaussian_sim --sim_snr=4
    python ~/dev/projects/osl-tokenize/examples/run_tokenizer.py burst_sim_large --sim_snr=2
    python ~/dev/projects/osl-tokenize/examples/run_tokenizer.py burst_sim_medium --sim_snr=2
    python ~/dev/projects/osl-tokenize/examples/run_tokenizer.py burst_sim_small --sim_snr=2

    '''

    #####################################

    if USING_BMRC:
        dev_dir = '/well/woolrich/users/vxw496'
    else:
        dev_dir = './dev'

    osl_tokenize_dir = f'{dev_dir}/projects/osl-tokenize'
    results_dir = f'{dev_dir}/results'

    sys.path.append(osl_tokenize_dir)

    #####################################
    # Settings

    # if True, then randomize tokens 
    # For testing null case where token labels are completely
    # random such that when ephys-gpt is run the result should be
    # 1/num_tokens.
    # Note that gaussian_sim, although just white noise, will still
    # have a non-uniform distribution of token occurences
    # and so the result will be >> 1/num_tokens
    random_tokens = False 

    TRUE_FREQS = None
    NUM_CHANS = None
    CHAN_ACTIVITY = None

    if dataset_name == 'camcan':
        NUM_SUBJECTS = 50
        LEARNING_RATE = 0.00001

    elif dataset_name == 'burst_sim_small':
        NUM_GROUPS = 1
        NUM_SUBJECTS_PERGROUP = 1
        NUM_MODES = 2
        CHANS_PER_MODE = 1
        NTOTAL_TPTS = 1000000 

    elif dataset_name == 'burst_sim_medium_small':

        NUM_GROUPS = 1
        if USING_BMRC:
            NUM_SUBJECTS_PERGROUP = 6
        else:
            NUM_SUBJECTS_PERGROUP = 6

        NUM_SUBJECTS = NUM_GROUPS*NUM_SUBJECTS_PERGROUP
        CHANS_PER_MODE = -1 # will not be used
        
        TRUE_FREQS = np.array([6.0,10.0,20.0])

        NUM_MODES = TRUE_FREQS.shape[0]
        NUM_CHANS = 1
        CHAN_ACTIVITY = np.ones([NUM_MODES, NUM_CHANS])

        print('TRUE_FREQS')
        print(TRUE_FREQS)
        print('CHAN_ACTIVITY')
        print(CHAN_ACTIVITY)

        NTOTAL_TPTS = 10*NUM_CHANS*NUM_GROUPS*NUM_SUBJECTS_PERGROUP*5*60*100

    elif dataset_name == 'burst_sim_medium':

        NUM_GROUPS = 2
        if USING_BMRC:
            NUM_SUBJECTS_PERGROUP = 4
        else:
            NUM_SUBJECTS_PERGROUP = 4

        NUM_SUBJECTS = NUM_GROUPS*NUM_SUBJECTS_PERGROUP
        CHANS_PER_MODE = -1 # will not be used
        
        TRUE_FREQS = np.array([3.0, 6.0, 10.0, 20.0])

        NUM_MODES = TRUE_FREQS.shape[0]
        NUM_CHANS = 9

        CHAN_ACTIVITY = np.zeros([NUM_MODES, NUM_CHANS])
        CHAN_ACTIVITY[:2, :] = 1 # all channels active with first two modes
        CHAN_ACTIVITY[:, :NUM_CHANS//3] = 1 # first third of channels active with all modes
        CHAN_ACTIVITY[2, NUM_CHANS//3:2*NUM_CHANS//3] = 1 # second third of channels active with mode 2 
        CHAN_ACTIVITY[3, 2*NUM_CHANS//3:] = 1 # last third of channels active with mode 3

        print('TRUE_FREQS')
        print(TRUE_FREQS)
        print('CHAN_ACTIVITY')
        print(CHAN_ACTIVITY)

        NTOTAL_TPTS = NUM_CHANS*NUM_GROUPS*NUM_SUBJECTS_PERGROUP*5*60*100

    elif dataset_name == 'burst_sim_large':
        NUM_GROUPS = 3
        if USING_BMRC:
            NUM_SUBJECTS_PERGROUP = 20
        else:
            NUM_SUBJECTS_PERGROUP = 4

        NUM_SUBJECTS = NUM_GROUPS*NUM_SUBJECTS_PERGROUP
        CHANS_PER_MODE = -1 # will not be used
        
        TRUE_FREQS = np.array([3.0, 6.0, 10.0, 20.0])

        NUM_MODES = TRUE_FREQS.shape[0]
        NUM_CHANS = 12

        CHAN_ACTIVITY = np.zeros([NUM_MODES, NUM_CHANS])
        CHAN_ACTIVITY[:2, :] = 1 # all channels active with first two modes
        CHAN_ACTIVITY[:, :NUM_CHANS//3] = 1 # first third of channels active with all modes
        CHAN_ACTIVITY[2, NUM_CHANS//3:2*NUM_CHANS//3] = 1 # second third of channels active with mode 2 
        CHAN_ACTIVITY[3, 2*NUM_CHANS//3:] = 1 # last third of channels active with mode 3

        print('TRUE_FREQS')
        print(TRUE_FREQS)
        print('CHAN_ACTIVITY')
        print(CHAN_ACTIVITY)

        NTOTAL_TPTS = NUM_CHANS*NUM_GROUPS*NUM_SUBJECTS_PERGROUP*5*60*100

    elif dataset_name == 'gaussian_sim':
        # Note that gaussian_sim, although just white noise, will still
        # have a non-uniform distribution of token occurences
        # and so the result will be >> 1/num_tokens
        NUM_GROUPS = 1
        NUM_SUBJECTS_PERGROUP = 1
        NUM_MODES = 1
        CHANS_PER_MODE = 1 # in this case NUM_CHANS = CHANS_PER_MODE
        NTOTAL_TPTS = 1000000
    elif dataset_name == 'load_dataset':
        
        NUM_GROUPS = 1
        NUM_SUBJECTS_PERGROUP = 1
        NUM_MODES = 1
        CHANS_PER_MODE = 1 # in this case NUM_CHANS = CHANS_PER_MODE
        NTOTAL_TPTS = 640004 # 1000000
        

    else:
        ValueError(f"Dataset {dataset_name} not recognized")

    if 'sim' in dataset_name or 'load' in dataset_name:
        NUM_SUBJECTS = NUM_GROUPS*NUM_SUBJECTS_PERGROUP
        if NUM_CHANS is None:
            NUM_CHANS = NUM_MODES*CHANS_PER_MODE
        LEARNING_RATE = 0.0001  
        dataset_name = f'{dataset_name}_snr{SIM_SNR}_cha{NUM_CHANS}_sub{NUM_SUBJECTS}_gro{NUM_GROUPS}_mod{NUM_MODES}'

        NTPTS_PER_CHAN_PER_SESSION = NTOTAL_TPTS // (NUM_CHANS*NUM_SUBJECTS)

        print(f"NTPTS_PER_CHAN_PER_SESSION={NTPTS_PER_CHAN_PER_SESSION}")

    tokenize_dir = f'{results_dir}/osl-tokenize/{dataset_name}'
    raw_data_dir = f'{results_dir}/raw_data/{dataset_name}' # dir where raw (untokenized) data is stored

    model_dir = f'{tokenize_dir}/token_model'
    tokenize_data_dir = f'{tokenize_dir}/tokenized_data'
    plot_dir = f"{tokenize_dir}/plots"
    
    # if dataset_name contains 'sim' then delete existing raw_data_dir and remake
    if 'sim' in dataset_name:
        if op.exists(raw_data_dir):
            print(f"Deleting existing raw_data_dir {raw_data_dir}")
            
            shutil.rmtree(raw_data_dir)
        os.makedirs(raw_data_dir)
    
    # Delete existing tokenize_dir and remake
    if op.exists(tokenize_dir):
        print(f"Deleting existing tokenize_dir {tokenize_dir}")
        
        shutil.rmtree(tokenize_dir)

    os.makedirs(tokenize_dir)
    os.makedirs(plot_dir)
    os.makedirs(tokenize_data_dir)

    #####################################
    # Simulate data

    if dataset_name == 'camcan':       
        
        data_files = sorted(glob(f'{raw_data_dir}/array*.npy'))
        data_files = data_files[:NUM_SUBJECTS]

    elif 'gaussian_sim' in dataset_name:

        FS = 100
        data_files = burst_sims.simulate(raw_data_dir,
                                        NTPTS=NTPTS_PER_CHAN_PER_SESSION, 
                                        NUM_MODES=0, 
                                        CHANS_PER_MODE=CHANS_PER_MODE, 
                                        NUM_GROUPS=NUM_GROUPS, 
                                        NUM_SUBJECTS_PERGROUP=NUM_SUBJECTS_PERGROUP,
                                        TRUE_FREQS = TRUE_FREQS,
                                        CHAN_ACTIVITY = CHAN_ACTIVITY,                                     
                                        FS=FS, SNR=SIM_SNR) # since NUM_MODES=0, NUM_CHANS is CHANS_PER_MODE

    elif 'burst_sim' in dataset_name:

        FS = 100
        data_files = burst_sims.simulate(raw_data_dir, 
                                        NTPTS=NTPTS_PER_CHAN_PER_SESSION, 
                                        NUM_MODES=NUM_MODES, 
                                        CHANS_PER_MODE=CHANS_PER_MODE, 
                                        NUM_GROUPS=NUM_GROUPS, 
                                        NUM_SUBJECTS_PERGROUP=NUM_SUBJECTS_PERGROUP,
                                        TRUE_FREQS = TRUE_FREQS,
                                        CHAN_ACTIVITY = CHAN_ACTIVITY,                                         
                                        FS=FS, SNR=SIM_SNR)

        if NUM_SUBJECTS>1:
            ground_truth_dir = f"{raw_data_dir}/ground_truth"
            burst_sims.plot_data(raw_data_dir, plot_dir, NUM_SUBJECTS, FS)

    elif 'load' in dataset_name:
        print("Loading dataset from location:", load_dataset_dir)
        FS = 100   ## what is that??
        data_files = [load_dataset_dir]

    else:
        ValueError(f"Dataset {dataset_name} not recognized")

    print('Session 0 data shape is:')
    print(data_files)
    print(np.load(data_files[0]).shape)
    
    #####################################
    # Fit model

    config = tokenize.Config(VOCAB_SIZE=VOCAB_SIZE,                        
                             LEARNING_RATE=LEARNING_RATE,)
    
    token_model = tokenize.Model(config)
    token_model.fit(data_files)

    if random_tokens:
        token_model.refactor_vocab(data_files, sort=True, trim=True, random_tokens=token_model)

    token_model.save(model_dir)

    ################################
    if USING_BMRC:
        local_results_dir = f'/Users/woolrich/dev/results_bmrc'
        local_tokenize_dir = f'{local_results_dir}/osl-tokenize'
        local_raw_data_dir = f'{local_results_dir}/raw_data'

        print('Run something like this on local machine to copy results:')
        print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{tokenizer_results_dir} {local_tokenize_dir}/')

        if 'sim' in dataset_name:
            print('Run something like this on local machine to copy raw simulated data:')
            print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{raw_data_dir}/{dataset_name} {local_raw_data_dir}/')

    return tokenize_dir, raw_data_dir
  
