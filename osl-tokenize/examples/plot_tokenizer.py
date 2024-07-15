'''

Plot results of running tokenizer on real data (e.g. Camcan), or on simulated data

'''

import sys
import os
from os import path as op
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../')
from osl_tokenize.models import conv as tokenize

def plot_tokenizer(tokenize_dir, 
                   raw_data_dir,
                   USING_BMRC=False,  
                   SESS_ID=0, 
                   NUM_SESS=None, 
                   dev_dir=None):

    '''
    Plot results of running tokenizer on real data (e.g. Camcan), or on simulated data
    
    Parameters
    ----------
    tokenize_dir
        Name of dir path for tokenizer results 
        E.g. as outputted by run_tokenizer()
    raw_data_dir : str
        Directory where raw data is stored
        E.g. as outputted by run_tokenizer() 
    USING_BMRC : bool
        Using BMRC cluster
    SESS_ID : int
        Index of session data to plot. Must be less than NUM_SESS
    NUM_SESS : int
        Number of sessions to calculate full stats for
    dev_dir : str
        Directory where dev code is stored
    
    Returns
    -------
    plot_dir : str
        Directory where plots are saved
    
    Examples
    --------
    
    >>> plot_tokenizer('gaussian_sim_snr3.0_cha1_sub1_gro1_mod1', SESS_ID=0)

    Or from the command line:

    python osl-tokenize/examples/plot_tokenizer.py gaussian_sim_snr3.0_cha1_sub1_gro1_mod1 --sess_id=0 
    python osl-tokenize/examples/plot_tokenizer.py burst_sim_medium_small_snr1.0_cha1_sub1_gro1_mod12
    python /well/woolrich/users/vxw496/projects/osl-tokenize/examples/plot_tokenizer.py  burst_sim_very_large_snr1.0_cha51_sub60_gro3_mod16 --using_bmrc=True

    '''

    if dev_dir is None:
        if USING_BMRC:
            dev_dir = '/well/woolrich/users/vxw496'
        else:
            dev_dir = './dev'

    osl_tokenize_dir = f'{dev_dir}/projects/osl-tokenize'
    sys.path.append(osl_tokenize_dir)

    
    
    #####################################
    # Settings

    random_tokens = False

    model_dir = f'{tokenize_dir}/token_model'
    tokenize_data_dir = f'{tokenize_dir}/tokenized_data'
    plot_dir = f"{tokenize_dir}/plots"

    print(f"Using pre-trained model {model_dir}")
    token_model = tokenize.Model(model_dir)

    #####################################
    # Setup data file names

    data_files = []
    for file in os.listdir(raw_data_dir):
        if file.endswith(".npy"):
            data_files.append(op.join(raw_data_dir, file))
            
    if NUM_SESS is not None:
        data_files = data_files[:NUM_SESS]
    
    # error if SESS_ID is not less than NUM_SESS
    NUM_SESS = len(data_files)
    assert SESS_ID < NUM_SESS, f"SESS_ID={SESS_ID} must be less than NUM_SESS={NUM_SESS}"   
        
    ################################

    print(f"VOCAB_SIZE={token_model.config.VOCAB_SIZE}")
    REFACTORED_VOCAB_SIZE = len(token_model.vocab['token_order'])+1 
    print(f"REFACTORED_VOCAB_SIZE={REFACTORED_VOCAB_SIZE}")

    ################################
    # Plot histories
        
    tokenize.plot_history(token_model.histories, plot_dir)

    ################################
    # Tokenize some data and show the fitted signal for SESS_ID

    true_signal_2use = None

    if 'sim' in tokenize_dir:
        ground_truth_dir = raw_data_dir + "/ground_truth"

        true_signal_2use = np.load(ground_truth_dir + f"/true_signal_{SESS_ID}.npy")
        
        # normalize true_signal_2use
        true_signal_2use = true_signal_2use - np.mean(true_signal_2use, axis=0)
        
        if np.std(true_signal_2use) > 0.0001:
            true_signal_2use = true_signal_2use / np.std(true_signal_2use, axis=0)

    original_data = np.load(data_files[SESS_ID])

    # normalize original data
    original_data = (original_data - np.mean(original_data, axis=0)) / np.std(original_data, axis=0)

    tokenized_data_file = token_model.tokenize_data(data_files[SESS_ID], tokenize_data_dir, force=True, random_tokens=random_tokens) # list len nsubj each (channels x time)
    tokenized_data = np.load(tokenized_data_file) # time x channels 
    fitted_signal = token_model.reconstruct_data(tokenized_data) # time x channels 

    # get token_weights
    _, token_weights = token_model._tokenize_data(data_files[SESS_ID])

    INDEX_FROM = 200
    INDEX_TO = INDEX_FROM + 300

    numchans2plot = 3
    numchans2plot = min(numchans2plot, original_data.shape[1])
    chans_to_plot = np.arange(numchans2plot)

    for chan_to_plot in chans_to_plot:

        fig, axs = plt.subplots(2, 1, figsize=(20, 5))

        axs[0].plot(original_data[INDEX_FROM:INDEX_TO, chan_to_plot], 'b')
        axs[0].plot(fitted_signal[INDEX_FROM:INDEX_TO, chan_to_plot], 'r')

        if true_signal_2use is not None:
            true_signal_2use = (true_signal_2use - np.mean(true_signal_2use, axis=0)) / np.std(true_signal_2use, axis=0)
            axs[0].plot(true_signal_2use[INDEX_FROM:INDEX_TO, chan_to_plot], 'y')

        # add legend
        axs[0].legend(['original', 'fitted', 'true'])

        axs[1].plot(token_weights[:INDEX_TO, chan_to_plot, :])

        # save fig
        fig.savefig(f"{plot_dir}/fitted_signal_chan{chan_to_plot}.png")
        plt.close()

    ################################
    # compute counts of neighbouring pairs of tokens

    tokens = np.load(tokenized_data_file)

    lags = [1, 5]
    for lag in lags:
        token_counts = np.zeros((REFACTORED_VOCAB_SIZE, REFACTORED_VOCAB_SIZE))
        for i in range(tokens.shape[0]-lag):
            token_counts[tokens[i], tokens[i+lag]] += 1

        # plot token_counts
        plt.figure(figsize=(10, 10))
        plt.imshow(token_counts, cmap='hot', interpolation='nearest')
        plt.clim(0, np.percentile(token_counts, 99.99))
        plt.colorbar()
        plt.savefig(f"{plot_dir}/neighbouring_token_counts_lag{lag}.png")
        plt.close()

    ################################
    # percent variance explained

    print("Calculating percent variance explained")

    # loop through all sessions calculating pve
    pves = []
    for SESS_ID in range(len(data_files)):
        pve = np.mean(token_model.get_pve(data_files[SESS_ID]))
        pves.append(pve)

    print(f'Variance explained for session {SESS_ID} = {pves[SESS_ID]:.1f}')
    print(f'Variance explained on average = {np.mean(pves):.1f}')
    print(f'pves.shape={len(pves)}')

    # plot histogram of pves
    plt.figure(figsize=(10, 10))
    plt.hist(pves, bins=20)
    plt.title(f"Variance explained on average = {np.mean(pves):.1f} %")
    plt.savefig(f"{plot_dir}/pve_hist.png")
    plt.close()

    ################################
    # plot bar graph of token_counts   

    print("Plotting bar graph of token_counts")

    #refactor_vocab
    token_model.refactor_vocab(data_files=data_files)

    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(token_model.vocab['token_counts'].shape[0]), token_model.vocab['token_counts'])
    plt.savefig(f"{plot_dir}/token_counts.png")
    plt.close()

    ################################
    # plot token_counts for each subject and save to file

    print("Plotting token_counts for each subject")

    token_counts_persubj = token_model.vocab['token_counts_persubj']

    for SESS_ID in range(len(data_files)):
        # save to file
        np.save(f"{plot_dir}/token_counts_{SESS_ID}.npy", token_counts_persubj[SESS_ID])

    # plot distribution of token_counts for each subject on same plot
    plt.figure(figsize=(10, 10))
    for SESS_ID in range(len(data_files)):
        plt.plot(token_counts_persubj[SESS_ID])
    
    plt.savefig(f"{plot_dir}/token_counts_persubj.png")
    plt.close()

    ################################
    # plot tokens

    tokens, input = token_model.get_vocab(input="impulse")
    num_rows = 5
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    for jj in range(num_rows):
        for ii in range(num_cols):
            vv = jj*num_cols + ii
            axs[jj, ii].plot(tokens[vv])
            axs[jj, ii].plot(input, 'r')
            axs[jj, ii].set_ylim([-1.1, 1.1]) 

    plt.savefig(f"{plot_dir}/tokens.png")
    plt.close()

    ################################

    print(f"Plots saved to {plot_dir}")
    print('')

    if USING_BMRC:
        # get filename from tokenize_dir
        data_name = tokenize_dir.split('/')[-1]

        local_tokenize_dir = f'/Users/woolrich/dev/results_bmrc/osl-tokenize/{data_name}'

        print('Run something like this on local machine to copy raw data:')
        print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{raw_data_dir} /Users/woolrich/dev/results_bmrc/raw_data/')
        print('')

        print('Run something like this on local machine to copy tokenize model:')
        print(f'mkdir {local_tokenize_dir}')
        print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{tokenize_dir}/token_model {local_tokenize_dir}/')
        print('')

        print('Run something like this on local machine to copy tokenized data:')
        print(f'mkdir {local_tokenize_dir}')
        print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{tokenize_dir}/tokenized_data {local_tokenize_dir}/')
        print('')

        print('Run something like this on local machine to copy plots:')
        print(f'mkdir {local_tokenize_dir}')
        print(f'rsync -Phr vxw496@cluster1.bmrc.ox.ac.uk:{tokenize_dir}/plots {local_tokenize_dir}/')
        print('')

    return plot_dir

################################################################################################
# Run

if __name__ == '__main__':
    
    import argparse

    def get_args():
        parser = argparse.ArgumentParser(description='Plot results of running tokenizer on real data (e.g. Camcan), or on simulated data')
        parser.add_argument('tokenize_dir', type=str, help='directory where tokenizer results are')
        parser.add_argument('raw_data_dir', type=str, help='directory where raw data is stored')
        parser.add_argument('--using_bmrc', type=str, help='using bmrc cluster', default='False')
        parser.add_argument('--sess_id', type=int, help='index of session of data to plot', default=0)
        parser.add_argument('--num_sess', type=int, help='number of sessions to calculate stats for', default=None)
        parser.add_argument('--dev_dir', type=str, help='directory where dev code is stored', default=None)

        args = parser.parse_args()

        args.using_bmrc = args.using_bmrc == 'True'

        return vars(args)

    args = get_args()

    # print args
    print(args)

    tokenize_dir = args['tokenize_dir']
    raw_data_dir = args['raw_data_dir']
    USING_BMRC = args['using_bmrc']
    SESS_ID = args['sess_id']
    NUM_SESS = args['num_sess']
    dev_dir = args['dev_dir']

    plot_tokenizer(tokenize_dir, 
                   raw_data_dir,
                   USING_BMRC=USING_BMRC, 
                   SESS_ID=SESS_ID, NUM_SESS=NUM_SESS, 
                   dev_dir=dev_dir)
