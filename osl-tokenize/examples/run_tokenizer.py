import sys
import os
from os import path as op
import numpy as np
import argparse

from tokenizer import run_tokenizer
from plot_tokenizer import plot_tokenizer

sys.path.append('../')
from osl_tokenize.models import conv as tokenize
sys.path.append('../../')
from analysis import plot_PSD

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Run tokenizer on real data (e.g. Camcan), or simulated data and run tokenizer on that')
parser.add_argument('--dataset_name', type=str, default='camcan', help='dataset name')
parser.add_argument('--load_dataset_dir', type=str, default='', help='dir location for dataset to be laoded')
parser.add_argument('--vocab_size', type=int, default=128, help='VOCAB_SIZE')
parser.add_argument('--sim_snr', type=float, default=5, help='SNR to use for simulated data')
parser.add_argument('--using_bmrc', action='store_true', default=False, help='using bmrc cluster')
parser.add_argument('--do_train', action='store_true', default=False, help='train tokeniser')
parser.add_argument('--do_reconstruct', action='store_true', default=False, help='train tokeniser')
parser.add_argument('--tokenised_dir', type=str, default='', help='dir location for tokenised dataset to be reconstructed')
args = parser.parse_args()   
print(args)

if __name__ == '__main__':

    dataset_name = args.dataset_name
    USING_BMRC = args.using_bmrc
    VOCAB_SIZE = args.vocab_size
    SIM_SNR = args.sim_snr
    
    if USING_BMRC:
        dev_dir = '/well/woolrich/users/vxw496'
    else:
        dev_dir = './dev'

    if args.do_train:
        print("Train tokeniser")
        tokenize_dir, raw_data_dir = run_tokenizer(dataset_name, SIM_SNR=SIM_SNR, USING_BMRC=USING_BMRC, VOCAB_SIZE=VOCAB_SIZE, load_dataset_dir=args.load_dataset_dir)

        osl_tokenize_dir = f'{dev_dir}/projects/osl-tokenize'
        sys.path.append(osl_tokenize_dir)
        plot_tokenizer(tokenize_dir, raw_data_dir, USING_BMRC=USING_BMRC, SESS_ID=0, NUM_SESS=1, dev_dir=dev_dir)

    if args.do_reconstruct:

        #####################################
        # Settings
        untokenize_dir = './dev/results/raw_data/load_dataset_snr5_cha1_sub1_gro1_mod1'
        untokenize_data_dir = f'{untokenize_dir}/meg_data-JUL12_.npy'
        untok_data = np.load(untokenize_data_dir)   
        print("untok_data shape", untok_data.shape)
        print("before tokenisation PSD")
        sampling_rate = 100
        plot_PSD(untok_data, fs=sampling_rate, n = 4*320000 + 4 )#untok_data.shape[0])

        tokenize_dir = args.tokenised_dir #'./dev/results/osl-tokenize/load_dataset_snr5_cha1_sub1_gro1_mod1'
        random_tokens = False

        model_dir = f'{tokenize_dir}/token_model'
        #tokenize_data_dir = f'{tokenize_dir}/tokenized_data/meg_data2_tokenized_data.npy'
        tokenize_data_dir = f'{tokenize_dir}/generated_data/generated_data_recursively.npy'
        
        plot_dir = f"{tokenize_dir}/plots"
        data = np.load(tokenize_data_dir)   
        #data = np.squeeze(data, axis=1)
        print("tok_data shape", data.shape)

        

        print(f"Using pre-trained model {model_dir}")
        token_model = tokenize.Model(model_dir)

        #####################################

        print("Sanity check: Reconstruct data")    
        recon_data = token_model.reconstruct_data(tokenize_data_dir)
        recon_data = np.squeeze(recon_data, axis=1)
        print("recon data shape", recon_data.shape)
        np.save('generated_data_untokenised.npy', recon_data)
        # Sanity check
        #Load initial data
        #original_data = np.load(args.load_dataset_dir)
        #print("Original data shape", original_data.shape)
       

        time_points = np.arange(recon_data.shape[0])
        plt.plot(time_points, recon_data)
        ## plot PSD of recon data
        sampling_rate = 100
        print("data timepoints", recon_data.shape[0])
        plot_PSD(recon_data, fs=sampling_rate, n=recon_data.shape[0])



