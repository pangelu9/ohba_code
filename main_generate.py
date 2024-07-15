from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import numpy as np

from generate_data import generate_hmm_signal, validate
from analysis import plot_PSD
from example import runningHMM

#sys.path.append('../')


parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
parser.add_argument('--type', type=str, default="hmm", metavar='S',
                    help='generate data from an hmm, each state corresponds to a synthetic signal generatiion: sine + noise')
parser.add_argument('--plotting_fig', action='store_true', default=False,
                    help='plotting figures from osl-dynamics toolbox')
parser.add_argument('--validate', action='store_true', default=False,
                    help='validate my generated data indeed have the properties I have specified')
parser.add_argument('--tde', action='store_true', default=False,
                    help='perform time-delay embeddings')
parser.add_argument('--pca', action='store_true', default=False,
                    help='perform time-delay embeddings')
parser.add_argument('--n_pca_components', type=int, default=1,
                    help='no. of pca components')
parser.add_argument('--no_channels', type=int, default=1,
                    help='no. of channels')
parser.add_argument('--hmm_epochs', type=int, default=1,
                    help='no. of epoch train the HMM')
parser.add_argument('--num_tde', type=int, default=5,
                    help='no. of TDE elements')

args = parser.parse_args()


if __name__ == "__main__":

    # Define parameters
    n_samples = 4*320000 + args.num_tde-1 # Number of time points + no. of tde components.
    sampling_rate = 100  # Hz
    n_components = 1  # Number of sine wave components for each state.
    #noise_std = 0.2  # Standard deviation of the white noise
    n_states = 3 
    frequencies = [10, 20, 30] # one frequency for each state, as I have one component per state.
    #amplitudes = [1.0, 1.0, 1.0]
    if args.type == "hmm":

        t, signal, states, sim_stc, sim_covs, sim_tp, ts = generate_hmm_signal(args, n_samples, n_states, sampling_rate, n_components, plotting_fig = args.plotting_fig, freq_given = frequencies)#, ampl_given=amplitudes)

        loc_dir = "./osl-tokenize/examples/dev/results/raw_data/load_dataset_snr5_cha1_sub1_gro1_mod1/meg_data.npy"
        print("Saving generated data at: ", loc_dir)
        np.save(loc_dir, signal)


        print(states.shape)
        print(signal.shape)
        print(sim_stc.shape, sim_covs.shape, sim_tp.shape)
        plt.figure(figsize=(15, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, signal)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Generated Signal with HMM')

        plt.subplot(2, 1, 2)
        plt.plot(t, states, drawstyle='steps-pre')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.title('Hidden States')

        plt.tight_layout()
        plt.show()

    print("signal shape:", signal.shape)
    print("ts shape:", ts.shape)
    if args.validate:
        plot_PSD(signal, fs=sampling_rate, n=n_samples)
        validate(args, signal, sim_stc, sim_covs, sim_tp)
        #validate(args, ts, sim_stc, sim_covs, sim_tp)
        #runningHMM(ts, sim_stc, sim_covs)

        

