import numpy as np
import matplotlib.pyplot as plt
import os
from osl_tokenize.simulation import hmm

def _generate(NTPTS, CHAN_ACTIVITY, TRUE_FREQS, FS=100, random_seed=222):

    '''
    Generate data with bursts of oscillatory activity

    Parameters
    ----------
    NTPTS : int
        Number of timepoints per chan per subject
    CHAN_ACTIVITY : array, shape (NUM_MODES, NUM_CHANS)
        Indicates which channels are active for each freq mode
    TRUE_FREQS : array, shape (NUM_SUBJ, NUM_MODES)
        True frequencies for each subject and freq mode
    FS : int
        Sampling frequency
    random_seed : int
        Random seed

    Returns
    -------

    true_signal : array, shape (NUM_SUBJ, NTPTS, NUM_CHANS)
        True signal
    mode_tcs : list of arrays, length NUM_MODES
        Mode timecourses

    '''

    # Each mode has a unique bursting frequency

    # TRUE_FREQS is NUM_SUBJ X NUM_MODES
    # CHAN_ACTIVITY is NUM_CHANS x NUM_MODE
    # containing 1's and 0's indicating which modes a channel is active for

    NUM_CHANS = CHAN_ACTIVITY.shape[1]
    NUM_MODES = TRUE_FREQS.shape[1]
    NUM_SUBJ = TRUE_FREQS.shape[0]

    ts = np.arange(0, NTPTS) / FS

    hmm_tmp = hmm.HMM(
        trans_prob="uniform",
        stay_prob=0.98,
        n_states=NUM_MODES+1,
        random_seed=random_seed,
    )

    # mode timecourses and activity timecourses 
    mode_tcs = []
    sinusoidal_activity = np.zeros([NUM_MODES, NUM_SUBJ, NUM_CHANS, NTPTS])
    phase_diff = np.linspace(0, 0.25, NUM_CHANS)
    for kk in range(NUM_MODES):
        mode_tcs.append(hmm_tmp.generate_states(NTPTS)[:, 0])
        # Each mode has a unique oscillatory activity
        for cc in range(NUM_CHANS):
            # Each chan has a unique phase
            #phase = (0.5*np.sin(2 * np.pi * 0.005 * ts)+np.random.uniform())*2 * np.pi 
            phase = (0.5*np.sin(2 * np.pi * 0.005 * ts)+phase_diff[cc])*2 * np.pi 

            for ss in range(NUM_SUBJ):        
                sinusoidal_activity[kk,ss,cc,:] = np.sin(2 * np.pi * TRUE_FREQS[ss,kk] * ts + phase)
        
    # build signal
    true_signal = np.zeros([NUM_SUBJ, NTPTS, NUM_CHANS])
    for ss in range(NUM_SUBJ):
        for kk in range(NUM_MODES):
            for cc in range(NUM_CHANS):
                if CHAN_ACTIVITY[kk, cc] == 1:
                    true_signal[ss, mode_tcs[kk] == 1, cc] += sinusoidal_activity[kk, ss, cc, mode_tcs[kk] == 1]

    return true_signal, mode_tcs

def simulate(data_dir, 
             NUM_GROUPS=1, 
             NUM_SUBJECTS_PERGROUP=1, 
             use_pre_simulated_data=False,
             NTPTS = 1000, 
             CHANS_PER_MODE = 8, 
             NUM_MODES = 4, 
             TRUE_FREQS = None,
             CHAN_ACTIVITY = None,
             FS = 100, SNR=4, 
             random_seed=222):

    '''
    Simulate data with bursts of oscillatory activity
    
    Parameters
    ----------
    data_dir : str
        Directory to save data
    NUM_GROUPS : int
        Number of groups
    NUM_SUBJECTS_PERGROUP : int
        Number of subjects per group
    use_pre_simulated_data : bool
        Use pre-simulated data
    NTPTS : int
        Number of timepoints        
    CHANS_PER_MODE : int
        Number of channels per freq mode, or if NUM_MODE==0 then is number of channels
    NUM_MODE : int
        Number of freq modes
    TRUE_FREQS : array, shape (NUM_MODES, 1)
        True frequencies for each mode. 
        The NUM_MODES arg is ignored
    CHAN_ACTIVITY : array, shape (NUM_MODES, NUM_CHANS)
        Indicates which channels are active for each freq mode
        The CHANS_PER_MODE arg is ignored
    FS : int
        Sampling frequency
    SNR : float
        Signal to noise ratio
    random_seed : int
        Random seed

    Returns
    -------
    data_files : list of str
        List of data files

    '''
        
    if not use_pre_simulated_data:
        print("Simulating data") 

        NUM_SUBJECTS = NUM_GROUPS*NUM_SUBJECTS_PERGROUP

        # file names to store simulated data
        data_files=[]
        for ss in range(NUM_SUBJECTS):
            data_files.append(f"{data_dir}/x_{ss}.npy")

        # Make data_dir directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        ground_truth_dir = data_dir + "/ground_truth"
        if not os.path.exists(ground_truth_dir):
            os.makedirs(ground_truth_dir)

        if NUM_MODES > 0:
            if TRUE_FREQS is None:
                TRUE_FREQS = np.linspace(8, 25, NUM_MODES) # Hz

            # tile TRUE_FREQS for all subjects
            TRUE_FREQS = np.tile(TRUE_FREQS, [NUM_SUBJECTS, 1])

            group_freq_shift = 0.5 # Hz

            for gg in range(NUM_GROUPS):
                for ss in range(NUM_SUBJECTS_PERGROUP):
                    TRUE_FREQS[gg*NUM_SUBJECTS_PERGROUP + ss, :] += (gg-(NUM_GROUPS+1)/2.0)*group_freq_shift

            np.save(ground_truth_dir + "/true_freqs.npy", TRUE_FREQS)

            if CHAN_ACTIVITY is None:
                CHAN_ACTIVITY = np.tile(np.eye(NUM_MODES), [CHANS_PER_MODE]) # NUM_MODES x NUM_CHANS

            np.save(ground_truth_dir + "/chan_activity.npy", CHAN_ACTIVITY)

            true_signal, mode_tcs = _generate(NTPTS, CHAN_ACTIVITY, TRUE_FREQS, FS=FS, random_seed=random_seed)

        else:
            TRUE_FREQS = None
            CHAN_ACTIVITY = None
            true_signal = np.zeros([NUM_SUBJECTS, NTPTS, CHANS_PER_MODE])
            mode_tcs = []
            mode_tcs.append(np.zeros([NTPTS]))

        # add measurement noise 
        noise_std = 1/SNR       
        data = true_signal + np.random.normal(0, noise_std, size=true_signal.shape)

        np.save(ground_truth_dir + "/mode_tcs.npy", mode_tcs)
        for ss in range(NUM_SUBJECTS):
            np.save(data_files[ss], data[ss, :, :])
            np.save(ground_truth_dir + f"/true_signal_{ss}.npy", true_signal[ss, :, :])

    else:
        print("Using pre-simulated data")
        
        # find all files in data_dir
        data_files = []
        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                data_files.append(os.path.join(data_dir, file))

    return data_files

def plot_data(raw_data_dir, plot_dir, NUM_SUBJECTS, FS):

    '''
    Plot simulated data
    
    Parameters
    ----------
    raw_data_dir : str
        Directory where raw data is saved
    plot_dir : str
        Directory to save plots
    NUM_SUBJECTS : int
        Number of subjects
    FS : int
        Sampling frequency in Hz
    
    Returns
    -------
    None
    '''

    data_files=[]
    for ss in range(NUM_SUBJECTS):
        data_files.append(f"{raw_data_dir}/x_{ss}.npy")

    ground_truth_dir = raw_data_dir + "/ground_truth"

    mode_tcs = np.load(ground_truth_dir + "/mode_tcs.npy")
    NUM_MODES = mode_tcs.shape[0] 

    # plot mode tcs as subplots
    # num is mimimum of 2000 timepoints or length of mode_tcs
    num = min(2000, mode_tcs[0].shape[0])
    
    ts = np.arange(num)/FS
    fig, axs = plt.subplots(NUM_MODES, 1, figsize=(10, 5))
    for jj in range(NUM_MODES):
        axs[jj].plot(ts, mode_tcs[jj][:num])
        axs[jj].set_ylim([-0.1, 1.1])
        # remove xticks for all but bottom plot
        if jj < NUM_MODES-1:
            axs[jj].set_xticks([])
        else:
            axs[jj].set_xlabel('Time (s)')
        axs[jj].set_yticks([0, 1])

    plt.savefig(f"{plot_dir}/mode_tcs.png")
    
    sub1 = 0
    sub2 = NUM_SUBJECTS-1

    # plot data
    num = 400
    ts = np.arange(num)/FS
    data_s0 = np.load(data_files[sub1])
    data_s1 = np.load(data_files[sub2])
    chans2plot = [0]
    for chan in chans2plot:
        plt.figure(figsize=(15, 5))
        plt.plot(ts, data_s0[:num, chan], 'r')
        plt.plot(ts, data_s1[:num, chan], 'g')
        # add legend
        plt.legend([f'Sub{sub1}', f'Sub{sub2}'])
        plt.xlabel('Time (s)')
        # No yticks
        plt.yticks([])
        plt.savefig(f"{plot_dir}/data_chan{chan}.png")

    # plot PSDs using Welch's method
    data_s0 = np.load(data_files[sub1])
    data_s1 = np.load(data_files[sub2])
    #chans2plot = [0, 25, 50]
    chans2plot = [0]
    for chan in chans2plot:
        plt.figure(figsize=(15, 5))
        f, Pxx = plt.psd(data_s0[:, chan], Fs=FS, NFFT=1024, color='r')
        f, Pxx = plt.psd(data_s1[:, chan], Fs=FS, NFFT=1024, color='g')
        # add legend
        plt.legend([f'Sub{sub1}', f'Sub{sub2}'])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        plt.savefig(f"{plot_dir}/psd_chan{chan}.png")

    # plot true signals
    for chan in chans2plot:

        plt.figure(figsize=(15, 5))
        true_signal = np.load((ground_truth_dir + f"/true_signal_{sub1}.npy"))
        plt.plot(true_signal[:400, chan], 'r')
        true_signal = np.load((ground_truth_dir + f"/true_signal_{sub2}.npy"))
        plt.plot(true_signal[:400, chan], 'g')
        # title with channel number
        plt.title(f"Channel {chan}")
        # add legend
        plt.legend(['Sub1', 'Sub2'])
        plt.savefig(f"{plot_dir}/true_signal_chan{chan}.png")

    # plot channel activity
    CHAN_ACTIVITY = np.load(ground_truth_dir + "/chan_activity.npy")

    plt.figure()
    plt.imshow(CHAN_ACTIVITY)
    # axis labels
    plt.xlabel('Channel')
    plt.ylabel('Mode')
    # integer yticks
    plt.yticks(np.arange(CHAN_ACTIVITY.shape[0]))
    plt.savefig(f"{plot_dir}/CHAN_ACTIVITY.png")

    # plot true freqs
    TRUE_FREQS = np.load(ground_truth_dir + "/true_freqs.npy")

    plt.figure()
    plt.imshow(TRUE_FREQS.T)
    # axis labels    
    plt.xlabel('Subjects')
    plt.ylabel('Freq Mode')
    # integer yticks
    plt.yticks(np.arange(TRUE_FREQS.shape[1]))
    
    plt.colorbar()
    plt.savefig(f"{plot_dir}/TRUE_FREQS.png")

    print("Plots saved to {}".format(plot_dir))


