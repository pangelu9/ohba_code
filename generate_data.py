import numpy as np
import matplotlib.pyplot as plt

from osl_dynamics import simulation
from osl_dynamics.utils import plotting
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config
from osl_dynamics.models.hmm import Model
from osl_dynamics.inference import modes
from osl_dynamics.inference import metrics


def generate_hmm_signal(args, n_samples, n_states, sampling_rate, n_components, 
             noise_std_range=(0.05, 0.2), plotting_fig=False, freq_given=None, ampl_given=None):
    print("Simulate data")
    t = np.linspace(0, n_samples / sampling_rate, n_samples)

    # Initiate the HMM_MVN class simulates HMM data
    sim = simulation.HMM_MVN(
        n_samples=n_samples,
        n_states=n_states,
        n_channels=args.no_channels,
        trans_prob="sequence",
        stay_prob=0.9,  # diagonal of the transition probability matrix
        means="zero",
        covariances="random",  # will automatically generate some covariances 'randomly'
        #random_seed=123,
    )

    # standardize (z-transform) the data.
    #sim.standardize()

    # We can access the simulated data using the time series attribute
    ts = sim.time_series
    #print(ts)
    
    sim_tp = sim.trans_prob # Simulated transition probability matrix
    sim_stc = sim.state_time_course # Simulated state time course
    sim_covs = sim.covariances # Simulated covariances
    
    states = np.argmax(sim_stc, axis=1) # sim_stc is one-hot encoded, here converted to states 1,2,3 ...

    if plotting_fig:
        
        plotting.plot_matrices(sim_tp)
        plotting.show()
        
        plotting.plot_alpha(sim_stc)
        plotting.show()

        plotting.plot_alpha(sim_stc, n_samples=2000)
        plotting.show()

        #plotting.plot_matrices(sim_covs, titles=[f"State {i}" for i in range(1, 6)])
        #plotting.show()

    signal = np.zeros(n_samples)
    for state in range(n_states):
        noise_std = 0 #np.random.uniform(noise_std_range[0], noise_std_range[1])
        freq_given_state = freq_given[state] # take the freq. that corresponds to this state, freq_given is of size [n_states]
        #ampl_given_state = np.random.uniform(0, 1) #ampl_given[state]
        _, signal_with_noise, _ = generate_signal_with_noise(n_samples, sampling_rate, n_components, noise_std=noise_std, freq_given=freq_given_state)#, ampl_given=ampl_given_state)
        state_signal = signal_with_noise
        signal[states == state] += state_signal[states == state]
    
   

    return t, signal, states, sim_stc, sim_covs, sim_tp, ts


def generate_signal_with_noise(n_samples, sampling_rate, n_components, 
                               amplitude_range=(0.2, 1.0), phase_range=(0, 2*np.pi), freq_range=(1, 50), 
                               noise_std=0.1, freq_given=None, ampl_given=None):
    """
    Generate a synthetic signal as a sum of sine waves with random amplitudes and phases, and add white noise.
    
    Parameters:
    - n_samples: int, number of time points
    - sampling_rate: int, sampling rate in Hz
    - n_components: int, number of sine wave components
    - amplitude_range: tuple, range (min, max) for random amplitude sampling
    - phase_range: tuple, range (min, max) for random phase sampling
    - freq_range: tuple, range (min, max) for random frequency sampling
    - noise_std: float, standard deviation of the white noise to be added
    
    Returns:
    - t: numpy array, time vector
    - signal: numpy array, generated signal with added noise
    - pure_signal: numpy array, generated signal without noise
    """
    t = np.linspace(0, n_samples / sampling_rate, n_samples)
    
    # Sample random frequencies, amplitudes, and phases
    if freq_given is not None: # if frequencies are given as arguments
        frequencies = freq_given
    else:
        frequencies = np.random.uniform(freq_range[0], freq_range[1], n_components) #if frequencies are not given as arguments, they are sampled.
    if ampl_given is not None: # if frequencies are given as arguments
        amplitudes = ampl_given
    else:
        amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], n_components)
    
    phases = np.random.uniform(phase_range[0], phase_range[1], n_components)
    
    # Generate the signal as a sum of sine waves
    pure_signal = np.zeros(n_samples)
    if n_components>1:
        for freq, amp, phase in zip(frequencies, amplitudes, phases):
            pure_signal += amp * np.sin(2 * np.pi * freq * t + phase)
    else:
        pure_signal = amplitudes * np.sin(2 * np.pi * frequencies * t + phases)

    # Generate and add white noise
    noise = np.random.normal(0, noise_std, n_samples)
    signal = pure_signal + noise
    
    return t, signal, pure_signal




def generate_givenPDS(power_spectrum):
    # Parameters
    n_points = 1024  # Number of points in the signal
    dt = 0.01       # Time step (sampling interval)
    freq = np.fft.fftfreq(n_points, dt)  # Frequency bins

    # Define the power spectrum (example: 1/f noise)
    positive_freq_indices = np.where(freq > 0)
    freq_pos = freq[positive_freq_indices]
    power_spectrum = 1 / (freq_pos**2 + 1)  # Avoid division by zero at freq=0

    # Generate random phases
    random_phases = np.exp(1j * np.random.uniform(0, 2*np.pi, size=power_spectrum.shape))

    # Combine amplitude and phase
    signal_freq_half = np.sqrt(power_spectrum) * random_phases

    # Construct the full spectrum (ensure conjugate symmetry)
    signal_freq = np.zeros(n_points, dtype=complex)
    signal_freq[positive_freq_indices] = signal_freq_half
    signal_freq[-positive_freq_indices[0]] = np.conj(signal_freq_half[::-1])

    # Inverse Fourier Transform to get the time domain signal
    signal_time = np.fft.ifft(signal_freq).real

    # Plot the generated signal
    plt.figure(figsize=(10, 4))
    plt.plot(np.real(signal_time))
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Generated Random Signal with Desired Power Spectrum')
    plt.grid(True)
    plt.show()

    # Plot the power spectrum of the generated signal
    plt.figure(figsize=(10, 4))
    plt.loglog(freq_pos, np.abs(signal_freq[positive_freq_indices])**2)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.title('Power Spectrum of the Generated Signal')
    plt.grid(True)
    plt.show()

def validate(args, signal, sim_stc, sim_covs, sim_tp):

    training_data = Data(signal)
    # Perform time-delay embedding
    if args.tde:
        if args.pca:
            methods = {
                #"tde": {"n_embeddings": args.num_tde},
                "tde_pca": {"n_embeddings": args.num_tde, "n_pca_components": args.n_pca_components},
                "standardize": {},
            }
            n_ch = args.n_pca_components
        else:
            methods = {
                "tde": {"n_embeddings": args.num_tde},
                "standardize": {},
            }
            n_ch = args.num_tde

        training_data.prepare(methods)
        print("new training_data shape:", training_data)
        #original_data = training_data.time_series(prepared=False)
        #prepared_data = training_data.time_series()
        #for i in range(training_data.n_sessions):
        #    print(original_data[i].shape, prepared_data[i].shape)
        config = Config(
            n_states=8,
            n_channels=n_ch,
            sequence_length=1000,
            learn_means=False,
            learn_covariances=True,
            batch_size=32,
            learning_rate=1e-3,
            n_epochs=args.hmm_epochs, 
        )
    else:
        
        config = Config(
        n_states=3,
        n_channels=args.no_channels,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=10, 
    )
        
    model = Model(config)
    model.summary()
    init_history = model.random_state_time_course_initialization(training_data, n_init=3, n_epochs=1)

    history = model.fit(training_data)

    plotting.plot_line(
    [range(1, len(history["loss"]) + 1)],
    [history["loss"]],
    x_label="Epoch",
    y_label="Loss",
)
    # Get the inferred state probabilities for the training data
    alpha = model.get_alpha(training_data)
    print(alpha.shape)
    print(signal.shape)


    # Take the most probably state to get a state time course
    inf_stc = modes.argmax_time_courses(alpha)

    # Printing the inf_stc array, we see it is binary.
    # The column with a value of 1 indicates which state is active
    print(inf_stc)
    inf_covs = model.get_covariances()
    inf_tp = model.get_trans_prob()

    # Get the order that matches the states
    _, order = modes.match_modes(sim_stc, inf_stc, return_order=True)
    print("order", order)
    print("shapes", sim_stc.shape, inf_stc.shape)
    # Re-order the inferred parameters
    inf_stc_ord = inf_stc[:, order]
    inf_covs_ord = inf_covs[order]
    inf_tp_ord = inf_tp[np.ix_(order, order)]

    # Compare the state time courses
    plotting.plot_alpha(
        sim_stc,
        inf_stc_ord,
        n_samples=2000,
        y_labels=["Ground Truth", "Inferred"],
    )
    plotting.show()
    if not args.tde:
        print("Dice coefficient:", metrics.dice_coefficient(sim_stc, inf_stc_ord))
    else:
        print(sim_stc.shape, inf_stc_ord.shape)
        
        subtract = args.num_tde // 2

        print("subtract", subtract)
        print("Dice coefficient:", metrics.dice_coefficient(sim_stc[subtract:-subtract,:], inf_stc_ord))
        
    # Compare the covariances
    plotting.plot_matrices(sim_covs, main_title="Ground Truth")
    plotting.show()
    plotting.plot_matrices(inf_covs_ord, main_title="Inferred")
    plotting.show()

    # Compare the transition probabilty matrices
    plotting.plot_matrices([sim_tp, inf_tp_ord], titles=["Ground Truth", "Inferred"])
    plotting.show()







