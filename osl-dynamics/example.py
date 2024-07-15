"""Example script for running HMM inference on simulated HMM-MVN data.

This script should take less than a couple minutes to run and
achieve a dice coefficient of ~0.99.
"""


import os
import pickle
import numpy as np

from osl_dynamics.simulation import HMM_MVN
from osl_dynamics.data import Data
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.inference import modes, metrics
from osl_dynamics.utils import plotting


def runningHMM(ts, sim_stc, sim_covs):

    # Create directory for results
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create Data object for training
    data = Data(ts)

    # Prepare data
    data.standardize()

    config = Config(
        n_states=5,
        n_channels=3,
        sequence_length=200,
        learn_means=False,
        learn_covariances=True,
        batch_size=16,
        learning_rate=0.01,
        n_epochs=20,
    )

    model = Model(config)
    model.summary()

    # Initialization
    init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)

    # Full training
    history = model.fit(data)

    # Save model
    model_dir = f"{results_dir}/model"
    model.save(model_dir)

    # Calculate the free energy
    free_energy = model.free_energy(data)
    history["free_energy"] = free_energy

    # Save training history and free energy
    pickle.dump(init_history, open(f"{model_dir}/init_history.pkl", "wb"))
    pickle.dump(history, open(f"{model_dir}/history.pkl", "wb"))

    # Inferred state probabilities
    alp = model.get_alpha(data)

    # Observation model parameters
    means, covs = model.get_means_covariances()

    # Save
    inf_params_dir = f"{results_dir}/inf_params"
    os.makedirs(inf_params_dir, exist_ok=True)

    pickle.dump(alp, open(f"{inf_params_dir}/alp.pkl", "wb"))
    np.save(f"{inf_params_dir}/means.npy", means)
    np.save(f"{inf_params_dir}/covs.npy", covs)

    #Calculate summary statistics
    # Viterbi path
    stc = modes.argmax_time_courses(alp)

    # Calculate summary statistics
    fo = modes.fractional_occupancies(stc)
    lt = modes.mean_lifetimes(stc)
    intv = modes.mean_intervals(stc)
    sr = modes.switching_rates(stc)

    # Save
    summary_stats_dir = f"{results_dir}/summary_stats"
    os.makedirs(summary_stats_dir, exist_ok=True)

    np.save(f"{summary_stats_dir}/fo.npy", fo)
    np.save(f"{summary_stats_dir}/lt.npy", lt)
    np.save(f"{summary_stats_dir}/intv.npy", intv)
    np.save(f"{summary_stats_dir}/sr.npy", sr)

    # Compare inferred parameters to ground truth simulation

    # Re-order simulated state time courses to match inferred
    inf_stc, sim_stc = modes.match_modes(stc, sim_stc)
    
    # Calculate dice coefficient
    dice = metrics.dice_coefficient(inf_stc, sim_stc)

    print("Dice coefficient:", dice)

    # Delete temporary directory

    # Compare the covariances
    plotting.plot_matrices(sim_covs, main_title="Ground Truth")
    plotting.show()
    plotting.plot_matrices(covs, main_title="Inferred")
    plotting.show()

    plotting.plot_alpha(
        sim_stc,
        covs,
        n_samples=2000,
        y_labels=["Ground Truth", "Inferred"],
    )
    plotting.show()


    data.delete_dir()