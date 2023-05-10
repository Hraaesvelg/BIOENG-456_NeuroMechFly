import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from tqdm import trange
from flygym.util.config import all_leg_dofs

from scipy.signal import medfilt
from scipy.integrate import ode

import PIL.Image

"""
=============Helper data===================
"""

legs = ["RF", "RM", "RH", "LF", "LM", "LH"]
n_oscillators = len(legs)

n_steps = 10
run_time = 1
frequencies = np.ones(n_oscillators) * n_steps / run_time


"""
=============Helper Functions===================
"""


def plot_phase_amp_output(phases, amps, outs, labels=None):
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))
    axs[0].plot(phases, label=labels)
    axs[0].set_ylabel('Phase')
    axs[1].plot(amps, label=labels)
    axs[1].set_ylabel('Amplitude')
    axs[1].legend(loc="lower right")
    axs[2].plot(outs, label=labels)
    axs[2].set_ylabel('Output')
    axs[2].legend(loc="lower right")

    if labels:
        axs[0].legend(loc="lower right")
        axs[1].legend(loc="lower right")
        axs[2].legend(loc="lower right")

    plt.show()


def sine_output(phases, amplitudes):
    return amplitudes * (1 + np.cos(phases))


def advancement_transfer(phases, match_leg_to_joints, step_dur=7):
    """From phase define what is the corresponding timepoint in the joint dataset
    In the case of the oscillator, the period is 2pi and the step duration is the period of the step
    We have to match those two"""

    period = 2 * np.pi
    # match length of step to period phases should have a period of period mathc this perios to the one of the step
    t_indices = np.round(np.mod(phases * step_dur / period, step_dur - 1)).astype(int)
    t_indices = t_indices[match_leg_to_joints]

    return t_indices

def show_cpg_result(nmf,step_data_block_base, num_joints_to_visualize, joint_angles_1, phases,
                    n_oscillators, amplitudes, joint_angles):
    # Plot 1
    plt.plot(np.arange(step_data_block_base.shape[1]) * nmf.timestep,
             step_data_block_base[:num_joints_to_visualize].T,
             label=nmf.actuated_joints[:num_joints_to_visualize])
    plt.legend(ncol=3)
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (radian)')
    plt.show()

    # Plot 2
    joint_angles_1 = np.array(joint_angles_1)
    fig = plt.figure(figsize=(10, 5))
    plt.plot(joint_angles_1[:, ::5], label=nmf.actuated_joints[::5])
    plt.legend(ncol=3)
    plt.show()

    # Plot 3

    fig, axs = plt.subplots(4, 1, figsize=(10, 10))
    axs[0].plot(phases + np.arange(n_oscillators) * 0.2, label=legs)
    axs[0].set_title("Phases")
    axs[0].legend()
    axs[1].plot(amplitudes)
    axs[1].set_title("Amplitudes")
    axs[2].plot(sine_output(phases, amplitudes), label=legs)
    axs[2].set_title("Shifted sine output")
    axs[3].plot(joint_angles)
    axs[3].set_title("Joint angles")

    plt.tight_layout()