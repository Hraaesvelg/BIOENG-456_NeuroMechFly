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


"""
body = ['Thorax', 'A1A2', 'A3', 'A4', 'A5', 'A6', 'Head_roll', 'Head_yaw',
 'Head', 'LEye', 'LPedicel_roll', 'LPedicel_yaw', 'LPedicel',
 'LFuniculus_roll', 'LFuniculus_yaw', 'LFuniculus', 'LArista_roll',
 'LArista_yaw', 'LArista', 'REye', 'Rostrum', 'Haustellum',
 'RPedicel_roll', 'RPedicel_yaw', 'RPedicel', 'RFuniculus_roll',
 'RFuniculus_yaw', 'RFuniculus', 'RArista_roll', 'RArista_yaw',
 'RArista', 'LFCoxa_roll', 'LFCoxa_yaw', 'LFCoxa', 'LFFemur',
 'LFFemur_roll', 'LFTibia', 'LFTarsus1', 'LFTarsus2', 'LFTarsus3',
 'LFTarsus4', 'LFTarsus5', 'LHaltere_roll', 'LHaltere_yaw',
 'LHaltere', 'LHCoxa_roll', 'LHCoxa_yaw', 'LHCoxa', 'LHFemur',
 'LHFemur_roll', 'LHTibia', 'LHTarsus1', 'LHTarsus2', 'LHTarsus3',
 'LHTarsus4', 'LHTarsus5', 'LMCoxa_roll', 'LMCoxa_yaw', 'LMCoxa',
 'LMFemur', 'LMFemur_roll', 'LMTibia', 'LMTarsus1', 'LMTarsus2',
 'LMTarsus3', 'LMTarsus4', 'LMTarsus5', 'LWing_roll', 'LWing_yaw',
 'LWing', 'RFCoxa_roll', 'RFCoxa_yaw', 'RFCoxa', 'RFFemur',
 'RFFemur_roll', 'RFTibia', 'RFTarsus1', 'RFTarsus2', 'RFTarsus3',
 'RFTarsus4', 'RFTarsus5', 'RHaltere_roll', 'RHaltere_yaw',
 'RHaltere', 'RHCoxa_roll', 'RHCoxa_yaw', 'RHCoxa', 'RHFemur',
 'RHFemur_roll', 'RHTibia', 'RHTarsus1', 'RHTarsus2', 'RHTarsus3',
 'RHTarsus4', 'RHTarsus5', 'RMCoxa_roll', 'RMCoxa_yaw', 'RMCoxa',
 'RMFemur', 'RMFemur_roll', 'RMTibia', 'RMTarsus1', 'RMTarsus2',
 'RMTarsus3', 'RMTarsus4', 'RMTarsus5', 'RWing_roll', 'RWing_yaw',
 'RWing']

joints = ['joint_Head_roll', 'joint_Head_yaw', 'joint_Head',
'joint_LPedicel_roll', 'joint_LPedicel_yaw', 'joint_LPedicel',
'joint_LFuniculus_roll', 'joint_LFuniculus_yaw',
'joint_LFuniculus', 'joint_LArista_roll', 'joint_LArista_yaw',
'joint_LArista', 'joint_RPedicel_roll', 'joint_RPedicel_yaw',
'joint_RPedicel', 'joint_RFuniculus_roll', 'joint_RFuniculus_yaw',
'joint_RFuniculus', 'joint_RArista_roll', 'joint_RArista_yaw',
'joint_RArista', 'joint_LFCoxa_roll', 'joint_LFCoxa_yaw',
'joint_LFCoxa', 'joint_LFFemur', 'joint_LFFemur_roll',
'joint_LFTibia', 'joint_LFTarsus1', 'joint_LFTarsus2',
'joint_LFTarsus3', 'joint_LFTarsus4', 'joint_LFTarsus5',
'joint_LHCoxa_roll', 'joint_LHCoxa_yaw', 'joint_LHCoxa',
'joint_LHFemur', 'joint_LHFemur_roll', 'joint_LHTibia',
'joint_LHTarsus1', 'joint_LHTarsus2', 'joint_LHTarsus3',
'joint_LHTarsus4', 'joint_LHTarsus5', 'joint_LMCoxa_roll',
'joint_LMCoxa_yaw', 'joint_LMCoxa', 'joint_LMFemur',
'joint_LMFemur_roll', 'joint_LMTibia', 'joint_LMTarsus1',
'joint_LMTarsus2', 'joint_LMTarsus3', 'joint_LMTarsus4',
'joint_LMTarsus5', 'joint_RFCoxa_roll', 'joint_RFCoxa_yaw',
'joint_RFCoxa', 'joint_RFFemur', 'joint_RFFemur_roll',
'joint_RFTibia', 'joint_RFTarsus1', 'joint_RFTarsus2',
'joint_RFTarsus3', 'joint_RFTarsus4', 'joint_RFTarsus5',
'joint_RHCoxa_roll', 'joint_RHCoxa_yaw', 'joint_RHCoxa',
'joint_RHFemur', 'joint_RHFemur_roll', 'joint_RHTibia',
'joint_RHTarsus1', 'joint_RHTarsus2', 'joint_RHTarsus3',
'joint_RHTarsus4', 'joint_RHTarsus5', 'joint_RMCoxa_roll',
'joint_RMCoxa_yaw', 'joint_RMCoxa', 'joint_RMFemur',
'joint_RMFemur_roll', 'joint_RMTibia', 'joint_RMTarsus1',
'joint_RMTarsus2', 'joint_RMTarsus3', 'joint_RMTarsus4',
'joint_RMTarsus5']

"""