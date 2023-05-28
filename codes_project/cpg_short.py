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
from alive_progress import alive_bar
import help_hybrid as hlph

import PIL.Image


# Initialize simulation
simulation = hlph.Simulation(run_time=1, friction=1)

#Initialize model mujoco
nmf = simulation.init_model()
simulation.compute_oscilators()
# Make the Solver
simulation.compute_solver()
solver, phases, amplitudes, joint_angles, obs_list_tripod, num_steps = simulation.set_solver_simu()


with alive_bar(num_steps) as bar:
    for i in range(num_steps):

        res = solver.integrate(nmf.curr_time)
        phase = res[:simulation.n_oscillators]
        amp = res[simulation.n_oscillators:2 * simulation.n_oscillators]

        phases[i, :] = phase
        amplitudes[i, :] = amp

        if i > simulation.n_stabilisation_steps:
            indices = simulation.advancement_transfer(phase)
            # scale the amplitude of the joint angles to the output amplitude (High values of amplitude will highly alter the steps)
            # With an amplitude of one, the joint angles will be the same as the one from the base step
            # With an amplitude of zero, the joint angles will be the same as the zero inidices of the base step (default pose)
            # The rest is a linear interpolation between those two
            action = {'joints': simulation.step_data_block_manualcorrect[simulation.joint_ids, 0] + \
                                (simulation.step_data_block_manualcorrect[simulation.joint_ids, indices] - simulation.step_data_block_manualcorrect[
                                    simulation.joint_ids, 0]) * amp[simulation.match_leg_to_joints]}
            # action = {'joints': step_data_block_base[joint_ids, indices]}
        else:
            action = {'joints': simulation.step_data_block_manualcorrect[simulation.joint_ids, 0]}

        joint_angles[i, :] = action['joints']

        print(action)

        obs, info = nmf.step(action)
        obs_list_tripod.append(obs)
        nmf.render()
        bar()

nmf.save_video(simulation.out_dir / "tripod.mp4")
