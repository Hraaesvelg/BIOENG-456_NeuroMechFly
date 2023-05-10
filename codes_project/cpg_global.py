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
from ipywidgets import Video
from alive_progress import alive_bar
import time
import helpers as hlp

# Initialize simulation
run_time = hlp.run_time
out_dir = Path('../CPGs')
friction = 1.0

physics_config = {
    'joint_stiffness': 2500,
    'friction': (friction, 0.005, 0.0001),
    'gravity': (0, 0, -9.81e5)}
terrain_config = {'fly_pos': (0, 0, 300),
                  'friction': (friction, 0.005, 0.0001)}

nmf = NeuroMechFlyMuJoCo(render_mode='saved',
                         timestep=1e-4,
                         render_config={'playspeed': 0.1, 'camera': 'Animat/camera_left_top'},
                         init_pose='stretch',
                         actuated_joints=all_leg_dofs)

num_steps_base = int(run_time / nmf.timestep)

legs = hlp.legs
n_oscillators = hlp.n_oscillators

dt = nmf.timestep  # seconds
t = np.arange(0, run_time, dt)

# lets say we want 10 oscillations in the time period
n_steps = 10
frequencies = hlp.frequencies

# For now each oscillator have the same amplitude
target_amplitude = 1.0
target_amplitudes = np.ones(n_oscillators) * target_amplitude
rate = 10
rates = np.ones(n_oscillators) * rate

# Here we just build a chain of oscillators form 1 to 6
# They all have a bias of pi/6 from one to the other
bias = np.pi/3
# bias only above or bellow the main diagonal => only coupling between neighbouring oscillators
phase_biases = np.diag(np.ones(n_oscillators-1)*bias,k=1) - np.diag(np.ones(n_oscillators-1) * bias, k=-1)

# We wont play with the coupling weights yet so lets set it as the same as the phase biases
# As a consequence the oscillators that have a phase difference of zero are not coupled (depending on your implementation you might want to change that)
coupling_weights = (np.abs(phase_biases) > 0).astype(float) * 10.0 #* 10.0

def phase_oscillator(_time, state):
    """Phase oscillator model used in Ijspeert et al. 2007"""

    phases = state[:n_oscillators]
    amplitudes = state[n_oscillators:2 * n_oscillators]

    # NxN matrix with the phases of the oscillators
    phase_matrix = np.tile(phases, (n_oscillators, 1))

    # NxN matrix with the amplitudes of the oscillators
    amp_matrix = np.tile(amplitudes, (n_oscillators, 1))

    freq_contribution = 2 * np.pi * frequencies

    #  scaling of the phase differences between oscillators by the amplitude of the oscillators and the coupling weights
    scaling = np.multiply(amp_matrix, coupling_weights)

    # phase matrix and transpose substraction are analogous to the phase differences between oscillators, those should be close to the phase biases
    phase_shifts_contribution = np.sin(phase_matrix - phase_matrix.T - phase_biases)

    # Here we compute the contribution of the phase biases to the derivative of the phases
    # we mulitply two NxN matrices and then sum over the columns (all j oscillators contributions) to get a vector of size N
    coupling_contribution = np.sum(np.multiply(scaling, phase_shifts_contribution), axis=1)

    # Here we compute the derivative of the phases given by the equations defined previously.
    # We are using for that matrix operations to speed up the computation
    dphases = freq_contribution + coupling_contribution

    damplitudes = np.multiply(rates, target_amplitudes - amplitudes)

    return np.concatenate([dphases, damplitudes])

np.random.seed(42)

# Set solver
solver = ode(f=phase_oscillator)
solver.set_integrator('dop853')
initial_values = np.random.rand(2*n_oscillators)
solver.set_initial_value(y=initial_values, t=nmf.curr_time)

# Initialize states and amplitudes
phases = np.zeros((num_steps_base, n_oscillators))
amplitudes = np.zeros((num_steps_base, n_oscillators))
output = np.zeros((num_steps_base, n_oscillators))

obs_list_tripod = []

for i in range(num_steps_base):

    res = solver.integrate(i*nmf.timestep)
    phase = res[:n_oscillators]
    amp = res[n_oscillators:2*n_oscillators]

    phases[i, :] = phase
    amplitudes[i, :] = amp
    output[i, :] = hlp.sine_output(phases[i, :], amplitudes[i, :])

hlp.plot_phase_amp_output(phases, amplitudes, output, labels=["osc_" + str(i) for i in range(n_oscillators)])

# Load recorded data
data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
    data = pickle.load(f)


# Interpolate 5x
step_duration = len(data['joint_LFCoxa'])
interp_step_duration = int(step_duration * data['meta']['timestep'] / nmf.timestep)
step_data_block_base = np.zeros((len(nmf.actuated_joints), interp_step_duration))
measure_t = np.arange(step_duration) * data['meta']['timestep']
interp_t = np.arange(interp_step_duration) * nmf.timestep
for i, joint in enumerate(nmf.actuated_joints):
    step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

num_joints_to_visualize = 7

step_data_block_manualcorrect = step_data_block_base.copy()

for side in ["L", "R"]:
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}MCoxa")] += np.deg2rad(10) # Protract the midlegs
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HFemur")] += np.deg2rad(-5) # Retract the hindlegs
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}FFemur")] += np.deg2rad(15) # Protract the forelegs (slightly to conterbalance Tarsus flexion)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}FTarsus1")] -= np.deg2rad(15) # Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

n_joints = len(nmf.actuated_joints)

leg_ids = np.arange(len(legs)).astype(int)
joint_ids = np.arange(n_joints).astype(int)
# Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
match_leg_to_joints = np.array([i for joint in nmf.actuated_joints for i, leg in enumerate(legs) if leg in joint])

# Now we can try this mapping function to generate joint angles from the phases
joint_angles_1 = []

for phase in phases[-5000:]: # using the phases from the toy example with constant delay
    t_indices = hlp.advancement_transfer(phase, match_leg_to_joints=match_leg_to_joints)
    joint_angles_1.append(step_data_block_manualcorrect[joint_ids, t_indices])


# The bias matrix is define as follow: each line is the i oscillator and each column is the j oscillator couplign goes from i to j
# We express the bias in percentage of cycle
phase_biases_measured = np.array([[0, 0.425, 0.85, 0.51, 0, 0],
                                  [0.425, 0, 0.425, 0, 0.51, 0],
                                  [0.85, 0.425, 0, 0, 0, 0.51],
                                  [0.51, 0, 0, 0, 0.425, 0.85],
                                  [0, 0.51, 0, 0.425, 0, 0.425],
                                  [0, 0, 0.51, 0.85, 0.425, 0]])

phase_biases_idealized = np.array([[0, 0.5, 1.0, 0.5, 0, 0],
                                   [0.5, 0, 0.5, 0, 0.5, 0],
                                   [1.0, 0.5, 0, 0, 0, 0.5],
                                   [0.5, 0, 0, 0, 0.5, 1.0],
                                   [0, 0.5, 0, 0.5, 0, 0.5],
                                   [0, 0, 0.5, 1.0, 0.5, 0]])
# Phase bias of one is the same as zero (1 cycle difference)
# If we would use a phase bias of zero, we would need to change the coupling weight strategy

phase_biases = phase_biases_idealized * 2 * np.pi

coupling_weights = (np.abs(phase_biases) > 0).astype(float) * 10.0  # * 10.0

np.random.seed(42)

_ = nmf.reset()

# Set solver
solver = ode(f=phase_oscillator)
solver.set_integrator('dop853')
initial_values = np.random.rand(2 * n_oscillators)
solver.set_initial_value(y=initial_values, t=nmf.curr_time)

n_stabilisation_steps = 1000
num_steps = n_stabilisation_steps + num_steps_base

phases = np.zeros((num_steps, n_oscillators))
amplitudes = np.zeros((num_steps, n_oscillators))

joint_angles = np.zeros((num_steps, n_joints))

obs_list_tripod = []

with alive_bar(num_steps) as bar:
    for i in range(num_steps):
        res = solver.integrate(nmf.curr_time)
        phase = res[:n_oscillators]
        amp = res[n_oscillators:2 * n_oscillators]

        phases[i, :] = phase
        amplitudes[i, :] = amp

        if i > n_stabilisation_steps:
            indices = hlp.advancement_transfer(phase, match_leg_to_joints=match_leg_to_joints)
            # scale the amplitude of the joint angles to the output amplitude (High values of amplitude will highly alter the steps)
            # With an amplitude of one, the joint angles will be the same as the one from the base step
            # With an amplitude of zero, the joint angles will be the same as the zero inidices of the base step (default pose)
            # The rest is a linear interpolation between those two
            action = {'joints': step_data_block_manualcorrect[joint_ids, 0] + \
                                (step_data_block_manualcorrect[joint_ids, indices] - step_data_block_manualcorrect[
                                    joint_ids, 0]) * amp[match_leg_to_joints]}
            # action = {'joints': step_data_block_base[joint_ids, indices]}
        else:
            action = {'joints': step_data_block_manualcorrect[joint_ids, 0]}

        joint_angles[i, :] = action['joints']

        obs, info = nmf.step(action)
        obs_list_tripod.append(obs)
        nmf.render()
        bar()

name = input('Enter the name of the video')
name = name +'.mp4'

nmf.save_video(out_dir/name)
print('Video saved')

hlp.show_cpg_result(nmf,step_data_block_base, num_joints_to_visualize, joint_angles_1
                    , phases, n_oscillators, amplitudes, joint_angles)



