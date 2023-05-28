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


class Simulation():
    def __init__(self,
                 run_time=1,
                 out_dir=Path('../decentralized_ctrl'),
                 friction=1.0
                 ):
        self.joint_ids = None
        self.leg_ids = None
        self.step_data_block_manualcorrect = None
        self.n_stabilisation_steps = None
        self.match_leg_to_joints = None
        self.phase_biases_measured = None
        self.phase_biases_idealized = None
        self.coupling_weights = None
        self.phase_biases = None
        self.rates = None
        self.target_amplitudes = None
        self.frequencies = None
        self.n_oscillators = None
        self.t = None
        self.dt = None

        self.run_time = run_time
        self.out_dir = out_dir
        self.friction = friction
        self.physics_config = {
            'joint_stiffness': 2500,
            'friction': (friction, 0.005, 0.0001),
            'gravity': (0, 0, -9.81e5)}
        self.terrain_config = {'fly_pos': (0, 0, 300),
                               'friction': (friction, 0.005, 0.0001)}
        self.nmf_model = None
        self.num_steps_base = 0

        # Oscilators parameters
        self.legs = ["RF", "RM", "RH", "LF", "LM", "LH"]
        self.n_steps = 10
        self.rate = 10
        self.target_amplitude = 1.0
        self.bias = np.pi / 3

        # Oscilators
        self.dphases = 0
        self.damplitudes = 0

        # Solver
        self.obs_list_tripod = []
        self.interp_step_duration = 0

        print('Simulation Initialized')

    def init_model(self):
        # Initialize the model
        self.nmf_model = NeuroMechFlyMuJoCo(render_mode='saved',
                                            terrain="blocks",
                                            output_dir=self.out_dir,
                                            timestep=1e-4,
                                            render_config={'playspeed': 0.1, 'camera': 'Animat/camera_left_top'},
                                            init_pose='stretch',
                                            actuated_joints=all_leg_dofs,
                                            physics_config=self.physics_config,
                                            terrain_config=self.terrain_config)
        # Compute the number of steps
        self.num_steps_base = int(self.run_time / self.nmf_model.timestep)
        self.dt = self.nmf_model.timestep  # seconds
        self.t = np.arange(0, self.run_time, self.dt)

        # return the model
        print("Model Initialized with ", self.num_steps_base, " steps" )
        return self.nmf_model

    def compute_oscilators(self):
        self.n_oscillators = len(self.legs)
        self.frequencies = np.ones(self.n_oscillators) * self.n_steps / self.run_time

        # For now each oscillator have the same amplitude
        self.target_amplitudes = np.ones(self.n_oscillators) * self.target_amplitude
        self.rates = np.ones(self.n_oscillators) * self.rate
        # Here we just build a chain of oscillators form 1 to 6
        # They all have a bias of pi/6 from one to the other
        # bias only above or bellow the main diagonal => only coupling between neighbouring oscillators
        self.phase_biases = np.diag(np.ones(self.n_oscillators - 1) * self.bias, k=1) - np.diag(np.ones(self.n_oscillators - 1) * self.bias, k=-1)
        # We wont play with the coupling weights yet so lets set it as the same as the phase biases
        # As a consequence the oscillators that have a phase difference of zero are not coupled (depending on your implementation you might want to change that)
        self.coupling_weights = (np.abs(self.phase_biases) > 0).astype(float) * 10.0  # * 10.0

    def phase_oscillator(self, _time, state):
        """Phase oscillator model used in Ijspeert et al. 2007"""
        phases = state[:self.n_oscillators]
        amplitudes = state[self.n_oscillators:2 * self.n_oscillators]

        # NxN matrix with the phases of the oscillators
        phase_matrix = np.tile(phases, (self.n_oscillators, 1))

        # NxN matrix with the amplitudes of the oscillators
        amp_matrix = np.tile(amplitudes, (self.n_oscillators, 1))

        freq_contribution = 2 * np.pi * self.frequencies

        #  scaling of the phase differences between oscillators by the amplitude of the oscillators and the coupling weights
        scaling = np.multiply(amp_matrix, self.coupling_weights)

        # phase matrix and transpose substraction are analogous to the phase differences between oscillators, those should be close to the phase biases
        phase_shifts_contribution = np.sin(phase_matrix - phase_matrix.T - self.phase_biases)

        # Here we compute the contribution of the phase biases to the derivative of the phases
        # we mulitply two NxN matrices and then sum over the columns (all j oscillators contributions) to get a vector of size N
        coupling_contribution = np.sum(np.multiply(scaling, phase_shifts_contribution), axis=1)

        # Here we compute the derivative of the phases given by the equations defined previously.
        # We are using for that matrix operations to speed up the computation
        self.dphases = freq_contribution + coupling_contribution
        self.damplitudes = np.multiply(self.rates, self.target_amplitudes - amplitudes)

        return np.concatenate([self.dphases, self.damplitudes])

    def sine_output(self, phases, amplitudes):
        return amplitudes * (1 + np.cos(phases))


    def compute_solver(self):
        # Set solver
        solver = ode(f=self.phase_oscillator)
        solver.set_integrator('dop853')
        initial_values = np.random.rand(2 * self.n_oscillators)
        solver.set_initial_value(y=initial_values, t=self.nmf_model.curr_time)

        # Initialize states and amplitudes
        phases = np.zeros((self.num_steps_base, self.n_oscillators))
        amplitudes = np.zeros((self.num_steps_base, self.n_oscillators))
        output = np.zeros((self.num_steps_base, self.n_oscillators))

        for i in range(self.num_steps_base):
            res = solver.integrate(i * self.nmf_model.timestep)
            phase = res[:self.n_oscillators]
            amp = res[self.n_oscillators:2 * self.n_oscillators]

            phases[i, :] = phase
            amplitudes[i, :] = amp
            output[i, :] = self.sine_output(phases[i, :], amplitudes[i, :])

        # Load recorded data
        data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
        with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
            data = pickle.load(f)

        # Interpolate 5x
        step_duration = len(data['joint_LFCoxa'])
        self.interp_step_duration = int(step_duration * data['meta']['timestep'] / self.nmf_model.timestep)
        step_data_block_base = np.zeros((len(self.nmf_model.actuated_joints), self.interp_step_duration))
        measure_t = np.arange(step_duration) * data['meta']['timestep']
        interp_t = np.arange(self.interp_step_duration) * self.nmf_model.timestep

        for i, joint in enumerate(self.nmf_model.actuated_joints):
            step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

        num_joints_to_visualize = 7
        plt.plot(np.arange(step_data_block_base.shape[1]) * self.nmf_model.timestep,
                 step_data_block_base[:num_joints_to_visualize].T,
                 label=self.nmf_model.actuated_joints[:num_joints_to_visualize])
        plt.legend(ncol=3)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (radian)')
        plt.show()

        self.step_data_block_manualcorrect = step_data_block_base.copy()

        for side in ["L", "R"]:
            self.step_data_block_manualcorrect[self.nmf_model.actuated_joints.index(f"joint_{side}MCoxa")] += np.deg2rad(
                10)  # Protract the midlegs
            self.step_data_block_manualcorrect[self.nmf_model.actuated_joints.index(f"joint_{side}HFemur")] += np.deg2rad(
                -5)  # Retract the hindlegs
            self.step_data_block_manualcorrect[self.nmf_model.actuated_joints.index(f"joint_{side}HTarsus1")] -= np.deg2rad(
                15)  #  Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
            self.step_data_block_manualcorrect[self.nmf_model.actuated_joints.index(f"joint_{side}FFemur")] += np.deg2rad(
                15)  # Protract the forelegs (slightly to conterbalance Tarsus flexion)
            self.step_data_block_manualcorrect[self.nmf_model.actuated_joints.index(f"joint_{side}FTarsus1")] -= np.deg2rad(
                15)  #  Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

        n_joints = len(self.nmf_model.actuated_joints)

        self.leg_ids = np.arange(len(self.legs)).astype(int)
        self.joint_ids = np.arange(n_joints).astype(int)
        # Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
        self.match_leg_to_joints = np.array(
            [i for joint in self.nmf_model.actuated_joints for i, leg in enumerate(self.legs) if leg in joint])


        # Now we can try this mapping function to generate joint angles from the phases
        joint_angles = []

        for phase in phases[-5000:]:  # using the phases from the toy example with constant delay
            t_indices = self.advancement_transfer(phase, self.interp_step_duration)
            joint_angles.append(self.step_data_block_manualcorrect[self.joint_ids, t_indices])

        joint_angles = np.array(joint_angles)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(joint_angles[:, ::5], label=self.nmf_model.actuated_joints[::5])
        plt.legend(ncol=3)
        plt.show()

        # The bias matrix is define as follow: each line is the i oscillator and each column is the j oscillator couplign goes from i to j
        # We express the bias in percentage of cycle
        self.phase_biases_measured = np.array([[0, 0.425, 0.85, 0.51, 0, 0],
                                          [0.425, 0, 0.425, 0, 0.51, 0],
                                          [0.85, 0.425, 0, 0, 0, 0.51],
                                          [0.51, 0, 0, 0, 0.425, 0.85],
                                          [0, 0.51, 0, 0.425, 0, 0.425],
                                          [0, 0, 0.51, 0.85, 0.425, 0]])

        self.phase_biases_idealized = np.array([[0, 0.5, 1.0, 0.5, 0, 0],
                                           [0.5, 0, 0.5, 0, 0.5, 0],
                                           [1.0, 0.5, 0, 0, 0, 0.5],
                                           [0.5, 0, 0, 0, 0.5, 1.0],
                                           [0, 0.5, 0, 0.5, 0, 0.5],
                                           [0, 0, 0.5, 1.0, 0.5, 0]])
        # Phase bias of one is the same as zero (1 cycle difference)
        # If we would use a phase bias of zero, we would need to change the coupling weight strategy

        phase_biases = self.phase_biases_idealized * 2 * np.pi

        coupling_weights = (np.abs(phase_biases) > 0).astype(float) * 10.0  # * 10.0

        np.random.seed(42)

        _ = self.nmf_model.reset()

    def set_solver_simu(self):
        # Set solver
        n_joints = len(self.nmf_model.actuated_joints)
        solver = ode(f=self.phase_oscillator)
        solver.set_integrator('dop853')
        initial_values = np.random.rand(2 * self.n_oscillators)
        solver.set_initial_value(y=initial_values, t=self.nmf_model.curr_time)

        self.n_stabilisation_steps = 1000
        num_steps = self.n_stabilisation_steps + self.num_steps_base

        phases = np.zeros((num_steps, self.n_oscillators))
        amplitudes = np.zeros((num_steps, self.n_oscillators))

        joint_angles = np.zeros((num_steps, n_joints))

        obs_list_tripod = []
        print("Solver initialized within ", num_steps, 'steps')
        return solver, phases, amplitudes, joint_angles, obs_list_tripod, num_steps

    def advancement_transfer(self, phases, step_dur= 0):
        """From phase define what is the corresponding timepoint in the joint dataset
        In the case of the oscillator, the period is 2pi and the step duration is the period of the step
        We have to match those two"""

        period = 2 * np.pi
        # match length of step to period phases should have a period of period mathc this perios to the one of the step
        t_indices = np.round(np.mod(phases * step_dur / period, step_dur - 1)).astype(int)
        t_indices = t_indices[self.match_leg_to_joints]

        return t_indices

    def show_cpg_values(self, phases, amps, outs, labels=None):
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
