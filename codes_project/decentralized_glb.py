import numpy as np
import pkg_resources
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from flygym.envs.nmf_mujoco import NeuroMechFlyMuJoCo
from tqdm import trange
from flygym.util.config import all_leg_dofs

import PIL.Image
from ipywidgets import Video

# Initialize simulation
run_time = 1
out_dir = Path('../decentralized_ctrl')

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


# Load the data --> change the path
# Load recorded data
data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
    data = pickle.load(f)

print('Dict keys:', data.keys())
print('Length of time series:', data['joint_RFCoxa'])
print('Metadata:', data['meta'])

# Interpolate 5x
num_steps_base = int(run_time / nmf.timestep)
step_duration = len(data['joint_LFCoxa'])
interp_step_duration = int(step_duration * data['meta']['timestep'] / nmf.timestep)
step_data_block_base = np.zeros((len(nmf.actuated_joints), interp_step_duration))
measure_t = np.arange(step_duration) * data['meta']['timestep']
interp_t = np.arange(interp_step_duration) * nmf.timestep
for i, joint in enumerate(nmf.actuated_joints):
    step_data_block_base[i, :] = np.interp(interp_t, measure_t, data[joint])

print(run_time, num_steps_base, nmf.timestep, interp_step_duration, step_duration, 0.5/nmf.timestep)


num_joints_to_visualize = 7
plt.plot(np.arange(step_data_block_base.shape[1]) * nmf.timestep,
         step_data_block_base[:num_joints_to_visualize].T,
         label=nmf.actuated_joints[:num_joints_to_visualize])
plt.legend(ncol = 3)
plt.xlabel('Time (s)')
plt.ylabel('Angle (radian)')
plt.show()

n_timesteps = 1000
for i in range(n_timesteps):
    action = {'joints': step_data_block_base[:, 0]}
    _, _ = nmf.step(action)

all_viewpoints = []
h, w = 400, 500
for viewpoint in ['camera_front', 'camera_left', 'camera_bottom']:
    all_viewpoints.append(nmf.physics.render(camera_id=f"Animat/{viewpoint}", width=w, height=h))


im = PIL.Image.fromarray(np.hstack(all_viewpoints))
#show image

step_data_block_manualcorrect = step_data_block_base.copy()

for side in ["L", "R"]:
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}MCoxa")] += np.deg2rad(
        13)  # Protract the midlegs

    # step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HCoxa")] -= np.deg2rad(0)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HFemur")] += np.deg2rad(
        -5)  # Retract the hindlegs
    # step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HTibia")] += np.deg2rad(0)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}HTarsus1")] -= np.deg2rad(
        15)  #  Tarsus more parallel to the ground (flexed) (also helps with the hindleg retraction)
    # step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}FTibia")] += np.deg2rad(0)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}FFemur")] += np.deg2rad(
        15)  # Protract the forelegs (slightly to conterbalance Tarsus flexion)
    step_data_block_manualcorrect[nmf.actuated_joints.index(f"joint_{side}FTarsus1")] -= np.deg2rad(
        15)  #  Tarsus more parallel to the ground (flexed) (add some retraction of the forelegs)

nmf.reset()

action = {'joints': step_data_block_manualcorrect[:, 0]}
obs, _ = nmf.step(action)
i = 0
legs = ["RF", "LF", "RH", "LH", "RM", "LM"]
all_tarsus5_pos = {l: [] for l in legs}

# Run until the fly speed is less than 50 micron/sec or 50000 steps
while np.sum(np.abs(obs["fly"][1])) > 50 and i < 50000:
    action = {'joints': step_data_block_manualcorrect[:, 0]}
    obs, _ = nmf.step(action)
    i += 1
    nmf.render()

# Keep track of the time it takes to stabilize
n_steps_stabil = i

nmf.save_video(out_dir / "video_initial_stance.mp4")
all_viewpoints = []
h, w = 400, 500
for viewpoint in ['camera_front', 'camera_left', 'camera_bottom']:
    all_viewpoints.append(nmf.physics.render(camera_id=f"Animat/{viewpoint}", width=w, height=h))

im = PIL.Image.fromarray(np.hstack(all_viewpoints))
# show concatenated image

Video.from_file(out_dir / 'video_initial_stance.mp4')

# Check load distribution from manual correction

legs = ["RF", "LF", "RM", "LM", "RH", "LH"]
leg_force_sensors_ids = {leg:[] for leg in legs}
for i, collision_geom in enumerate(nmf.collision_tracked_geoms):
    for leg in legs:
        if collision_geom.startswith(leg+"Tarsus"):
            leg_force_sensors_ids[leg].append(i)

contact_forces = np.array([np.sum(obs["contact_forces"][leg_force_sensors_ids[leg]]) for leg in legs])
color = contact_forces <= 0
plt.scatter(np.arange(len(legs)), contact_forces, c=color)
plt.xticks(np.arange(len(legs)), legs)
plt.xlabel("Leg")
plt.ylabel("Tarsal contact forces")
plt.title("Tarsal contact forces after manual correction")
plt.show()

nmf.reset()

legs = ["RH", "LH", "RF", "LF", "RM", "LM"]
leg_ids = np.arange(len(legs)).astype(int)
leg_corresp_id = dict(zip(legs, leg_ids))
n_joints = len(nmf.actuated_joints)
joint_ids = np.arange(n_joints).astype(int)
# Map the id of the joint to the leg it belongs to (usefull to go through the steps for each legs)
match_leg_to_joints = np.array([i for joint in nmf.actuated_joints for i, leg in enumerate(legs) if leg in joint])

# Map the id of the end effector (Tarsus5) to the leg it belongs to
leg_to_end_effector_id = {leg: i for i, end_effector in enumerate(nmf.end_effector_names) for leg in legs if
                          leg in end_effector if leg in end_effector}

# Number of timesteps between each (fly) step
n_rest_timesteps = 2000

# Map the id of the force sensors to the leg it belongs to
leg_force_sensors_ids = {leg: [] for leg in legs}
for i, collision_geom in enumerate(nmf.collision_tracked_geoms):
    for leg in legs:
        if collision_geom.startswith(leg + "Tarsus"):
            leg_force_sensors_ids[leg].append(i)

        # Record the touch sensor data for each leg for each timepoint
touch_sensor_data = np.zeros((len(legs), interp_step_duration + n_rest_timesteps - 1))

# Get the position of the last segment of the tarsus for each leg in the
leg_tarsi_pos_id = {leg: [i] for leg in legs for i, joint in enumerate(nmf.actuated_joints) if
                    leg in joint and "Tarsus1" in joint}
position_data = np.zeros((len(legs), interp_step_duration + n_rest_timesteps - 1, 3))

# Run the simulation until the fly is stable
for k in range(n_steps_stabil):
    action = {'joints': step_data_block_manualcorrect[joint_ids, 0]}
    obs, info = nmf.step(action)
    nmf.render()

# Lets step each leg on after the other collect touch sensor data as well as 3d coordinates of the last segment of the tarsus
for i, leg in enumerate(legs):

    # "Boolean" like indexer for the stepping leg
    joints_to_actuate = np.zeros(len(nmf.actuated_joints)).astype(int)
    joints_to_actuate[match_leg_to_joints == i] = 1

    for k in range(interp_step_duration):
        # Advance the stepping in the joints of the stepping leg
        joint_pos = step_data_block_manualcorrect[joint_ids, joints_to_actuate * k]
        action = {'joints': joint_pos}
        obs, info = nmf.step(action)
        # Get the touch sensor data from physics (sum of the Tarsus bellonging to a leg)
        touch_sensor_data[i, k] = np.sum(obs['contact_forces'][leg_force_sensors_ids[leg]])
        # Get the position data from physics
        position_data[i, k, :] = obs["end_effectors"].reshape(len(legs), 3)[
            leg_to_end_effector_id[leg]]  # Get the position data from physics

        nmf.render()

    # Rest between steps
    for j in range(n_rest_timesteps):
        action = {'joints': step_data_block_manualcorrect[joint_ids, 0]}
        obs, info = nmf.step(action)
        touch_sensor_data[i, k + j] = np.sum(obs['contact_forces'][leg_force_sensors_ids[leg]])
        position_data[i, k + j, :] = obs["end_effectors"].reshape(len(legs), 3)[
            leg_to_end_effector_id[leg]]  # Get the position data from physics
        nmf.render()

nmf.save_video(out_dir / "video_steps.mp4")

fig, axs = plt.subplots(4, len(legs), figsize=(20, 10))
t = np.arange(touch_sensor_data.shape[1]) * nmf.timestep

leg_swing_starts = {}
leg_stance_starts = {}

stride = 20  # Number of timesteps to check for contact
eps = 10  # Threshold for detecting contact

for i, leg in enumerate(legs):
    # Plot contact forces
    axs[0, i].plot(t, touch_sensor_data[i, :])
    k = 0
    # Until you find a swing onset keep going (as long as k is less than the length of the data)
    while k < len(touch_sensor_data[i]) and not np.all(touch_sensor_data[i, k:k + stride] == 0):
        k += 1
    swing_start = k
    if k < len(touch_sensor_data[i]):
        # Find the first time the contact force is above the threshold
        stance_start = np.where(touch_sensor_data[i, swing_start:] > eps)[0][0] + swing_start
        axs[0, i].axvline(t[swing_start], color='r', label="Swing start")
        axs[0, i].axvline(t[stance_start], color='g', label="Stance start")
        if i == 0:
            axs[0, i].legend()
        leg_swing_starts[leg] = swing_start
        leg_stance_starts[leg] = stance_start
    else:
        leg_swing_starts[leg] = 0
        leg_stance_starts[leg] = 0

    # Plot 3d coordinates of the last segment of the tarsus
    axs[1, i].plot(t, position_data[i, :, 0])
    axs[2, i].plot(t, position_data[i, :, 1])
    axs[3, i].plot(t, position_data[i, :, 2])
    axs[0, i].set_title(leg)
    if i == 0:
        axs[0, i].set_ylabel('Contact force')
        axs[1, i].set_ylabel('Xpos')
        axs[2, i].set_ylabel('Ypos')
        axs[3, i].set_ylabel('Zpos')
for j in range(len(legs)):
    axs[-1, j].set_xlabel('Time [s]')

plt.tight_layout()

Video.from_file(out_dir / 'video_steps.mp4')

# Save the data, the step_timing and the number of stabilisation steps

with open(out_dir / "manual_corrected_data.pickle", 'wb') as handle:
    pickle.dump([step_data_block_manualcorrect, leg_swing_starts, leg_stance_starts, n_steps_stabil], handle, protocol=pickle.HIGHEST_PROTOCOL)

# Initialize simulation
n_stabilisation_steps = n_steps_stabil
# Run the simulation for a few steps to stabilise the system before starting the contoller
num_steps = num_steps_base + n_stabilisation_steps

#Define rule variables
legs = ["RF", "LF", "RM", "LM", "RH", "LH"]
leg_ids = np.arange(len(legs)).astype(int)
leg_corresp_id = dict(zip(legs, leg_ids))
n_joints = len(nmf.actuated_joints)
joint_ids = np.arange(n_joints).astype(int)
match_leg_to_joints = np.array([i  for joint in nmf.actuated_joints for i, leg in enumerate(legs) if leg in joint])

rule1_corresponding_legs = {"LH":["LM"], "LM":["LF"], "LF":[], "RH":["RM"], "RM":["RF"], "RF":[]}
rule2_corresponding_legs = {"LH":["LM", "RH"], "LM":["LF", "RM"], "LF":["RF"], "RH":["RM", "LH"], "RM":["RF", "LM"], "RF":["LF"]}
rule3_corresponding_legs = {"LH":["RH"], "LM":["LH", "RM"], "LF":["LM", "RF"], "RH":["LH"], "RM":["RH", "LM"], "RF":["LF", "RM"]}

#Rule 1 should supress lift off (if a leg is in swing coupled legs should not be lifted most important leg to guarantee stability)
rule1_weight = -1e4
#Rule 2 should facilitate early protraction (upon touchdown of a leg coupled legs are encouraged to swing)
rule2_weight = 2.5
rule2_weight_contralateral = 1
#Rule 3 should enforce late protraction (the later in the stance the more it facilitates stance initiation)
rule3_weight = 3
rule3_weight_contralateral = 2

# This represents the score of each leg in the current step
leg_scores = np.zeros(len(legs))
all_leg_scores = np.zeros((len(legs), num_steps))

# Monitor the evolution of each part of the score for each leg
all_legs_rule1_scores = np.zeros((len(legs), num_steps))
all_legs_rule2_scores = np.zeros((len(legs), num_steps))
all_legs_rule3_scores = np.zeros((len(legs), num_steps))

# one percent margin if leg score within this margin to the max score random choice between the very likely legs
percent_margin = 0.001

# For each leg the ids of the force sensors that are attached to it
leg_force_sensors_ids = {leg:[] for leg in legs}
for i, collision_geom in enumerate(nmf.collision_tracked_geoms):
    for leg in legs:
        if collision_geom.startswith(leg):
            leg_force_sensors_ids[leg].append(i)


def update_stepping_advancement(stepping_advancement, legs, interp_step_duration):
    # Advance the stepping advancement of each leg that are stepping, reset the advancement of the legs that are done stepping
    for k, leg in enumerate(legs):
        if stepping_advancement[k] >= interp_step_duration - 1:
            stepping_advancement[k] = 0
        elif stepping_advancement[k] > 0:
            stepping_advancement[k] += 1
    return stepping_advancement


def compute_leg_scores(rule1_corresponding_legs, rule1_weight,
                       rule2_corresponding_legs, rule2_weight, rule2_weight_contralateral,
                       rule3_corresponding_legs, rule3_weight, rule3_weight_contralateral,
                       stepping_advancement, leg_corresp_id, leg_stance_starts, interp_step_duration):
    # Compute the leg scores for the current  timestep based on the rules and the stepping advancement
    # Fills the global variables all_legs_rule1_scores, all_legs_rule2_scores, all_legs_rule3_scores, all_leg_scores to monitor the evolution of the score (debugging)
    leg_scores = np.zeros(len(legs))

    # Iterate through legs to compute score
    for k, leg in enumerate(legs):
        # For the first rule
        leg_scores[[leg_corresp_id[l] for l in rule1_corresponding_legs[leg]]] += rule1_weight * (
                    stepping_advancement[k] > 0 and stepping_advancement[k] < leg_stance_starts[leg]).astype(float)
        all_legs_rule1_scores[[leg_corresp_id[l] for l in rule1_corresponding_legs[leg]], i] += rule1_weight * (
                    stepping_advancement[k] > 0 and stepping_advancement[k] < leg_stance_starts[leg]).astype(float)

        # For the second rule strong contact force happens at the beggining of the stance phase
        for l in rule2_corresponding_legs[leg]:
            # Decrease with stepping advancement
            if l[0] == leg[0]:
                # ipsilateral leg
                leg_scores[leg_corresp_id[l]] += rule2_weight * ((interp_step_duration - leg_stance_starts[leg]) - (
                            stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] -
                                                                                   leg_stance_starts[leg]) > 0 else 0
                all_legs_rule2_scores[leg_corresp_id[l], i] += rule2_weight * (
                            (interp_step_duration - leg_stance_starts[leg]) - (
                                stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] -
                                                                                       leg_stance_starts[
                                                                                           leg]) > 0 else 0
            else:
                # contralateral leg
                leg_scores[leg_corresp_id[l]] += rule2_weight_contralateral * (
                            (interp_step_duration - leg_stance_starts[leg]) - (
                                stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] -
                                                                                       leg_stance_starts[
                                                                                           leg]) > 0 else 0
                all_legs_rule2_scores[leg_corresp_id[l], i] += rule2_weight_contralateral * (
                            (interp_step_duration - leg_stance_starts[leg]) - (
                                stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] -
                                                                                       leg_stance_starts[
                                                                                           leg]) > 0 else 0

        # For the third rule
        for l in rule3_corresponding_legs[leg]:
            # Increase with stepping advancement
            if l[0] == leg[0]:
                leg_scores[leg_corresp_id[l]] += rule3_weight * (
                (stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] - leg_stance_starts[
                    leg]) > 0 else 0
                all_legs_rule3_scores[leg_corresp_id[l], i] += rule3_weight * (
                (stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] - leg_stance_starts[
                    leg]) > 0 else 0
            else:
                leg_scores[leg_corresp_id[l]] += rule3_weight_contralateral * (
                (stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] - leg_stance_starts[
                    leg]) > 0 else 0
                all_legs_rule3_scores[leg_corresp_id[l], i] += rule3_weight_contralateral * (
                (stepping_advancement[k] - leg_stance_starts[leg])) if (stepping_advancement[k] - leg_stance_starts[
                    leg]) > 0 else 0

    return leg_scores


np.random.seed(42)

# This serves to keep track of the advancement of each leg in the stepping sequence
stepping_advancement = np.zeros(len(legs)).astype(int)

nmf.reset()

# Track the number of steps taken. It will be used to determine the stpping probability in the random stepper
number_of_taken_steps = 0
leg_scores = np.zeros(len(legs))

obs_list_cruse_flat = []
all_initiated_legs = []

# Run the actual simulation
for i in trange(num_steps):

    # Decide in which leg to step
    initiating_leg = np.argmax(leg_scores)
    within_margin_legs = leg_scores[initiating_leg] - leg_scores <= leg_scores[initiating_leg] * percent_margin

    # If multiple legs are within the margin choose randomly among those legs
    if np.sum(within_margin_legs) > 1:
        initiating_leg = np.random.choice(np.where(within_margin_legs)[0])

    # If the maximal score is zero or less (except for the first step after stabilisation to initate the locomotion) or if the leg is already stepping
    if (leg_scores[initiating_leg] <= 0 and not i == n_stabilisation_steps + 1) or stepping_advancement[
        initiating_leg] > 0:
        initiating_leg = None
    else:
        stepping_advancement[initiating_leg] += 1
        all_initiated_legs.append([initiating_leg, i])
        number_of_taken_steps += 1
        # print("Stepping leg: ", legs[initiating_leg], " at step: ", i)

    joint_pos = step_data_block_manualcorrect[joint_ids, stepping_advancement[match_leg_to_joints]]
    action = {'joints': joint_pos}
    obs, info = nmf.step(action)
    nmf.render()
    obs_list_cruse_flat.append(obs)

    stepping_advancement = update_stepping_advancement(stepping_advancement, legs, interp_step_duration)

    leg_scores = compute_leg_scores(rule1_corresponding_legs, rule1_weight,
                                    rule2_corresponding_legs, rule2_weight, rule2_weight_contralateral,
                                    rule3_corresponding_legs, rule3_weight, rule3_weight_contralateral,
                                    stepping_advancement, leg_corresp_id, leg_stance_starts, interp_step_duration)

    all_leg_scores[:, i] = leg_scores

nmf.save_video(out_dir / "cruse_flat.mp4")
nmf.close()

distances = obs_list_cruse_flat[n_stabilisation_steps]["fly"][0][:2] - obs_list_cruse_flat[-1]["fly"][0][:2]
print("Forward distance travelled: ", distances[0])
print("Lateral distance travelled: ", distances[1])

# Plot the evolution of the scores for each leg
fig, axs = plt.subplots(4, 1, figsize=(15, 15), sharex=True, sharey=True)

t_ids = np.arange(0, num_steps, 1)
time = t_ids * nmf.timestep

leg_of_interest = ["RF", "LF", "RM", "LM", "RH", "LH"]
colors = ["r", "b", "g", "k", "c", "m"]
j_ids = np.tile([legs.index(leg) for leg in leg_of_interest], (len(t_ids), 1)).T

for a, (ax, rule_score) in enumerate(
        zip(axs, [all_leg_scores[j_ids, t_ids], all_legs_rule1_scores, all_legs_rule2_scores, all_legs_rule3_scores])):
    for i, l_score in enumerate(rule_score[j_ids, t_ids]):
        if i == 0:
            ax.set_title("Scores for each leg")
        else:
            ax.set_title("Scores for each leg for rule " + str(a))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Scores")

        ax.plot(time, l_score, c=colors[i], linestyle="-", label=leg_of_interest[i])

    for leg_id, step in all_initiated_legs:
        color_index = leg_of_interest.index(legs[leg_id])
        # print(color_index, step)
        ax.axvline(time[step], color=colors[color_index], linestyle=":")
    ax.legend()

fig.tight_layout()
plt.show()
Video.from_file(out_dir / 'cruse_flat.mp4')


# plot the gait diagram

# Build an where lines are legs, columns are timepoint and there is a one in the array if the left leg is in contact with the ground
# and a zero if it is not
all_legs_contact = np.zeros((len(legs), num_steps))
gait_diagram_leg_order = ["RF", "RM", "RH", "LF", "LM", "LH"]

assert set(gait_diagram_leg_order) == set(legs)

for t in range(num_steps):
    for l, leg in enumerate(gait_diagram_leg_order):
        all_legs_contact[l, t] = np.sum(obs_list_cruse_flat[t]["contact_forces"][leg_force_sensors_ids[leg]]) > 0

# plot the gait diagram leg vs time; Should appear black if the leg is in contact with the ground
# median filter the contat to smooth the gait diagram

from scipy.signal import medfilt

plt.figure(figsize=(10, 4))
#all_legs_contact_smooth = medfilt(all_legs_contact, kernel_size=(1, 3))
plt.imshow((np.logical_not(all_legs_contact)), cmap="gray", aspect="auto", interpolation="none")
plt.xlabel("Time [s]")
plt.ylabel("Leg")
plt.yticks(np.arange(len(legs)), gait_diagram_leg_order)
plt.xticks(t_ids[::10000], time[::10000])
plt.title("Gait diagram")
plt.show()

np.random.seed(42)
# Comparison with random is interesting but probability of stepping should be the same

p_step = (number_of_taken_steps) / (
            num_steps - n_stabilisation_steps) * 1.3  # stepping probability with 30% more to give an advantage to the random controller

nmf.reset()

stepping_advancement = np.zeros(len(legs)).astype(int)

obs_list_random_flat = []
n_step_random_taken = 0
n_late_steps = 0

# Run the actual simulation
for i in trange(num_steps):

    # Decide if going to step or not
    p = np.random.rand()
    if i > n_stabilisation_steps and p < p_step:
        # Decide in which leg to step if all legs are already stepping increment late step counter
        try:
            initiating_leg = np.random.choice(leg_ids[stepping_advancement <= 0])
            n_step_random_taken += 1
            stepping_advancement[initiating_leg] += 1
        except:
            n_late_steps += 1

    if n_late_steps > 0 and np.any(stepping_advancement <= 0):
        # If some step could not have been perfomed previously, we an perfom them as soon as the steppinf advancement is zero
        initiating_leg = np.random.choice(leg_ids[stepping_advancement <= 0])
        n_step_random_taken += 1
        stepping_advancement[initiating_leg] += 1
        n_late_steps -= 1

    joint_pos = step_data_block_manualcorrect[joint_ids, stepping_advancement[match_leg_to_joints]]
    action = {'joints': joint_pos}
    obs, info = nmf.step(action)
    nmf.render()
    obs_list_random_flat.append(obs)

    # Compute score and update stepping advancement
    leg_scores = np.zeros(len(legs))
    for k, leg in enumerate(legs):
        # Update_the_stepping_advancement
        if stepping_advancement[k] >= interp_step_duration - 1:
            stepping_advancement[k] = 0
        elif stepping_advancement[k] > 0:
            stepping_advancement[k] += 1

    all_leg_scores[:, i] = leg_scores

distances = obs_list_random_flat[n_stabilisation_steps]["fly"][0][:2] - obs_list_random_flat[-1]["fly"][0][:2]
print("Forward distance travelled: ", distances[0])
print("Lateral distance travelled: ", distances[1])

print("Target number of steps: ", number_of_taken_steps, "\n"
                                                         "Number of steps taken by random: ", n_step_random_taken)

nmf.save_video(out_dir / 'random_flat.mp4')
# nmf.close()

Video.from_file(out_dir / 'random_flat.mp4')