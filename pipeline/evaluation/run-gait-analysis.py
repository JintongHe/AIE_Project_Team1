import matplotlib

matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from loco_mujoco import LocoEnv
from mushroom_rl.core import Agent
from ModelsAndUtils import MLP, get_right_ankle_substate
import os
from scipy.signal import find_peaks, butter, filtfilt

sns.set(style="whitegrid")


def apply_lowpass_filter(data, cutoff=0.1, fs=1.0, order=4):
    """
    Apply a low-pass Butterworth filter to smooth the data.
    """
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def identify_gait_cycles(actions, time_steps, min_distance=50, prominence=30, threshold=-100):
    """
    Identify gait cycles using valleys below threshold.
    """
    # First find all potential valleys
    valleys, _ = find_peaks(-np.array(actions), distance=min_distance, prominence=prominence)

    # Only keep valleys where the action value is less than the threshold
    filtered_valleys = [v for v in valleys if actions[v] < threshold]

    # Make sure we have enough valleys for the cycles
    if len(filtered_valleys) < 6:  # Need at least 6 valleys for 5 cycles
        print(
            f"Warning: Only found {len(filtered_valleys)} valleys below threshold {threshold}. Need at least 6 for 5 complete cycles.")

    # The valleys represent the beginning of each gait cycle
    cycles = []
    cycle_times = []

    # Extract each cycle
    for i in range(len(filtered_valleys) - 1):
        start_idx = filtered_valleys[i]
        end_idx = filtered_valleys[i + 1]

        # Append the action values for this cycle
        cycles.append(actions[start_idx:end_idx])
        cycle_times.append(time_steps[start_idx:end_idx])

    return cycles, cycle_times, filtered_valleys


def resample_cycles(cycles, target_length=100):
    """
    Resample cycles to a common length for averaging.
    """
    resampled_cycles = []

    for cycle in cycles:
        # Create a linear space of the target length
        original_indices = np.linspace(0, len(cycle) - 1, len(cycle))
        new_indices = np.linspace(0, len(cycle) - 1, target_length)

        # Resample the cycle to the target length
        resampled_cycle = np.interp(new_indices, original_indices, cycle)
        resampled_cycles.append(resampled_cycle)

    return np.array(resampled_cycles)


def run_rollout(agent_type, agent, model=None, num_steps=1000, filter_cutoff=0.05):
    """
    Run a rollout using either the expert agent or MLP model.
    """
    mdp = LocoEnv.make("HumanoidTorque.walk.perfect", use_box_feet=True)
    state = mdp.reset()
    done = False

    actions = []
    joint_angles = []  # Record the joint angle (kinematics)
    time_steps = []

    step = 0
    while not done and step < num_steps:
        # Get action based on agent type
        if agent_type == "expert":
            action = agent.draw_action(state)
            ankle_action = action[7]  # Get right ankle action
        else:  # MLP
            # Extract substate for prediction
            right_ankle_substate = get_right_ankle_substate(state)
            substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ankle_action = model(substate_tensor).squeeze().item()

            # Get full action from expert but replace ankle
            action = agent.draw_action(state)
            action[7] = ankle_action

        # Record actions and time steps
        actions.append(ankle_action * 500)  # Scale to Nm

        # Record joint angle (state[7] is right ankle joint angle)
        joint_angles.append(state[7])

        time_steps.append(step)

        # Step environment
        state, _, done, _ = mdp.step(action)
        step += 1

    # Apply low-pass filter to smooth the data
    smoothed_actions = apply_lowpass_filter(actions, cutoff=filter_cutoff)
    smoothed_angles = apply_lowpass_filter(joint_angles, cutoff=filter_cutoff)

    return smoothed_actions, actions, smoothed_angles, joint_angles, time_steps


def plot_normalized_cycles(expert_cycles, mlp_cycles, cycle_type="kinetics", num_cycles=5):
    """
    Plot the first n cycles with normalized time.
    """
    # Resample both sets of cycles to a common length
    resampled_expert_cycles = resample_cycles(expert_cycles[:num_cycles])
    resampled_mlp_cycles = resample_cycles(mlp_cycles[:num_cycles])

    # Calculate mean and std for each point in the cycle
    expert_mean = np.mean(resampled_expert_cycles, axis=0)
    expert_std = np.std(resampled_expert_cycles, axis=0)

    mlp_mean = np.mean(resampled_mlp_cycles, axis=0)
    mlp_std = np.std(resampled_mlp_cycles, axis=0)

    # Create a normalized x-axis for the cycle (0 to 100% of gait cycle)
    cycle_percent = np.linspace(0, 100, len(expert_mean))

    # Determine y-axis label based on cycle type
    if cycle_type == "kinetics":
        y_label = 'Action Value (Nm)'
        title_prefix = 'Kinetics:'
    else:  # kinematics
        y_label = 'Joint Angle (rad)'
        title_prefix = 'Kinematics:'

    # Create figure for mean and std visualization
    plt.figure(figsize=(14, 8))

    # Plot expert agent with shaded std
    plt.plot(cycle_percent, expert_mean, 'b-', linewidth=2, label='Expert Agent Mean')
    plt.fill_between(cycle_percent,
                     expert_mean - expert_std,
                     expert_mean + expert_std,
                     alpha=0.3, color='blue')

    # Plot MLP agent with shaded std
    plt.plot(cycle_percent, mlp_mean, 'r--', linewidth=2, label='MLP Agent Mean')
    plt.fill_between(cycle_percent,
                     mlp_mean - mlp_std,
                     mlp_mean + mlp_std,
                     alpha=0.3, color='red')

    # Add labels and title
    plt.xlabel('Gait Cycle (%)', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'{title_prefix} Mean and Standard Deviation of First {num_cycles} Gait Cycles\nNormalized Time',
              fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=12)

    # Plot individual cycles
    plt.figure(figsize=(15, 10))

    for i in range(min(num_cycles, len(resampled_expert_cycles))):
        plt.subplot(num_cycles, 1, i + 1)

        # Use normalized time (percentage of gait cycle)
        plt.plot(cycle_percent, resampled_expert_cycles[i], 'b-', label=f'Expert Cycle {i + 1}')
        plt.plot(cycle_percent, resampled_mlp_cycles[i], 'r--', label=f'MLP Cycle {i + 1}')

        if i == 0:
            plt.title(f'{title_prefix} Individual Gait Cycles (Normalized Time)', fontsize=16)
        if i == num_cycles - 1:
            plt.xlabel('Gait Cycle (%)', fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.grid(True)
        plt.legend(loc='upper right')

    plt.tight_layout()


def main():
    # Initialize parameters
    filter_cutoff = 0.05
    num_cycles = 5
    num_steps = 1000

    # Load expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_80_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)

    # Load MLP model
    model = MLP(22, 128, 1)
    model_load_path = os.path.join(os.path.dirname(__file__), "mlp_state_22_hidden_128_perfect_80_prosthesis_inertia.pth")
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    model.eval()

    print("Running expert agent rollout...")
    expert_smooth_action, expert_raw_action, expert_smooth_angle, expert_raw_angle, expert_time = run_rollout(
        "expert", agent, filter_cutoff=filter_cutoff)

    print("Running MLP agent rollout...")
    mlp_smooth_action, mlp_raw_action, mlp_smooth_angle, mlp_raw_angle, mlp_time = run_rollout(
        "mlp", agent, model, filter_cutoff=filter_cutoff)

    # Identify gait cycles using kinetics (actions)
    print("Identifying expert gait cycles...")
    expert_action_cycles, expert_action_cycle_times, expert_valleys = identify_gait_cycles(
        expert_smooth_action, expert_time)

    print("Identifying MLP gait cycles...")
    mlp_action_cycles, mlp_action_cycle_times, mlp_valleys = identify_gait_cycles(
        mlp_smooth_action, mlp_time)

    # Extract kinematics cycles using the same valley indices
    expert_angle_cycles = []
    mlp_angle_cycles = []

    for i in range(len(expert_valleys) - 1):
        start_idx = expert_valleys[i]
        end_idx = expert_valleys[i + 1]
        expert_angle_cycles.append(expert_smooth_angle[start_idx:end_idx])

    for i in range(len(mlp_valleys) - 1):
        start_idx = mlp_valleys[i]
        end_idx = mlp_valleys[i + 1]
        mlp_angle_cycles.append(mlp_smooth_angle[start_idx:end_idx])

    # Print cycle information
    print(f"Found {len(expert_action_cycles)} expert gait cycles")
    print(f"Found {len(mlp_action_cycles)} MLP gait cycles")

    # Plot raw and smoothed kinetics data with valley markers
    plt.figure(figsize=(14, 10))

    # Plot expert kinetics data
    plt.subplot(2, 1, 1)
    plt.plot(expert_time, expert_raw_action, 'b-', alpha=0.3, label='Expert Raw')
    plt.plot(expert_time, expert_smooth_action, 'b-', linewidth=2, label='Expert Smoothed')

    # Mark the valleys for expert
    valley_times = [expert_time[v] for v in expert_valleys]
    valley_values = [expert_smooth_action[v] for v in expert_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='Expert Valleys')

    plt.title('Expert Agent Actions (Kinetics) with Identified Gait Cycles', fontsize=16)
    plt.ylabel('Action Value (Nm)', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Plot MLP kinetics data
    plt.subplot(2, 1, 2)
    plt.plot(mlp_time, mlp_raw_action, 'r-', alpha=0.3, label='MLP Raw')
    plt.plot(mlp_time, mlp_smooth_action, 'r-', linewidth=2, label='MLP Smoothed')

    # Mark the valleys for MLP
    valley_times = [mlp_time[v] for v in mlp_valleys]
    valley_values = [mlp_smooth_action[v] for v in mlp_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='MLP Valleys')

    plt.title('MLP Agent Actions (Kinetics) with Identified Gait Cycles', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Action Value (Nm)', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plot raw and smoothed kinematics data with the same valley markers
    plt.figure(figsize=(14, 10))

    # Plot expert kinematics data
    plt.subplot(2, 1, 1)
    plt.plot(expert_time, expert_raw_angle, 'b-', alpha=0.3, label='Expert Raw')
    plt.plot(expert_time, expert_smooth_angle, 'b-', linewidth=2, label='Expert Smoothed')

    # Mark the valleys with same indices as kinetics
    valley_times = [expert_time[v] for v in expert_valleys]
    valley_values = [expert_smooth_angle[v] for v in expert_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='Expert Valleys')

    plt.title('Expert Agent Joint Angle (Kinematics) with Identified Gait Cycles', fontsize=16)
    plt.ylabel('Joint Angle (rad)', fontsize=14)
    plt.grid(True)
    plt.legend()

    # Plot MLP kinematics data
    plt.subplot(2, 1, 2)
    plt.plot(mlp_time, mlp_raw_angle, 'r-', alpha=0.3, label='MLP Raw')
    plt.plot(mlp_time, mlp_smooth_angle, 'r-', linewidth=2, label='MLP Smoothed')

    # Mark the valleys with same indices as kinetics
    valley_times = [mlp_time[v] for v in mlp_valleys]
    valley_values = [mlp_smooth_angle[v] for v in mlp_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='MLP Valleys')

    plt.title('MLP Agent Joint Angle (Kinematics) with Identified Gait Cycles', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Joint Angle (rad)', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plot normalized kinetics gait cycles
    plot_normalized_cycles(expert_action_cycles, mlp_action_cycles, "kinetics", num_cycles)

    # Plot normalized kinematics gait cycles
    plot_normalized_cycles(expert_angle_cycles, mlp_angle_cycles, "kinematics", num_cycles)

    plt.show()


if __name__ == "__main__":
    main()