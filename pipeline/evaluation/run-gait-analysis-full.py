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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from scipy.signal import find_peaks, butter, filtfilt

sns.set(style="whitegrid")


# PPO Policy
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)  # Increased network size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)

        # Mean output for continuous actions
        self.mean = nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = nn.Linear(32, action_dim)

        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = F.tanh(self.mean(x))  # Tanh ensures output in [-1, 1] range
        std = torch.exp(torch.clamp(self.logstd(x), -9, 0.5))
        return mean, std

    def get_distribution(self, state):
        """Get the distribution over actions for a given state"""
        mean, std = self.forward(state)
        return Normal(mean, std)

    def sample_action(self, state):
        """Sample actions from distribution for training"""
        dist = self.get_distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

    def get_best_action(self, state):
        """Deterministic action selection for testing"""
        mean, _ = self.forward(state)
        return mean

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


def run_rollout(agent_type, agent, model=None, num_steps=1000, filter_cutoff=0.05, num_dim=36):
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
        elif agent_type == "mlp":  # MLP
            # Extract substate for prediction
            right_ankle_substate = get_right_ankle_substate(state, num_dim)
            substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ankle_action = model(substate_tensor).squeeze().item()

            # Get full action from expert but replace ankle
            action = agent.draw_action(state)
            action[7] = ankle_action
        else:
            # Get the best action from the policy
            right_ankle_substate = get_right_ankle_substate(state, num_dim)
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(right_ankle_substate, dtype=torch.float32).to('cpu')
                ankle_action = model.get_best_action(ankle_state_tensor)[0]
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


def plot_normalized_cycles(expert_cycles, mlp_cycles, rl_cycles, cycle_type="kinetics", num_cycles=5):
    """
    Plot the first n cycles with normalized time.
    """
    # Resample both sets of cycles to a common length
    resampled_expert_cycles = resample_cycles(expert_cycles[:num_cycles])
    resampled_mlp_cycles = resample_cycles(mlp_cycles[:num_cycles])
    resampled_rl_cycles = resample_cycles(rl_cycles[:num_cycles])
    mlp_error_cycles = np.abs(resampled_mlp_cycles - resampled_expert_cycles)
    rl_error_cycles = np.abs(resampled_rl_cycles - resampled_expert_cycles)

    # Calculate mean and std for each point in the cycle
    expert_mean = np.mean(resampled_expert_cycles, axis=0)
    expert_std = np.std(resampled_expert_cycles, axis=0)

    mlp_mean = np.mean(resampled_mlp_cycles, axis=0)
    mlp_std = np.std(resampled_mlp_cycles, axis=0)

    rl_mean = np.mean(resampled_rl_cycles, axis=0)
    rl_std = np.std(resampled_rl_cycles, axis=0)

    mlp_error_mean = np.mean(mlp_error_cycles, axis=0)
    mlp_error_std = np.std(mlp_error_cycles, axis=0)

    rl_error_mean = np.mean(rl_error_cycles, axis=0)
    rl_error_std = np.std(rl_error_cycles, axis=0)

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
    plt.plot(cycle_percent, expert_mean, 'k-', linewidth=2, label='Expert Agent Mean')
    plt.fill_between(cycle_percent,
                     expert_mean - expert_std,
                     expert_mean + expert_std,
                     alpha=0.3, color='black')

    # Plot MLP agent with shaded std
    plt.plot(cycle_percent, mlp_mean, 'b-', linewidth=2, label='MLP Agent Mean')
    plt.fill_between(cycle_percent,
                     mlp_mean - mlp_std,
                     mlp_mean + mlp_std,
                     alpha=0.3, color='blue')

    # Plot RL agent with shaded std
    plt.plot(cycle_percent, rl_mean, 'g-', linewidth=2, label='RL Agent Mean')
    plt.fill_between(cycle_percent,
                     rl_mean - rl_std,
                     rl_mean + rl_std,
                     alpha=0.3, color='green')

    # Plot MLP error with shaded std
    plt.plot(cycle_percent, mlp_error_mean, 'r-', linewidth=2, label='MLP Error Mean')
    plt.fill_between(cycle_percent,
                     mlp_error_mean - mlp_error_std,
                     mlp_error_mean + mlp_error_std,
                     alpha=0.3, color='red')

    # Plot RL error with shaded std
    plt.plot(cycle_percent, rl_error_mean, 'm-', linewidth=2, label='RL Agent Mean')
    plt.fill_between(cycle_percent,
                     rl_error_mean - rl_error_std,
                     rl_error_mean + rl_error_std,
                     alpha=0.3, color='magenta')

    # Add labels and title
    plt.xlabel('Gait Cycle (%)', fontsize=14)
    plt.ylabel(y_label, fontsize=14)
    plt.title(f'{title_prefix} Mean and Standard Deviation of First {num_cycles} Gait Cycles\n'
              f'MLP MAE: {np.mean(mlp_error_mean):.2f}, RL MAE: {np.mean(rl_error_mean):.2f}',
              fontsize=16)
    plt.grid(True)
    # plt.legend(fontsize=12)

    # Plot individual cycles
    plt.figure(figsize=(15, 10))

    for i in range(min(num_cycles, len(resampled_expert_cycles))):
        plt.subplot(num_cycles, 1, i + 1)

        # Use normalized time (percentage of gait cycle)
        plt.plot(cycle_percent, resampled_expert_cycles[i], 'b-', label=f'Expert Cycle {i + 1}')
        plt.plot(cycle_percent, resampled_mlp_cycles[i], 'r--', label=f'MLP Cycle {i + 1}')
        plt.plot(cycle_percent, resampled_rl_cycles[i], 'g--', label=f'RL Cycle {i + 1}')

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
    num_cycles = 3
    num_steps = 1000
    num_dim = 16

    # Load expert agent
    # perfect_88_original.msh
    # perfect_140_prosthesis_inertia.msh
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)

    # Load MLP model
    model = MLP(num_dim, 128, 1)
    # mlp_state_36_hidden_128_perfect_88.pth
    # mlp_state_36_hidden_128_prosthesis.pth
    # mlp_state_22_hidden_128_perfect_88.pth
    # mlp_state_22_hidden_128_prosthesis.pth
    # mlp_state_16_hidden_128_prosthesis.pth
    model_load_path = os.path.join(os.path.dirname(__file__), "mlp_state_16_hidden_128_prosthesis.pth")
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    model.eval()

    # Load RL model
    rl_model = PolicyNet(num_dim, 1)
    # new_36_states_survival.pth
    # new_36_states_survival_pros.pth
    # new_22_states_survival.pth
    # new_22_states_survival_pros3.pth
    # new_16_states_survival.pth
    # new_16_states_survival_pros2.pth
    rl_model_load_path = os.path.join(os.path.dirname(__file__), "new_16_states_survival_pros2.pth")
    rl_model.load_state_dict(torch.load(rl_model_load_path, map_location=torch.device('cpu')))
    rl_model.eval()

    print("Running RL agent rollout...")
    rl_smooth_action, rl_raw_action, rl_smooth_angle, rl_raw_angle, rl_time = run_rollout(
        "rl", agent, rl_model, filter_cutoff=filter_cutoff, num_dim=num_dim, num_steps=num_steps)
    print("Running expert agent rollout...")
    expert_smooth_action, expert_raw_action, expert_smooth_angle, expert_raw_angle, expert_time = run_rollout(
        "expert", agent, filter_cutoff=filter_cutoff, num_dim=num_dim, num_steps=num_steps)

    print("Running MLP agent rollout...")
    mlp_smooth_action, mlp_raw_action, mlp_smooth_angle, mlp_raw_angle, mlp_time = run_rollout(
        "mlp", agent, model, filter_cutoff=filter_cutoff, num_dim=num_dim, num_steps=num_steps)

    # print("Running RL agent rollout...")
    # rl_smooth_action, rl_raw_action, rl_smooth_angle, rl_raw_angle, rl_time = run_rollout(
    #     "rl", agent, rl_model, filter_cutoff=filter_cutoff)

    # Identify gait cycles using kinetics (actions)
    print("Identifying expert gait cycles...")
    expert_action_cycles, expert_action_cycle_times, expert_valleys = identify_gait_cycles(
        expert_smooth_action, expert_time)

    print("Identifying MLP gait cycles...")
    mlp_action_cycles, mlp_action_cycle_times, mlp_valleys = identify_gait_cycles(
        mlp_smooth_action, mlp_time)

    print("Identifying RL gait cycles...")
    rl_action_cycles, rl_action_cycle_times, rl_valleys = identify_gait_cycles(
        rl_smooth_action, rl_time)

    # Extract kinematics cycles using the same valley indices
    expert_angle_cycles = []
    mlp_angle_cycles = []
    rl_angle_cycles = []

    for i in range(len(expert_valleys) - 1):
        start_idx = expert_valleys[i]
        end_idx = expert_valleys[i + 1]
        expert_angle_cycles.append(expert_smooth_angle[start_idx:end_idx])

    for i in range(len(mlp_valleys) - 1):
        start_idx = mlp_valleys[i]
        end_idx = mlp_valleys[i + 1]
        mlp_angle_cycles.append(mlp_smooth_angle[start_idx:end_idx])

    for i in range(len(rl_valleys) - 1):
        start_idx = rl_valleys[i]
        end_idx = rl_valleys[i + 1]
        rl_angle_cycles.append(rl_smooth_angle[start_idx:end_idx])

    # Print cycle information
    print(f"Found {len(expert_action_cycles)} expert gait cycles")
    print(f"Found {len(mlp_action_cycles)} MLP gait cycles")
    print(f"Found {len(rl_action_cycles)} RL gait cycles")

    # Plot raw and smoothed kinetics data with valley markers
    plt.figure(figsize=(21, 10))

    # Plot expert kinetics data
    plt.subplot(3, 1, 1)
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
    plt.subplot(3, 1, 2)
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

    # Plot RL kinetics data
    plt.subplot(3, 1, 3)
    plt.plot(rl_time, rl_raw_action, 'r-', alpha=0.3, label='RL Raw')
    plt.plot(rl_time, rl_smooth_action, 'r-', linewidth=2, label='RL Smoothed')

    # Mark the valleys for RL
    valley_times = [rl_time[v] for v in rl_valleys]
    valley_values = [rl_smooth_action[v] for v in rl_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='RL Valleys')

    plt.title('RL Agent Actions (Kinetics) with Identified Gait Cycles', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Action Value (Nm)', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plot raw and smoothed kinematics data with the same valley markers
    plt.figure(figsize=(21, 10))

    # Plot expert kinematics data
    plt.subplot(3, 1, 1)
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
    plt.subplot(3, 1, 2)
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

    # Plot RL kinematics data
    plt.subplot(3, 1, 3)
    plt.plot(rl_time, rl_raw_angle, 'r-', alpha=0.3, label='RL Raw')
    plt.plot(rl_time, rl_smooth_angle, 'r-', linewidth=2, label='RL Smoothed')

    # Mark the valleys with same indices as kinetics
    valley_times = [rl_time[v] for v in rl_valleys]
    valley_values = [rl_smooth_angle[v] for v in rl_valleys]
    plt.plot(valley_times, valley_values, 'go', markersize=8, label='RL Valleys')

    plt.title('RL Agent Joint Angle (Kinematics) with Identified Gait Cycles', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Joint Angle (rad)', fontsize=14)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Plot normalized kinetics gait cycles
    plot_normalized_cycles(expert_action_cycles, mlp_action_cycles, rl_action_cycles, "kinetics", num_cycles)

    # Plot normalized kinematics gait cycles
    plot_normalized_cycles(expert_angle_cycles, mlp_angle_cycles, rl_angle_cycles, "kinematics", num_cycles)

    plt.show()


if __name__ == "__main__":
    main()
