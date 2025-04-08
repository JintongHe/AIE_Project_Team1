from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from loco_mujoco import LocoEnv
from ModelsAndUtils import MLP, get_right_ankle_substate, get_action_substate
from RL import PolicyNet

def analyze_gait_cycles(data, window_size=50):
    """
    Analyze the gait cycles from time series data
    
    Args:
        data: Dictionary containing time series data
        window_size: Window size for smoothing
        
    Returns:
        dict: Dictionary with gait cycle information
    """
    # Smooth the velocity data to better identify cycles
    def moving_average(x, w):
        return np.convolve(x, np.ones(w), 'valid') / w
    
    # Smooth velocity data
    smoothed_vel_x = moving_average(data['com_vel_x'], window_size)
    
    # Find peaks in velocity (can indicate steps)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(smoothed_vel_x, height=0, distance=10)
    
    # Calculate gait cycle information
    if len(peaks) > 1:
        # Calculate cycle durations
        cycle_times = []
        cycle_lengths = []
        
        for i in range(1, len(peaks)):
            idx1, idx2 = peaks[i-1], peaks[i]
            # Adjusted for the window size offset
            time1 = data['time'][idx1 + window_size//2]
            time2 = data['time'][idx2 + window_size//2]
            
            # Duration of this cycle
            cycle_time = time2 - time1
            cycle_times.append(cycle_time)
            
            # Distance traveled during this cycle
            pos1 = data['com_pos_x'][idx1 + window_size//2]
            pos2 = data['com_pos_x'][idx2 + window_size//2]
            cycle_length = pos2 - pos1
            cycle_lengths.append(cycle_length)
        
        gait_info = {
            'cycle_times': np.array(cycle_times),
            'cycle_lengths': np.array(cycle_lengths),
            'avg_cycle_time': np.mean(cycle_times),
            'avg_cycle_length': np.mean(cycle_lengths),
            'avg_velocity': np.mean(cycle_lengths) / np.mean(cycle_times),
            'peaks': peaks + window_size//2,  # Adjust peak indices for original data
        }
    else:
        gait_info = {
            'message': 'Not enough gait cycles detected',
            'peaks': peaks,
        }
    
    return gait_info

def plot_gait_comparison(ppo_data, expert_data):
    """
    Plot a comparison of gait cycles between PPO and expert models
    
    Args:
        ppo_data: Dictionary with PPO model data
        expert_data: Dictionary with expert model data
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot COM velocity in X direction (forward)
    axs[0, 0].plot(ppo_data['time'], ppo_data['com_vel_x'], 'b-', label='PPO Model')
    axs[0, 0].plot(expert_data['time'], expert_data['com_vel_x'], 'r-', label='Expert Model')
    axs[0, 0].set_title('COM Velocity - X Direction (Forward)')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Velocity (m/s)')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot COM position in X direction
    axs[0, 1].plot(ppo_data['time'], ppo_data['com_pos_x'], 'b-', label='PPO Model')
    axs[0, 1].plot(expert_data['time'], expert_data['com_pos_x'], 'r-', label='Expert Model')
    axs[0, 1].set_title('COM Position - X Direction')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Position (m)')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot vertical COM velocity (Z)
    axs[1, 0].plot(ppo_data['time'], ppo_data['com_vel_z'], 'b-', label='PPO Model')
    axs[1, 0].plot(expert_data['time'], expert_data['com_vel_z'], 'r-', label='Expert Model')
    axs[1, 0].set_title('COM Velocity - Z Direction (Vertical)')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Velocity (m/s)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot reward over time
    axs[1, 1].plot(ppo_data['time'], ppo_data['rewards'], 'b-', label='PPO Model')
    axs[1, 1].plot(expert_data['time'], expert_data['rewards'], 'r-', label='Expert Model')
    axs[1, 1].set_title('Reward Over Time')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Reward')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    # Make the layout tight
    plt.tight_layout()
    plt.savefig('gait_comparison.png')
    plt.show()

def simplified_test(policy, env, agent, device, steps=200, use_policy=True):
    state = env.reset()
    ankle_state = get_right_ankle_substate(state)
    
    # Data collection
    positions = []
    velocities = []
    
    for step in range(steps):
        env.render()
        
        # Get action
        if use_policy:
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(ankle_state, dtype=torch.float32).to(device)
                action = policy.get_best_action(ankle_state_tensor)
            
            body_action = agent.draw_action(state)
            body_action[7] = action.item()
        else:
            body_action = agent.draw_action(state)
        
        # Take step
        next_state, reward, done, _ = env.step(body_action)
        
        # Extract position and velocity (assuming first elements are position and 18-20 are velocity)
        positions.append(next_state[0:3].copy())
        velocities.append(next_state[18:21].copy() if len(next_state) > 20 else np.zeros(3))
        
        state = next_state
        ankle_state = get_right_ankle_substate(state)
        
        if done:
            break
    
    return np.array(positions), np.array(velocities)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    else: 
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.real"
    env = LocoEnv.make(env_id, use_box_feet=True)

    if hasattr(env, 'seed'):
        env.seed(42)  # Use the same fixed seed

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "real_180.msh")
    agent = Agent.load(agent_file_path)

    # Load your trained PPO policy
    state_dim = 36  # Number of features in the substate
    action_dim = 1  # Number of actions
    policy = PolicyNet(state_dim, action_dim).to(device)
    policy_load_path = os.path.join(os.path.dirname(__file__), "RL_survival_36 states_noprosthetic.pth")
    policy.load_state_dict(torch.load(policy_load_path, weights_only=True))  # Added weights_only=True to fix warning
    policy.eval()
    print(f"Model weights loaded from {policy_load_path}")
    
    # Run simplified tests
    print("Testing PPO model...")
    ppo_positions, ppo_velocities = simplified_test(policy, env, agent, device=device, steps=500, use_policy=True)
    
    env = LocoEnv.make(env_id, use_box_feet=True)
    print("Testing expert model...")
    expert_positions, expert_velocities = simplified_test(policy, env, agent, device=device, steps=500, use_policy=False)

    
    # Plot simplified comparison
    plt.figure(figsize=(15, 10))
    
    # Plot X positions
    plt.subplot(2, 2, 1)
    # # NORMALIZATION: ###
    ppo_pos_x_normalized = ppo_positions[:, 0] - abs(ppo_positions[:, 0][0]) 
    expert_pos_x_normalized = expert_positions[:, 0] - abs(expert_positions[:, 0][0])
    # plt.plot(ppo_positions[:, 0], label='PPO X Position')
    # plt.plot(expert_positions[:, 0], label='Expert X Position')
    plt.plot(ppo_pos_x_normalized, label='PPO X Position')
    plt.plot(expert_pos_x_normalized, label='Expert X Position')
    plt.title('X Position Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot X velocities
    plt.subplot(2, 2, 2)
    plt.plot(ppo_velocities[:, 0], label='PPO X Velocity')
    plt.plot(expert_velocities[:, 0], label='Expert X Velocity')
    plt.title('X Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot Z positions
    plt.subplot(2, 2, 3)
    plt.plot(ppo_positions[:, 2], label='PPO Z Position')
    plt.plot(expert_positions[:, 2], label='Expert Z Position')
    plt.title('Z Position Comparison')
    plt.legend()
    plt.grid(True)
    
    # Plot Z velocities
    plt.subplot(2, 2, 4)
    plt.plot(ppo_velocities[:, 2], label='PPO Z Velocity')
    plt.plot(expert_velocities[:, 2], label='Expert Z Velocity')
    plt.title('Z Velocity Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simplified_comparison.png')
    plt.show()
    
    # Close the environment
    env.close()
    print("Done!")

if __name__ == '__main__':
    main()