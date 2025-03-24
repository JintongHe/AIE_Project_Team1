from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from loco_mujoco import LocoEnv
from ModelsAndUtils import MLP, get_right_ankle_substate, get_action_substate

# Import your PolicyNet class from your existing code
class PolicyNet(torch.nn.Module):
    # Include your PolicyNet definition here
    # This should be identical to the one in your paste.txt
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        
        # Mean output for continuous actions
        self.mean = torch.nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = torch.nn.Linear(32, action_dim)
        
        self.action_dim = action_dim

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        mean = torch.nn.functional.tanh(self.mean(x))
        std = torch.exp(torch.clamp(self.logstd(x), -9, 0.5))
        return mean, std
    
    def get_distribution(self, state):
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)
    
    def sample_action(self, state):
        dist = self.get_distribution(state)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)
    
    def get_best_action(self, state):
        mean, _ = self.forward(state)
        return mean

def collect_gait_data(policy, env, agent, num_steps=2000, device="mps", render=False):
    """
    Collect gait cycle data for a policy
    
    Args:
        policy: The policy to evaluate (PPO or imitation learning)
        env: The environment
        agent: The expert agent
        num_steps: Number of steps to run
        device: The device to run on
        render: Whether to render the environment
        
    Returns:
        dict: A dictionary containing time series data of COM velocities and positions
    """
    state = env.reset()
    ankle_state = get_right_ankle_substate(state)
    
    # Data collection dictionaries
    data = {
        'time': [],
        'com_vel_x': [],
        'com_vel_y': [],
        'com_vel_z': [],
        'com_pos_x': [],
        'com_pos_y': [],
        'com_pos_z': [],
        'joint_angles': [],
        'rewards': [],
    }
    
    for step in range(num_steps):
        if render:
            env.render()
            
        # Get the action from policy
        if policy is not None:
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(ankle_state, dtype=torch.float32).to(device)
                action = policy.get_best_action(ankle_state_tensor)
                
            # Combine with expert agent
            body_action = agent.draw_action(state)
            body_action[7] = action.item()
        else:
            # If no policy provided, use expert agent only (imitation learning)
            body_action = agent.draw_action(state)
        
        # Take step in environment
        next_state, reward, done, info = env.step(body_action)
        
        # Access COM velocity data - this is the key part you were asking about
        # Based on your error, we need to access the simulator differently
        
        # In LocoMujoco, velocities are often part of the state vector
        # The first elements (0-2) are typically the COM velocity components
        com_vel = state[18:21]  # Positions 18, 19, 20 likely contain COM velocities
                                # This is an educated guess based on standard MuJoCo state vectors
        
        # Get COM position - typically the first 3 positions in the state
        com_pos = state[0:3]
        
        # Alternative: Try to access from info dictionary if available
        if 'com_vel' in info:
            com_vel = info['com_vel']
        if 'com_pos' in info:
            com_pos = info['com_pos']
        
        # Store data
        data['time'].append(step * env.dt)  # Assuming env.dt is the timestep
        data['com_vel_x'].append(com_vel[0])
        data['com_vel_y'].append(com_vel[1])
        data['com_vel_z'].append(com_vel[2])
        data['com_pos_x'].append(com_pos[0])
        data['com_pos_y'].append(com_pos[1])
        data['com_pos_z'].append(com_pos[2])
        data['rewards'].append(reward)
        
        # Store joint angles if needed
        joint_angles = sim.data.qpos[7:].copy()  # Skip first 7 DoFs (global pos/orientation)
        data['joint_angles'].append(joint_angles)
        
        # Update state
        state = next_state
        ankle_state = get_right_ankle_substate(state)
        
        if done:
            break
    
    # Convert lists to numpy arrays for easier processing
    for key in data:
        data[key] = np.array(data[key])
    
    return data

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

def inspect_environment_info(env):
    """
    Print information about the environment
    
    Args:
        env: The environment to inspect
    """
    print("Environment Information:")
    print("-----------------------")
    
    # Try to access observation space info
    try:
        print(f"Observation Space: {env.info.observation_space}")
        print(f"Observation Shape: {env.info.observation_space.shape}")
        print(f"Observation High: {env.info.observation_space.high}")
        print(f"Observation Low: {env.info.observation_space.low}")
    except:
        print("Could not access observation space info directly")
    
    # Try to access action space info
    try:
        print(f"\nAction Space: {env.info.action_space}")
        print(f"Action Shape: {env.info.action_space.shape}")
        print(f"Action High: {env.info.action_space.high}")
        print(f"Action Low: {env.info.action_space.low}")
    except:
        print("Could not access action space info directly")
    
    # Try alternate methods for LocoMujoco specifically
    print("\nAlternative Methods:")
    
    # Get state dimensions
    state = env.reset()
    print(f"State Type: {type(state)}")
    print(f"State Shape/Length: {np.shape(state)}")
    
    # Based on your output, we know:
    # - State shape is (36,)
    # - Action shape is (13,)
    
    # Try to inspect the environment attributes to find the simulator
    print("\nEnvironment Structure:")
    try:
        # Print all attributes of the env object to find the simulator
        env_attrs = dir(env)
        print(f"Environment attributes: {[attr for attr in env_attrs if not attr.startswith('_')]}")
        
        # Try to access the mdp attribute which might contain the simulator
        if hasattr(env, 'mdp'):
            mdp_attrs = dir(env.mdp)
            print(f"MDP attributes: {[attr for attr in mdp_attrs if not attr.startswith('_')]}")
    except Exception as e:
        print(f"Error inspecting environment: {e}")
        
    # Print state interpretation guidance
    print("\nState Vector Interpretation (Based on Standard MuJoCo):")
    print("Positions 0-2: Likely COM position (x, y, z)")
    print("Positions 3-6: Likely quaternion orientation")
    print("Positions 7-17: Likely joint angles")
    print("Positions 18-20: Likely COM velocity (x, y, z)")
    print("Positions 21-35: Likely joint velocities and other state information")
    
    # Print environment information that we know
    print("\nEnvironment Info from Output:")
    print("- Observation space shape: (36,)")
    print("- Action space shape: (13,)")
    print("- Actions are in range [-1, 1]")

def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps") 
    else: 
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.real"
    env = LocoEnv.make(env_id, use_box_feet=True)

    # Print environment information
    inspect_environment_info(env)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "real_180.msh")
    agent = Agent.load(agent_file_path)

    # Load your trained PPO policy
    state_dim = 22  # Number of features in the substate
    action_dim = 1  # Number of actions
    policy = PolicyNet(state_dim, action_dim).to(device)
    policy_load_path = os.path.join(os.path.dirname(__file__), "best_policy.pth")
    policy.load_state_dict(torch.load(policy_load_path, weights_only=True))  # Added weights_only=True to fix warning
    policy.eval()
    print(f"Model weights loaded from {policy_load_path}")

    # Try to add environment debugging
    print("\nAdditional Environment Debugging:")
    try:
        # Try to get the dt (simulation timestep)
        if hasattr(env, 'dt'):
            print(f"Environment dt: {env.dt}")
        elif hasattr(env, 'info') and hasattr(env.info, 'dt'):
            print(f"Environment dt: {env.info.dt}")
        else:
            print("Could not find dt, assuming 0.01 for calculations")
            env.dt = 0.01  # Default assumption for calculations
            
        # Try to access the simulator directly
        if hasattr(env, 'sim'):
            print("Found sim attribute directly on env")
        elif hasattr(env, 'mdp') and hasattr(env.mdp, 'sim'):
            print("Found sim attribute on env.mdp")
            env.sim = env.mdp.sim  # For convenience
            
        # Get one state and action to analyze structure
        state = env.reset()
        dummy_action = agent.draw_action(state)
        print(f"Example state slice (first 6): {state[:6]}")
        print(f"Example action: {dummy_action}")
    except Exception as e:
        print(f"Debug error: {e}")

    try:
        # Collect data for PPO model with error handling
        print("Collecting data for PPO model...")

        if hasattr(env, 'seed'):
            env.seed(42)  # Use a fixed seed

        # Save the initial state for reference
        initial_state = env.reset()
        ppo_data = collect_gait_data(policy, env, agent, num_steps=1000, device=device)
        
        # Collect data for expert model (using imitation learning - expert agent only)
        print("Collecting data for expert model (imitation learning)...")
        # Reset the environment to ensure fair comparison
        if hasattr(env, 'seed'):
            env.seed(42)  # Use the same fixed seed
        # Reset to match the initial state
        initial_state_expert = env.reset()
        expert_data = collect_gait_data(None, env, agent, num_steps=1000, device=device)
        
        # Analyze gait cycles
        print("Analyzing PPO gait cycles...")
        ppo_gait_info = analyze_gait_cycles(ppo_data)
        
        print("Analyzing expert gait cycles...")
        expert_gait_info = analyze_gait_cycles(expert_data)
        
        # Print gait cycle information
        print("\nPPO Model Gait Information:")
        if 'avg_cycle_time' in ppo_gait_info:
            print(f"Average Cycle Time: {ppo_gait_info['avg_cycle_time']:.4f} seconds")
            print(f"Average Cycle Length: {ppo_gait_info['avg_cycle_length']:.4f} meters")
            print(f"Average Velocity: {ppo_gait_info['avg_velocity']:.4f} m/s")
            print(f"Number of Cycles Detected: {len(ppo_gait_info['cycle_times'])}")
        else:
            print(ppo_gait_info['message'])
        
        print("\nExpert Model Gait Information:")
        if 'avg_cycle_time' in expert_gait_info:
            print(f"Average Cycle Time: {expert_gait_info['avg_cycle_time']:.4f} seconds")
            print(f"Average Cycle Length: {expert_gait_info['avg_cycle_length']:.4f} meters")
            print(f"Average Velocity: {expert_gait_info['avg_velocity']:.4f} m/s")
            print(f"Number of Cycles Detected: {len(expert_gait_info['cycle_times'])}")
        else:
            print(expert_gait_info['message'])
        
        # Plot comparison
        print("\nPlotting gait comparison...")
        plot_gait_comparison(ppo_data, expert_data)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        
        # Try a simplified approach with just testing the models
        print("\nFalling back to simplified approach...")
        
        # Define a simplified test function to observe behavior
        def simplified_test(policy, env, agent, steps=200, use_policy=True):
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
        
        # Run simplified tests
        print("Testing PPO model...")
        ppo_positions, ppo_velocities = simplified_test(policy, env, agent, steps=500, use_policy=True)
        
        env = LocoEnv.make(env_id, use_box_feet=True)
        print("Testing expert model...")
        expert_positions, expert_velocities = simplified_test(policy, env, agent, steps=500, use_policy=False)

        
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