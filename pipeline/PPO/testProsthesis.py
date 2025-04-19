from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelsAndUtils import MLP, get_ankle_substate
from torch.distributions import Normal
import numpy as np
import scipy.signal
from RL import PolicyNet

#function that takes a string and outputs the integer in the 4th and 5th indexes 
def get_model_number(model_name):
    """
    Extracts the model number from the model name.
    
    Args:
    model_name (str): The name of the model
    
    Returns:
    int: The model number
    """
    return int(model_name[4:6])


# 1 Run each model in a for loop and get the average reward
def get_average_steps(policy, env, agent, num_episodes=10, max_steps=2000, device="mps", state_dim=36):
    """
    Test the best policy by running it in the environment multiple times with rendering.
    
    Args:
    env (LocoEnv): The environment
    agent (Agent): The agent
    num_episodes (int): Number of episodes to run
    max_steps (int): Maximum steps per episode
    device (str): Device to use for computation ("cpu" or "mps")
    
    Returns:
    float: Average reward over all episodes
    """
    
    total_reward = 0.0
    for episode in range(num_episodes):
        state = env.reset()
        ankle_state = get_ankle_substate(state, state_dim )
        episode_reward = 0.0
        
        for step in range(max_steps):
            # Get the best action from the policy
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(ankle_state, dtype=torch.float32).to(device)
                action = policy.get_best_action(ankle_state_tensor)
            # Combine the policy action with the expert agent's action
            body_action = agent.draw_action(state)
            body_action[7] = action.item()

            # Take a step in the environment
            next_state, reward, done, _ = env.step(body_action)
            state = next_state
            ankle_state = get_ankle_substate(state, state_dim)
            episode_reward += reward
            
            if done:
                break
        
        total_reward += episode_reward
    
    average_reward = total_reward / num_episodes
    return average_reward

def main():
    device = 'cpu'
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    env = LocoEnv.make(env_id, use_box_feet=True)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)

    models = [
        "new_12_states_survival_pros.pth",
        "new_16_states_survival_pros.pth",
        "new_22_states_survival_pros.pth",
        "new_36_states_survival_pros.pth",
    ]

    # Initialize a list to store the average rewards for each model
    average_rewards = []
    # Loop through each model and calculate the average reward
    for model in models:
        state_dim =  get_model_number(model)
        action_dim = 1
        policy = PolicyNet(state_dim, action_dim).to(device)
        policy_load_path = os.path.join(os.path.dirname(__file__), model)
        policy.load_state_dict(torch.load(policy_load_path))
        policy.eval()
        average_reward = get_average_steps(policy, env, agent, num_episodes=10, max_steps=2000, device=device, state_dim=state_dim)
        average_rewards.append(average_reward)
        print(f"Model: {model}, Average Reward: {average_reward:.2f}")
     # Plot the average rewards
    plt.figure(figsize=(10, 5))
    # Extract just the state dimensions for x-axis
    state_dims = [get_model_number(model) for model in models]
    plt.plot(state_dims, average_rewards, 'o-', color='skyblue', linewidth=2, markersize=8)
    plt.xlabel('Number of States')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Number of States')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(state_dims)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    