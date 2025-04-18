from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelsAndUtils import MLP, get_right_ankle_substate, get_action_substate
from torch.distributions import Normal
import numpy as np
import scipy.signal

# Policy Network for continuous actions - adjusted for BipedalWalker
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


def test_best_policy(policy, env, agent, num_episodes=10, max_steps=2000, device="mps"):
    """
    Test the best policy by running it in the environment multiple times with rendering.
    
    Args:
    policy (PolicyNet): The trained policy network
    env (LocoEnv): The environment
    agent (Agent): The expert agent
    num_episodes (int): Number of episodes to run
    max_steps (int): Maximum number of steps per episode
    device (str): The device to run the policy on
    
    Returns:
    list: A list of total rewards for each episode
    """
    policy.eval()  # Set the policy to evaluation mode
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        ankle_state = get_right_ankle_substate(state)
        episode_reward = 0

        for step in range(max_steps):
            env.render()  # Render the environment

            # Get the best action from the policy
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(ankle_state, dtype=torch.float32).to(device)
                action = policy.get_best_action(ankle_state_tensor)

            # Combine the policy action with the expert agent's action
            body_action = agent.draw_action(state)
            body_action[7] = action.item()

            # Take a step in the environment
            next_state, reward, done, _ = env.step(body_action)

            # Update state and reward
            state = next_state
            ankle_state = get_right_ankle_substate(state)
            episode_reward += reward

            if done:
                break
        print("step", step)

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}")

    env.close()
    return episode_rewards

def main():
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps") 
    # else: 
    #     device = torch.device("cpu")
    device = 'cpu'

    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    env = LocoEnv.make(env_id, use_box_feet=True)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)
    

    #Initialize the model
    state_dim = 22  # Number of features in the substate
    action_dim = 1  # Number of actions
    policy = PolicyNet(state_dim, action_dim).to(device)
    policy_load_path = os.path.join(os.path.dirname(__file__), "new_22_states_survival_pros2.pth")
    policy.load_state_dict(torch.load(policy_load_path))
    policy.eval()
    print(f"Model weights loaded from {policy_load_path}")
    # After training, load the best policy
    # Run the test function
    print("\nTesting the best policy:")
    test_rewards = test_best_policy(policy, env, agent, num_episodes=10, device=device)

    # Print summary statistics
    print(f"\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Standard Deviation: {np.std(test_rewards):.2f}")
    print(f"Max Reward: {np.max(test_rewards):.2f}")
    print(f"Min Reward: {np.min(test_rewards):.2f}")
if __name__ == '__main__':
    main()