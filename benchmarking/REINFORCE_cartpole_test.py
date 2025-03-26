import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import gymnasium as gym
import time

# Network architecture for discrete actions
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32)        # Hidden layer
        self.action_head = nn.Linear(32, action_dim)  # Output layer for action probabilities

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

    def get_distribution(self, state):
        action_probs = self.forward(state)
        return Categorical(action_probs)

    def sample_action(self, state):
        dist = self.get_distribution(state)
        action = dist.sample()
        return action

    def log_prob(self, state, action):
        dist = self.get_distribution(state)
        return dist.log_prob(action)

class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # Input layer
        self.fc2 = nn.Linear(64, 32)         # Hidden layer
        self.fc3 = nn.Linear(32, 1)          # Output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns

def test_policy(policy, env, num_episodes, device, render_delay=0.01):
    """
    Test the trained policy for a specified number of episodes and render the environment.
    
    Args:
        policy: The trained policy network
        env: The environment to test in
        num_episodes: Number of test episodes to run
        device: Device to run computations on
        render_delay: Time delay between frames for better visualization
    
    Returns:
        List of rewards for each episode
    """
    policy.eval()  # Set the policy to evaluation mode
    test_rewards = []
    test_steps = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            # Render the environment
            env.render()
            
            # Add a small delay to make rendering visible
            time.sleep(render_delay)
            
            # Get action from policy
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():  # No need to track gradients during testing
                action = policy.sample_action(state_tensor)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            episode_reward += reward
            steps += 1
            
            # Update state
            state = next_state
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        test_steps.append(steps)
        print(f"Test Episode {episode+1}: Reward = {episode_reward}, Steps = {steps}")
    
    avg_reward = sum(test_rewards) / num_episodes
    avg_steps = sum(test_steps) / num_episodes
    print(f"\nAverage Test Reward: {avg_reward:.2f}")
    print(f"Average Test Steps: {avg_steps:.2f}")
    
    return test_rewards

test_env = gym.make("CartPole-v1", render_mode="human")
state_dim = test_env.observation_space.shape[0]  # 8 for LunarLander
action_dim = test_env.action_space.n 
policy = PolicyNet(state_dim, action_dim)
# Load the best policy for testing
policy_load_path = os.path.join(os.path.dirname(__file__), "official_cartpole.pth")
policy.load_state_dict(torch.load(policy_load_path))
policy.eval()

# device = torch.device("mps" if torch.backends.mps.is_available() else 
#                     "cuda" if torch.cuda.is_available() else 
#                     "cpu")  
device = 'cpu'
# Test the trained policy
print("\n--- Running test episodes with the best model ---")
test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)