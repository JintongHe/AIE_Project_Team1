import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import scipy.signal
import time
import os


# Policy Network for continuous actions
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


# Value Network
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)

# Test the trained policy
def test_policy(policy, env, num_episodes=5, device="mps", render_delay=0.01):
    policy.eval()
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            env.render()
            #time.sleep(render_delay)
            
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action, _ = policy.sample_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = sum(test_rewards) / num_episodes
    print(f"\nAverage Test Reward: {avg_reward:.2f}")
    
    return test_rewards

def main():
# Create a test environment with rendering
    test_env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    state_dim = test_env.observation_space.shape[0]  # 8 for LunarLander
    action_dim = test_env.action_space.shape[0] 
    policy = PolicyNet(state_dim, action_dim)
    # Load the best policy for testing
    policy_load_path = os.path.join(os.path.dirname(__file__), "bipedal_walker_ppo_best_model.pth")
    policy.load_state_dict(torch.load(policy_load_path))
    policy.eval()

    # device = torch.device("mps" if torch.backends.mps.is_available() else 
    #                  "cuda" if torch.cuda.is_available() else 
    #                  "cpu")  
    device = 'cpu'
    # Test the trained policy
    print("\n--- Running test episodes with the best model ---")
    test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)

if __name__ == "__main__":
    main()