import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
import gymnasium as gym
import time
import numpy as np

# Network architecture for discrete actions
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.action_head = nn.Linear(32, action_dim)
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return action_probs

    def sample_action(self, state, epsilon=0.1):
        """
        Epsilon-greedy action selection for training
        """
        action_probs = self.forward(state)
        return Categorical(action_probs).sample()

    def get_best_action(self, state):
        """
        Deterministic action selection for testing (always choose argmax)
        """
        action_probs = self.forward(state)
        return torch.argmax(action_probs)

    def log_prob(self, state, action):
        """Calculate log probability for policy gradient updates"""
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = action.to(action_probs.device)
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


def load_best_model(policy, value_function, device, path='best_model.pth'):
    """Load the best performing model from saved checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    value_function.load_state_dict(checkpoint['value_state_dict'])
    
    print(f"\nLoaded best model from epoch {checkpoint['epoch']} with reward {checkpoint['reward']}")
    return policy, value_function


def reinforce_train(policy, value_function, env, num_epochs, episode_steps, alpha, gamma, device, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
    # Move models to device
    policy.to(device)
    value_function.to(device)

    # Initialize optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)
    value_optimizer = torch.optim.Adam(value_function.parameters(), lr=alpha)
    
    # Initialize loss function
    loss_fn = nn.MSELoss()
    
    # Initialize epsilon for exploration
    epsilon = epsilon_start
    
    # Track best reward and model
    best_reward = float('-inf')
    best_policy_state = None
    
    epoch_rewards = []
    for t in range(num_epochs):
        # Initialize episode storage
        states, actions, rewards, action_log_probs = [], [], [], []
        
        # Reset environment
        state, _ = env.reset()
        
        for _ in range(episode_steps):
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Use epsilon-greedy for action selection during training
            action = policy.sample_action(state_tensor, epsilon)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            # Store trajectory information
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            action_log_probs.append(policy.log_prob(state_tensor, action.to(device)))
            
            # Update state
            state = next_state
            
            if terminated or truncated:
                break
        
        # Decay epsilon for next episode
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Compute returns and convert to tensors
        returns = compute_returns(rewards, gamma)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Value function predictions and advantages
        value_predictions = value_function(states_tensor).squeeze()
        advantages = returns_tensor - value_predictions.detach()
        
        # Policy loss and optimization
        policy_loss = -(torch.stack(action_log_probs) * advantages).mean()
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # Value function loss and optimization
        value_loss = loss_fn(value_predictions, returns_tensor)
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()
        
        # Track and report progress
        epoch_reward = sum(rewards)
        epoch_rewards.append(epoch_reward)
        
        
        print(f'Epoch {t+1}, Reward: {epoch_reward}, Epsilon: {epsilon:.4f}')
    
    # Return the best model state and its reward
    return policy.state_dict(), epoch_reward


def test_policy(policy, env, num_episodes, device, render_delay=2):
    """Test policy using deterministic (argmax) action selection"""
    policy.eval()  # Set to evaluation mode
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
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Get best action deterministically (no exploration)
            with torch.no_grad():
                action = policy.get_best_action(state_tensor)
            
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


# Initialize the training environment
env = gym.make("LunarLander-v3", continuous=False)

# Define the state and action dimensions
state_dim = env.observation_space.shape[0]  # 8 for Lunar Lander
action_dim = env.action_space.n  # 4 for Lunar Lander

# Initialize the policy and value networks
policy = PolicyNet(state_dim, action_dim)
value_function = ValueNet(state_dim)

# Set the device
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else 
                     "cpu")

# Train the policy with epsilon-greedy exploration
best_policy_state, best_reward = reinforce_train(
    policy, 
    value_function, 
    env, 
    num_epochs=500, 
    episode_steps=500,  # Increased steps for Lunar Lander
    alpha=0.01,  # Adjusted learning rate
    gamma=0.99,  # Adjusted discount factor
    device=device,
    epsilon_start=1.0,  # Start with full exploration
    epsilon_end=0.01,   # Minimum exploration rate
    epsilon_decay=0.99 # Decay rate per episode
)

# Create a test environment with rendering enabled
test_env = gym.make("LunarLander-v3", continuous=False, render_mode="human")

# Load the best model for testing
policy, value_function = load_best_model(policy, value_function, device)

# Run 5 test episodes with the best model
print("\n--- Running 5 test episodes with the best model ---")
test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)

# Close the environments
env.close()
test_env.close()
