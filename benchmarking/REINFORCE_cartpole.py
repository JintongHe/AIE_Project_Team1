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

def reinforce_train(policy, value_function, env, num_epochs, episode_steps, alpha, gamma, device):
    # Move the policy and value function to the specified device
    policy.to(device)
    value_function.to(device)

    # Initialize the optimizer
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)
    value_optimizer = torch.optim.Adam(value_function.parameters(), lr=alpha)
    
    # Initialize the loss function
    loss_fn = nn.MSELoss()
    
    epoch_rewards = []
    for t in range(num_epochs):
        # Initialize the lists to store the states, actions, and rewards
        states = []
        actions = []
        rewards = []
        action_log_probs = []
        
        # Reset the environment
        state, _ = env.reset()
        
        for _ in range(episode_steps):
            # Get the action probabilities from the policy
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action = policy.sample_action(state_tensor)
            
            # Take the action in the environment - convert to int for discrete action
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            
            # Store the state, action, and reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            action_log_probs.append(policy.log_prob(state_tensor, action))
            
            # Update the state
            state = next_state
            
            if terminated or truncated:
                break
        
        # Compute the discounted rewards
        returns = compute_returns(rewards, gamma)
        
        # Convert the states and returns to tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Compute the value function predictions
        value_predictions = value_function(states_tensor).squeeze()
        
        # Compute the advantages
        advantages = returns_tensor - value_predictions.detach()
        
        # Compute the policy loss
        policy_loss = -(torch.stack(action_log_probs) * advantages).mean()

        # Zero the gradients
        policy_optimizer.zero_grad()
        
        # Compute the gradients
        policy_loss.backward()
        
        # Update the policy and value function
        policy_optimizer.step()
        
        # Compute the value function loss
        value_loss = loss_fn(value_predictions, returns_tensor)
        
        # Zero the gradients
        value_optimizer.zero_grad()
        
        # Compute the gradients
        value_loss.backward()
        
        # Update the policy and value function
        value_optimizer.step()
        
        # Compute the epoch reward
        epoch_reward = sum(rewards)
        epoch_rewards.append(epoch_reward)
        
        # Print the average reward per epoch
        print(f'Epoch {t+1}, Reward: {epoch_reward}')
    return policy.state_dict().copy()

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

# Initialize the training environment
env = gym.make('CartPole-v1')

# Define the state and action dimensions
state_dim = env.observation_space.shape[0]  # 4 for CartPole
action_dim = env.action_space.n  # 2 for CartPole

# Initialize the policy and value networks
policy = PolicyNet(state_dim, action_dim)
value_function = ValueNet(state_dim)

# Set the device - you can use 'cuda', 'mps', or 'cpu' depending on what's available
# device = torch.device("cuda" if torch.cuda.is_available() else 
#                      "mps" if torch.backends.mps.is_available() else 
#                      "cpu")

device = 'cpu'

# Train the policy
final_policy = reinforce_train(policy, value_function, env, num_epochs=400, episode_steps=500, alpha=0.01, gamma=0.99, device=device)

# Save the best policy to a file
model_save_path = "official_cartpole.pth"
torch.save(final_policy, model_save_path)
print(f"Best model saved to {model_save_path}")

# Create a test environment with rendering enabled
test_env = gym.make('CartPole-v1', render_mode="human")

# Run 5 test episodes
print("\n--- Running 5 test episodes with rendering ---")
test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)

# Close the environments
env.close()
test_env.close()


