import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import gymnasium as gym
import time
import numpy as np

# Network architecture for continuous actions with multivariate Gaussian
# Network architecture for continuous actions with multivariate Gaussian
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        
        # Mean output for continuous actions
        self.mean = nn.Linear(32, action_dim)

        #standard deviation network
        self.logstd = nn.Linear(32, action_dim)
        
        # Initialize the mean layer with small weights for stability
        # nn.init.uniform_(self.mean.weight, -3e-3, 3e-3)
        # nn.init.uniform_(self.mean.bias, -3e-3, 3e-3)
        
        self.action_dim = action_dim

        # Option 1: State-independent log standard deviations
        # self.log_std = -2.3    
        # self.std = np.exp(self.log_std)    
        # Option 2: State-dependent log standard deviations
        # self.log_std_network = nn.Linear(32, action_dim)
        # nn.init.uniform_(self.log_std_network.weight, -3e-3, 3e-3)
        # nn.init.uniform_(self.log_std_network.bias, -3e-3, 3e-3)
        

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = F.tanh(self.mean(x))

        std = torch.exp(torch.clamp(self.logstd(x), -9, 0.5))

        
        
        
        #log_std = torch.clamp(log_std, -100, 0.0)  # Constrain log_std for stability
    
        return mean, std
    
    def sample_action(self, state, epsilon):
        """Sample actions from multivariate Gaussian distribution for training"""
    
        mean, std = self.forward(state)
            
        action = torch.distributions.Normal(mean, std).sample()
            
        return action
    
    def get_best_action(self, state):
        """Deterministic action selection for testing (always choose mean)"""
        mean, std = self.forward(state)
        
        # Scale from [-1, 1] to [-2, 2] for Pendulum environment
        # scaled_action = 2.0 * action
        
        return mean
    
    def log_prob(self, state, action):
        """Calculate log probability using multivariate Gaussian distribution"""
        
        # Get the mean and log_std
        mean, std = self.forward(state) 

        # print("mean:", mean)
        # print("\n std:", std)
        log_likelihood = torch.distributions.Normal(mean, std).log_prob(action).sum(dim=-1)
        
        # # For diagonal Gaussian, we can use Normal distribution
        # # since the dimensions are independent
        # dist = Normal(mean, std)
        
        # # Calculate log probability
        # log_prob = dist.log_prob(action)
        
        # Account for the tanh transformation (determinant of Jacobian)
        #log_det_jacobian = -torch.sum(torch.log(1 - action_internal.pow(2) + 1e-6), dim=-1)
        
        # Total log probability
        return log_likelihood #- log_det_jacobian

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

def normalize_advantages(advantages, epsilon=1e-8):
    """Normalize advantages to have mean=0, std=1"""
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)



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
            action_numpy = action.cpu().detach().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action_numpy)
            
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
        
        # Compute returns and convert to tensors
        returns = compute_returns(rewards, gamma)
        states_tensor = torch.tensor(states, dtype=torch.float32).to(device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Value function predictions and advantages
        value_predictions = value_function(states_tensor).squeeze()
        advantages = returns_tensor - value_predictions.detach()
        # print("Returns: ", returns)
        # print("Value Predictions: ", value_predictions)
        # print("Advantages: ", advantages)
        
        # Normalize advantages for training stability
        normalized_advantages = normalize_advantages(advantages)

# Policy loss and optimization (use normalized advantages)
        policy_loss = -(torch.stack(action_log_probs) * normalized_advantages).mean()
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
        
        print(f'Epoch {t+1}, Reward: {epoch_reward}')
        
        # Save best model
        if epoch_reward > best_reward:
            best_reward = epoch_reward
            best_policy_state = policy.state_dict()
            # # Save model checkpoint
            # torch.save({
            #     'epoch': t + 1,
            #     'policy_state_dict': policy.state_dict(),
            #     'value_state_dict': value_function.state_dict(),
            #     'reward': epoch_reward
            # }, 'best_model.pth')
    
    return best_policy_state, best_reward


def test_policy(policy, env, num_episodes, device, render_delay=0.01):
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
            time.sleep(render_delay)  # Small delay to visualize the rendering
            
            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            # Get best action deterministically (no exploration)
            with torch.no_grad():
                action = policy.get_best_action(state_tensor)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            
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
env = gym.make("Pendulum-v1")

# Define the state and action dimensions
state_dim = env.observation_space.shape[0]  # 3 for Pendulum
action_dim = env.action_space.shape[0]      # 1 for Pendulum

# Initialize the policy and value networks
# Using the "first way" described in the query - state-independent log std
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
    num_epochs=500,           # Adjusted for Pendulum
    episode_steps=500,        # Match Pendulum's episode length
    alpha=0.0001,              # Learning rate
    gamma=0.9,               # Discount factor
    device=device,       
)

# Create a test environment with rendering enabled
test_env = gym.make("Pendulum-v3", render_mode="human")

# Load the best model for testing
policy.load_state_dict(best_policy_state)

# Run 5 test episodes with the best model
print("\n--- Running 5 test episodes with the best model ---")
test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)

# Close the environments
env.close()
test_env.close()

#add standard deviation network
#add replay buffers
