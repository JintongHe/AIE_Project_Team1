import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import scipy.signal
import time

# Policy Network for continuous actions - adjusted for BipedalWalker
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        
        self.fc2 = nn.Linear(128, 64)
        
        self.fc3 = nn.Linear(64, 32)
        
        # Mean output for continuous actions
        self.mean = nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = nn.Linear(32, action_dim)
        
        self.action_dim = action_dim

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        x = F.relu(x)
        
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
        self.fc1 = nn.Linear(state_dim, 128)
        self.ln1 = nn.LayerNorm(128)  # Layer normalization after first linear layer
        
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)  # Layer normalization after second linear layer
        
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)  # Apply normalization before activation
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)  # Apply normalization before activation
        x = F.relu(x)
        
        return self.fc3(x).squeeze(-1)

# PPO Buffer for storing trajectories
class PPOBuffer:
    def __init__(self, state_dim, action_dim, size, gamma=0.99, lam=0.95):
        self.states = np.zeros((size, state_dim), dtype=np.float32)
        self.actions = np.zeros((size, action_dim), dtype=np.float32)
        self.advantages = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        
        self.gamma = gamma
        self.lam = lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, state, action, reward, value, log_prob, done):
        """Store one transition in the buffer"""
        assert self.ptr < self.max_size
        
        # Convert tensors to CPU before storing in NumPy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        self.ptr += 1
        
    def finish_path(self, last_value=0):
        """Calculate advantages and returns using GAE-Lambda"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        dones = np.append(self.dones[path_slice], 0)
        
        # GAE-Lambda advantage calculation
        deltas = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1]) - values[:-1]
        self.advantages[path_slice] = self._discount_cumsum(deltas, self.gamma * self.lam)
        
        # Compute returns for value function targets
        self.returns[path_slice] = self._discount_cumsum(rewards[:-1], self.gamma)
        
        self.path_start_idx = self.ptr
        
    def _discount_cumsum(self, x, discount):
        """Calculate discounted cumulative sum (used for returns and GAE)"""
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    
    def get(self):
        """Get all data from the buffer and normalize advantages"""
        assert self.ptr == self.max_size  # Buffer must be full before we can use it
        self.ptr, self.path_start_idx = 0, 0
        
        # Normalize advantages
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages) + 1e-8
        self.advantages = (self.advantages - adv_mean) / adv_std
        
        data = dict(
            states=self.states,
            actions=self.actions,
            returns=self.returns,
            advantages=self.advantages,
            log_probs=self.log_probs
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
    
    def sample_batch(self, batch_size):
        """Sample a random batch of data from the buffer"""
        indices = np.random.choice(self.max_size, batch_size, replace=False)
        batch = dict(
            states=self.states[indices],
            actions=self.actions[indices],
            returns=self.returns[indices],
            advantages=self.advantages[indices],
            log_probs=self.log_probs[indices]
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


# Compute KL divergence between old and new policy distributions
def compute_kl(policy, states, old_mean, old_std):
    new_mean, new_std = policy.forward(states)
    old_dist = Normal(old_mean, old_std)
    new_dist = Normal(new_mean, new_std)
    kl = torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1).mean().item()
    return kl

# PPO main training function - adjusted for BipedalWalker
def ppo_train(policy, value_function, env, num_epochs=200, steps_per_epoch=4000,
              gamma=0.99, lam=0.95, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
              train_pi_iters=80, train_v_iters=80, target_kl=0.01, max_ep_len=2000,
              batch_size=64, device="cpu"):
    """
    PPO-Clip algorithm implementation for BipedalWalker with mini-batch updates
    """
    # Set up optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=pi_lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(value_function.parameters(), lr=vf_lr)
    
    # Set device
    policy.to(device)
    value_function.to(device)

    # Create a copy of the policy to represent the policy from the previous epoch
    prev_epoch_policy = PolicyNet(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    prev_epoch_policy.load_state_dict(policy.state_dict())
    
    # Setup state/action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Best policy tracking
    best_reward = float('-inf')
    best_policy_state = None
    
    # Main training loop
    episode_rewards = []
    episode_lengths = []
    
    # Create buffer
    buffer = PPOBuffer(state_dim, action_dim, steps_per_epoch, gamma, lam)
    
    # Initialize environment state
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for t in range(num_epochs * steps_per_epoch):
        # Get action, value, and log probability from current policy
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
            action, log_prob = policy.sample_action(state_tensor)
            value = value_function(state_tensor)
            
        # Take action in environment
        action_numpy = action.cpu().detach().numpy()
        next_state, reward, terminated, truncated, _ = env.step(action_numpy)
        done = terminated or truncated
        
        # Store trajectory in buffer
        buffer.store(state, action_numpy, reward, value.item(), log_prob.item(), done)
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # End of trajectory handling
        timeout = episode_length == max_ep_len
        epoch_ended = (t + 1) % steps_per_epoch == 0
        
        if done or timeout or epoch_ended:
            if timeout or epoch_ended:
                # If trajectory didn't reach terminal state, bootstrap value
                with torch.no_grad():
                    state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
                    last_value = value_function(state_tensor).item()
            else:
                last_value = 0
            
            # Finish the current trajectory
            buffer.finish_path(last_value)
            
            if done or timeout:
                # Log episode stats
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                # Reset for new episode
                state, _ = env.reset()
                episode_reward = 0
                episode_length = 0
        
        # End of epoch handling - update policy using collected data
        if epoch_ended:
            epoch = (t + 1) // steps_per_epoch
            
            # Get the data from the buffer
            data = buffer.get()
            
            # Store old policy parameters for KL calculation
            with torch.no_grad():
                all_states = torch.as_tensor(buffer.states, dtype=torch.float32).to(device)
                old_mean, old_std = policy.forward(all_states)
            
            # Update policy using mini-batches
            for i in range(train_pi_iters):
                # Sample a fresh batch
                batch = buffer.sample_batch(batch_size)
                states = batch['states'].to(device)
                actions = batch['actions'].to(device)
                advantages = batch['advantages'].to(device)
                # Compute old log probabilities using prev_epoch_policy
                with torch.no_grad():
                    old_dist = prev_epoch_policy.get_distribution(states)
                    old_log_probs = old_dist.log_prob(actions).sum(dim=-1)
                
                policy_optimizer.zero_grad()
                
                # Get current distribution and log probabilities
                dist = policy.get_distribution(states)
                curr_log_probs = dist.log_prob(actions).sum(dim=-1)
                
                # Calculate policy ratio (π_θ / π_θ_old)
                ratio = torch.exp(curr_log_probs - old_log_probs)
                
                # Calculate PPO-Clip objective
                clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
                policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
                
                # Add entropy bonus for exploration
                # entropy_loss = -0.01 * dist.entropy().sum(dim=-1).mean()
                # total_policy_loss = policy_loss + entropy_loss
                
                # Update policy
                policy_loss.backward()
                policy_optimizer.step()
                
                # Check KL divergence every 20 iterations to save computation
                if (i+1) % 20 == 0:
                    kl = compute_kl(policy, all_states, old_mean, old_std)
                    if kl > 1.5 * target_kl:
                        print(f"Early stopping at step {i+1} due to reaching max KL {kl:.3f}")
                        break
            
            # Update value function with mini-batches too
            for _ in range(train_v_iters):
                # Sample a fresh batch
                batch = buffer.sample_batch(batch_size)
                states = batch['states'].to(device)
                returns = batch['returns'].to(device)
                
                value_optimizer.zero_grad()
                
                # Calculate value loss
                values = value_function(states)
                value_loss = ((values - returns) ** 2).mean()
                
                # Update value function
                value_loss.backward()
                value_optimizer.step()
            
            # Print progress
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
                print(f"Epoch {epoch}/{num_epochs} | Mean Reward: {mean_reward:.2f}")
                
                # Save best policy
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_policy_state = policy.state_dict().copy()
            prev_epoch_policy.load_state_dict(policy.state_dict())
    
    return best_policy_state, best_reward


# Test the trained policy
def test_policy(policy, env, num_episodes=5, device="mps"):
    policy.eval()
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                action = policy.get_best_action(state_tensor)
            
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        
        test_rewards.append(episode_reward)
        print(f"Test Episode {episode+1}: Reward = {episode_reward}")
    
    avg_reward = sum(test_rewards) / num_episodes
    print(f"\nAverage Test Reward: {avg_reward:.2f}")
    
    return test_rewards

# Main function
def main():
    print(f"Torch version: {torch.__version__}")
    # Initialize the environment
    env = gym.make("BipedalWalker-v3", hardcore=False)
    
    # Define the state and action dimensions
    state_dim = env.observation_space.shape[0]  # 24 for BipedalWalker
    action_dim = env.action_space.shape[0]      # 4 for BipedalWalker
    
    # Initialize the policy and value networks
    policy = PolicyNet(state_dim, action_dim)
    value_function = ValueNet(state_dim)
    
    # Set the device
    device = torch.device( 
                     "cuda" if torch.cuda.is_available() else 
                     "cpu") 
    #device = "mps" if torch.backends.mps.is_available() else device
    
    # Train using PPO
    best_policy_state, best_reward = ppo_train(
        policy=policy,
        value_function=value_function,
        env=env,
        num_epochs=1000,           # Increased for BipedalWalker
        steps_per_epoch=4000,     # Steps per epoch
        gamma=0.99,               # Discount factor
        lam=0.95,                 # GAE-Lambda parameter
        clip_ratio=0.05,           # PPO clip ratio
        pi_lr=3e-4,               # Policy learning rate
        vf_lr=1e-3,               # Value function learning rate
        train_pi_iters=80,        # Policy optimization iterations
        train_v_iters=80,         # Value function iterations
        target_kl=0.1,           # Target KL divergence for early stopping
        max_ep_len=2000,  
        batch_size=64,        # Maximum episode length for BipedalWalker
        device=device
    )
    
    # Load the best policy for testing
    policy.load_state_dict(best_policy_state)

    # Save the best policy to a file
    model_save_path = "test_hardcore.pth"
    torch.save(best_policy_state, model_save_path)
    print(f"Best model saved to {model_save_path} with reward {best_reward:.2f}")
    
    # Create a test environment with rendering
    test_env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="human")
    
    # Test the trained policy
    print("\n--- Running test episodes with the best model ---")
    test_rewards = test_policy(policy, test_env, num_episodes=5, device=device)
    
    # Close environments
    env.close()
    test_env.close()

if __name__ == "__main__":
    main()
