import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
import scipy.signal
import time

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)  # Limit log_std to prevent instability
        return mean, log_std
    
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q(x)
        return q_value
    
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Storage
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        
    def store(self, state, action, reward, next_state, done):
        """Store a transition in the buffer"""
        # Convert tensors to numpy arrays if needed
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size, device=None):
        """Sample a batch of transitions"""
        idx = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'states': torch.FloatTensor(self.states[idx]),
            'actions': torch.FloatTensor(self.actions[idx]),
            'rewards': torch.FloatTensor(self.rewards[idx]),
            'next_states': torch.FloatTensor(self.next_states[idx]),
            'dones': torch.FloatTensor(self.dones[idx])
        }
        
        if device is not None:
            batch = {k: v.to(device) for k, v in batch.items()}
            
        return batch
    
    def __len__(self):
        return self.size

def train_sac(env, policy_net, q_net1, q_net2, target_q_net1, target_q_net2, 
              policy_optimizer, q1_optimizer, q2_optimizer, 
              replay_buffer, device, 
              max_episodes=1000, max_steps=1000, 
              batch_size=256, updates_per_step=1,
              start_steps=10000, alpha=0.2):
    # Training loop
    total_steps = 0
    for episode in range(max_episodes):
        episode_reward = 0
        episode_steps = 0
        done = False
        
        # Reset the environment
        state, _ = env.reset()

        while not done and episode_steps < max_steps:
            #Select random action early for exploration and then use the policy
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    mean, log_std = policy_net(state_tensor)
                    std = log_std.exp()
                    normal = Normal(mean, std)
                    
                    # Reparameterization trick
                    z = normal.rsample()
                    
                    # Squash to [-1, 1]
                    action = torch.tanh(z)
                    action = action.cpu().numpy().flatten()


# Main function
def main():
    print(f"Torch version: {torch.__version__}")
    # Initialize the environment
    env = gym.make("BipedalWalker-v3", hardcore=False)
    
    # Define the state and action dimensions
    state_dim = env.observation_space.shape[0]  # 24 for BipedalWalker
    action_dim = env.action_space.shape[0]      # 4 for BipedalWalker
    
    # Network hyperparameters
    hidden_dim = 256
    lr = 3e-4
    tau = 0.005  # For soft update of target networks
    gamma = 0.99  # Discount factor
    alpha = 0.2  # Temperature parameter for entropy

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize networks
    policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Initialize twin Q-networks
    q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Initialize target Q-networks
    target_q_net1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
    target_q_net2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

    # Copy parameters to target networks (hard update)
    for target_param, param in zip(target_q_net1.parameters(), q_net1.parameters()):
        target_param.data.copy_(param.data)
    for target_param, param in zip(target_q_net2.parameters(), q_net2.parameters()):
        target_param.data.copy_(param.data)

    # Initialize optimizers
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    q1_optimizer = torch.optim.Adam(q_net1.parameters(), lr=lr)
    q2_optimizer = torch.optim.Adam(q_net2.parameters(), lr=lr)

    replay_buffer = ReplayBuffer(state_dim, action_dim)

    
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
