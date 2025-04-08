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

#Things to do for full body control:
#1. Change input to 36 states
#2. Change output to 13 actions
#3. Change the reward function to be the exp of MSE between the expert and policy actions

def ankle_reward(state, ankle_state, action):

    pass


# Policy Network for continuous actions - adjusted for BipedalWalker
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)  # Increased network size
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        
        # Mean output for continuous actions
        self.mean = nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = nn.Linear(32, action_dim)
        
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
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
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
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
def ppo_train(policy, value_function, env, agent, state_dim, action_dim, num_epochs=200, steps_per_epoch=4000,
              gamma=0.99, lam=0.95, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
              train_pi_iters=80, train_v_iters=80, target_kl=0.01, max_ep_len=2000, batch_size=64,
              device="mps"):
    """
    PPO-Clip algorithm implementation for BipedalWalker
    """
    # Set up optimizers
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=pi_lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(value_function.parameters(), lr=vf_lr)
    
    # Set device
    policy.to(device)
    value_function.to(device)

    # Create a copy of the policy to represent the policy from the previous epoch
    prev_epoch_policy = PolicyNet(state_dim, action_dim).to(device)
    prev_epoch_policy.load_state_dict(policy.state_dict())
    
    # Best policy tracking
    best_reward = float('-inf')
    best_policy_state = None
    
    # Main training loop
    episode_rewards = []
    episode_lengths = []
    
    # Create buffer
    buffer = PPOBuffer(state_dim, action_dim, steps_per_epoch, gamma, lam)
    
    # Initialize environment state
    state = env.reset()

    episode_reward = 0
    episode_length = 0
    
    step = 1
    for t in range(num_epochs * steps_per_epoch):
        # Get action, value, and log probability from current policy
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
            action, log_prob = policy.sample_action(state_tensor)
            value = value_function(state_tensor)
            
        # Take action in environment
        expert_action = agent.draw_action(state)
        next_state, reward, done, _ = env.step(action.cpu().detach().numpy())
        
        # Calculate reward as MSE between expert and policy actions
        reward = torch.exp(-F.mse_loss(torch.tensor(expert_action).to(device), torch.tensor(action).to(device)))+1
        
        # Store trajectory in buffer
        buffer.store(state, action, reward, value.item(), log_prob.item(), done)
        
        # Update state and counters
        state = next_state
        episode_reward += reward
        episode_length += 1
        
        # End of trajectory handling
        timeout = episode_length == max_ep_len
        epoch_ended = (t + 1) % steps_per_epoch == 0
        step+=1
        
        if done or timeout or epoch_ended:
            step = 1
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
                state = env.reset()
                ankle_state = get_right_ankle_substate(state)
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
        
            # Update policy using the PPO-Clip objective
            for i in range(train_pi_iters):
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
                
                # # Add entropy bonus for exploration
                # entropy_loss = -0.01 * dist.entropy().sum(dim=-1).mean()
                # total_policy_loss = policy_loss + entropy_loss
                
                # Update policy
                policy_loss.backward()
                policy_optimizer.step()
                
                # Calculate KL divergence and check for early stopping
                kl = compute_kl(policy, all_states, old_mean, old_std)
                if kl > 1.5 * target_kl:
                    print(f"Early stopping at step {i+1} due to reaching max KL {kl:.3f}")
                    break
            
            # Update value function
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
        episode_reward = 0

        for step in range(max_steps):
            env.render()  # Render the environment

            # Get the best action from the policy
            with torch.no_grad():
                state_tensor = torch.as_tensor(state, dtype=torch.float32).to(device)
                action = policy.get_best_action(state_tensor)
                action = action.cpu().numpy()
    
 

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)

            # Update state and reward
            state = next_state
            ankle_state = get_right_ankle_substate(state)
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Total Reward: {episode_reward:.2f}")

    env.close()
    return episode_rewards


def main():
    #Initialize device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps") 
    # else: 
    #     device = torch.device("cpu")
    device = 'cpu'

    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    env = LocoEnv.make(env_id, use_box_feet=True)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_88_original.msh")
    agent = Agent.load(agent_file_path)

    #Initialize the model
    state_dim = 36  # Number of features in the substate
    value_dim = 36
    action_dim = 13  # Number of actions
    hidden_dim = 64  # Number of hidden units
    policy = PolicyNet(state_dim, action_dim).to(device)
    value_function = ValueNet(state_dim).to(device)

    # Train using PPO
    best_policy_state, best_reward = ppo_train(
        policy=policy,
        value_function=value_function,
        env=env,
        agent=agent,
        state_dim=state_dim,
        action_dim=action_dim,
        num_epochs=3000,           # Increased for BipedalWalker
        steps_per_epoch=4000,     # Steps per epoch
        gamma=0.99,               # Discount factor
        lam=0.95,                 # GAE-Lambda parameter
        clip_ratio=0.05,           # PPO clip ratio
        pi_lr=5e-5,               # Policy learning rate
        vf_lr=1e-3,               # Value function learning rate
        train_pi_iters=80,        # Policy optimization iterations
        train_v_iters=80,         # Value function iterations
        target_kl=0.1,           # Target KL divergence for early stopping
        max_ep_len=800,          # Maximum episode length for BipedalWalker
        batch_size=64,            # Batch size for training
        device=device
    )

    # Save the best policy state
    torch.save(best_policy_state, 'full_body_IL.pth')
    # After training, load the best policy
    policy.load_state_dict(best_policy_state)

    # Run the test functio/
    print("\nTesting the best policy:")
    test_rewards = test_best_policy(policy, env, agent, num_episodes=10, device=device)

    # Print summary statistics
    print(f"\nTest Results:")
    print(f"Average Reward: {np.mean(test_rewards):.2f}")
    print(f"Standard Deviation: {np.std(test_rewards):.2f}")
    print(f"Max Reward: {np.max(test_rewards):.2f}")
    print(f"Min Reward: {np.min(test_rewards):.2f}")
    #env.close()

if __name__ == '__main__':
    main()

#update old log probs to be more recent log probs, currently the data used can be from the beginning. 
#do random sample batches, look into different batch strategies
#test performance with reward as MSE between expert and policy actions
#look into KL divergence and how it is calculated, update the early stopping condition to be more accurate