import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import time
import argparse

#test python test_lunar_lander.py --model_path path/to/your/model.pt


# Policy Network for continuous actions
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 32)
        
        # Mean output for continuous actions
        self.mean = nn.Linear(32, action_dim)
        # Log standard deviation network
        self.logstd = nn.Linear(32, action_dim)
        
        self.action_dim = action_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = F.tanh(self.mean(x))
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

# Test the trained policy
def test_policy(policy, env, num_episodes=5, device="cpu", render_delay=0.01):
    policy.eval()
    test_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            env.render()
            time.sleep(render_delay)
            
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test a trained PPO model on LunarLander')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to test')
    parser.add_argument('--render_delay', type=float, default=0.01, help='Delay between renders in seconds')
    args = parser.parse_args()
    
    # Set the device
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else 
                     "cpu")
    print(f"Using device: {device}")
    
    # Initialize the environment with rendering
    env = gym.make("LunarLander-v3", continuous=True, render_mode="human")
    
    # Define the state and action dimensions
    state_dim = env.observation_space.shape[0]  # 8 for LunarLander
    action_dim = env.action_space.shape[0]      # 2 for LunarLander continuous
    
    # Initialize the policy network
    policy = PolicyNet(state_dim, action_dim)
    
    # Load the trained model
    try:
        policy.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Move the model to the device
    policy = policy.to(device)
    
    # Test the trained policy
    print("\n--- Running test episodes ---")
    test_rewards = test_policy(
        policy=policy, 
        env=env, 
        num_episodes=args.num_episodes, 
        device=device,
        render_delay=args.render_delay
    )
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()
