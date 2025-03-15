from mushroom_rl.core import Core, Agent
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#from imitation_learning.utils import get_agent
import math
import copy
from ModelsAndUtils import Actor, ESPolicy, ankle_training_reward, get_right_ankle_substate, get_action_substate


def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True, reward_type="custom", 
                       reward_params=dict(reward_callback=ankle_training_reward))

    print("Observation Space:", mdp.info.observation_space.low)
    print("Observation Space Shape:", mdp.info.observation_space.shape)

    print("Observation Variables:")
    for obs in mdp.get_all_observation_keys():
        print(obs)
    print("Action Space:", mdp.info.action_space)
    print("Action Space Shape:", mdp.info.action_space.shape)

    # Check if MPS is available
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cpu")
    
    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "best_real_agent_141.msh")
    agent = Agent.load(agent_file_path)

    # Initialize the ES policy
    input_dim = 22  # Number of features in the substate
    output_dim = 1  # Number of actions for ankle
    max_action = 0.4
    policy = ESPolicy(input_dim, output_dim, max_action, device)
    
    # ES hyperparameters from Algorithm 1
    alpha = 0.01                # Learning rate α 
    initial_sigma = 0.2        # Initial noise standard deviation σ
    min_sigma = 0.000002           # Minimum sigma (0.5% of max_action)
    sigma = initial_sigma       # Current sigma value
    n = 50                      # Population size n
    
    # Parameters for adaptive sigma
    best_max_reward_sigma = 0.0
    best_max_reward = 0.0
    sigma_decay_factor = 0.5   # How much to decrease sigma when reward improves
    improvement_threshold = 80 # Minimum improvement to trigger sigma decrease
    amount_improved = 0.0
    epochs_no_improve = 0.0
    
    epoch_rewards = []
    sigma_values = []           # To track sigma changes
    num_epochs = 500
    episode_steps = 1000
    
    # Get initial policy parameters θ₀
    theta = policy.get_params()
    
    for t in range(num_epochs):
        # Sample ε₁, ..., εₙ ~ N(0, I)
        epsilons = [torch.randn_like(theta) for _ in range(n)]
        
        # Compute returns Fᵢ = F(θₜ + σεᵢ) for i = 1, ..., n
        returns = []
        for i in range(n):
            # Set perturbed parameters: θₜ + σεᵢ
            perturbed_params = theta + sigma * epsilons[i]
            policy.set_params(perturbed_params)
            
            # Evaluate the perturbed policy
            state = mdp.reset()
            episode_reward = 0.0
            
            for step in range(episode_steps):
                # Get action from expert for all joints except ankle
                expert_action = agent.draw_action(state)
                
                # Get action from ES policy for ankle
                right_ankle_substate = get_right_ankle_substate(state)
                model_action = policy.select_action(right_ankle_substate)
                
                # Replace expert action with model action for ankle
                expert_action[7] = model_action[0]
                
                # Take action in environment
                next_state, reward, done, _ = mdp.step(expert_action)
                
                if done:
                    reward = step + 1
                else:
                    reward = 0
                episode_reward += reward
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            returns.append(episode_reward)
            #print(f"Epoch {t + 1}, Policy {i + 1}/{n}: Return = {episode_reward}")
        max_index = returns.index(max(returns))
        max_reward = max(returns)
        #print max return
        print(f"Epoch {t + 1}, Return = {max(returns)}")

        if max_reward > best_max_reward:
            theta = theta + sigma * epsilons[max_index]
            policy.set_params(theta)
            best_max_reward = max_reward
            best_theta = theta
            epochs_no_improve = 0
            print(f"New best reward: {best_max_reward}")
            # Save the model weights
            model_save_path = os.path.join(os.path.dirname(__file__), "es_actor.pth")
            torch.save(policy.actor.state_dict(), model_save_path)
            print(f"Actor model weights saved to {model_save_path}")
        else:
            epochs_no_improve += 1


        if epochs_no_improve > 15:
            if sigma*2 > initial_sigma:
                sigma = initial_sigma
            else:
                sigma = sigma * 2
            epochs_no_improve = 0
            print(f"Sigma increased to {sigma}")
        

        # Set θₜ₊₁ ← θₜ + α(1/nσ) ∑ᵢ₌₁ⁿ Fᵢεᵢ
        returns_tensor = torch.tensor(returns, device=device)
        
        # Record the mean return for this epoch
        mean_return = returns_tensor.mean().item()
        epoch_rewards.append(mean_return)
        
        # Adaptive sigma: decrease sigma when rewards improve
        if max_reward > best_max_reward_sigma + improvement_threshold:
            best_max_reward_sigma = max_reward
            improvement_threshold = improvement_threshold/2
            # Decrease sigma but don't go below minimum
            sigma = max(sigma * sigma_decay_factor, min_sigma)
            print(f"Epoch {t + 1}: Reward improved to {max_reward}. Decreasing sigma to {sigma:.6f}")
        
        sigma_values.append(sigma)
        print(f"Epoch {t + 1} completed with mean return: {mean_return}, sigma: {sigma:.6f}")
    
    # Plot the rewards after each epoch
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot mean return
    ax1.plot(range(1, num_epochs + 1), epoch_rewards)
    ax1.set_ylabel('Mean Return')
    ax1.set_title('Mean Return After Each Epoch')
    
    # Plot sigma values
    ax2.plot(range(1, num_epochs + 1), sigma_values)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Sigma')
    ax2.set_title('Sigma Value Over Time')
    
    plt.tight_layout()
    plt.show()

    # Save the model weights
    policy.set_params(best_theta)
    model_save_path = os.path.join(os.path.dirname(__file__), "es_actor.pth")
    torch.save(policy.actor.state_dict(), model_save_path)
    print(f"Actor model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()
