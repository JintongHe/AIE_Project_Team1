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
from imitation_learning.utils import get_agent
import math
import copy
from pipeline.MLP.ModelsAndUtils import ankle_training_reward, get_right_ankle_substate, get_action_substate, TD3, ReplayBuffer




def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True, reward_type="custom", reward_params=dict(reward_callback=ankle_training_reward))

    print("Observation Space:", mdp.info.observation_space.low)
    print("Observation Space Shape:", mdp.info.observation_space.shape)

    print("Observation Variables:")
    for obs in mdp.get_all_observation_keys():
        print(obs)
    print("Action Space:", mdp.info.action_space)
    print("Action Space Shape:", mdp.info.action_space.shape)

    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device = torch.device("mps" if use_mps else "cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")


   

    # Initialize the model
    input_dim = 22  # Number of features in the substate
    output_dim = 1  # Number of actions
    max_action = 100.0
    model = TD3(input_dim, output_dim, max_action)
    model.actor.to(device)
    model.actor_target.to(device)
    model.critic.to(device)
    model.critic_target.to(device)

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "real_180.msh")
    agent = Agent.load(agent_file_path)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(input_dim, output_dim)

    # Initialize the optimizer
    epoch_rewards = []
    num_epochs = 500
    batch_size = 50
    best_epoch_reward = -math.inf


    # Run evaluation for 1000 episodes
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        step = 0
        batch_loss = 0.0
        epoch_reward = 0.0
        total_reward = 0.0

        for _ in range(batch_size):
            # Get action from expert
            action = agent.draw_action(state)
            right_ankle_action = get_action_substate(action)
            # Get action from ankle model
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            model_action = model.select_action(right_ankle_substate_tensor).squeeze()

            # Calculate loss imitating expert action
            model_action_tensor = torch.tensor(model_action, dtype=torch.float32).to(device)
            right_ankle_action_tensor = torch.tensor(right_ankle_action, dtype=torch.float32).to(device)
            imitation_loss = F.mse_loss(model_action_tensor, right_ankle_action_tensor)

            # Replace expert action with model action
            #action[7] = right_ankle_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)
            

            #calculate reward if done
            # if done:
            #     total_reward = -5 - imitation_loss.item()
            # else:

            #     total_reward = -imitation_loss.item()
            total_reward = -imitation_loss.item()
            epoch_reward += total_reward

            # Store transition in replay buffer
            replay_buffer.add(right_ankle_substate, model_action, get_right_ankle_substate(next_state), total_reward, done)

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

            # Train TD3 model
            if replay_buffer.size > batch_size:
                model.train(replay_buffer, batch_size)

        epoch_rewards.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed with average reward: {epoch_reward/step} and total steps: {step}")
        if epoch > 100:
            if epoch_reward > best_epoch_reward:
                best_epoch_reward = epoch_reward
                # Save the model weights
                model_save_path = os.path.join(os.path.dirname(__file__), "td3_actor.pth")
                torch.save(model.actor.state_dict(), model_save_path)
                print(f"Actor model weights saved to {model_save_path}")

                critic_save_path = os.path.join(os.path.dirname(__file__), "td3_critic.pth")
                torch.save(model.critic.state_dict(), critic_save_path)
                print(f"Critic model weights saved to {critic_save_path}")

    # Plot the rewards after each epoch
    plt.plot(range(1, num_epochs + 1), epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward After Each Epoch')
    plt.show()

if __name__ == '__main__':
    main()