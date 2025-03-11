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
from TD3IL import TD3, Actor, Critic, ReplayBuffer, ankle_training_reward, get_right_ankle_substate, get_action_substate


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

    # Check if GPU is available
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    sw = None  # TensorBoard logging can be added later
    # agent = get_agent(env_id, mdp, use_cuda, sw, conf_path="imitation_learning/confs.yaml")

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "best_real_agent_141.msh")
    agent = Agent.load(agent_file_path)
    # core = Core(agent, mdp)
    # dataset = core.evaluate(n_episodes=1000, render=True)


    # Reset the environment

    # Initialize the model
    input_dim = 22  # Number of features in the substate
    output_dim = 1  # Number of actions
    max_action = 100.0
    model = TD3(input_dim, output_dim, max_action)

    # Initialize the replay buffer
    replay_buffer = ReplayBuffer(input_dim, output_dim)

    # Initialize the optimizer
    epoch_rewards = []
    num_epochs = 500
    batch_size = 1000


    # Run evaluation for 1000 episodes
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        step = 0
        epoch_reward = 0.0

        for step in range(batch_size):
            # Get action from expert
            action = agent.draw_action(state)
            # Get action from ankle model
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            model_action = model.select_action(right_ankle_substate_tensor).squeeze()

            # Replace expert action with model action
            right_ankle_action = model_action.item()
            action[7] = right_ankle_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)
            
            if done:
                reward1 = -1000
            else:
                reward1 = 1
            epoch_reward += reward1
            # print(reward)

            # Store transition in replay buffer
            replay_buffer.add(right_ankle_substate, model_action, get_right_ankle_substate(next_state), reward1, done)

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

            # Train TD3 model
            if replay_buffer.size > batch_size:
                model.train(replay_buffer, batch_size)

        epoch_rewards.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed with total reward: {epoch_reward} and total steps: {step}")

        

    # Plot the rewards after each epoch
    plt.plot(range(1, num_epochs + 1), epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward After Each Epoch')
    plt.show()

    # Save the model weights
    model_save_path = os.path.join(os.path.dirname(__file__), "td3_actor.pth")
    torch.save(model.actor.state_dict(), model_save_path)
    print(f"Actor model weights saved to {model_save_path}")

    critic_save_path = os.path.join(os.path.dirname(__file__), "td3_critic.pth")
    torch.save(model.critic.state_dict(), critic_save_path)
    print(f"Critic model weights saved to {critic_save_path}")

if __name__ == '__main__':
    main()