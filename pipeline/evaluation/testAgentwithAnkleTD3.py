import os
import torch
import numpy as np
from loco_mujoco import LocoEnv
from TD3RL import TD3
from TD3IL import get_right_ankle_substate
from mushroom_rl.core import Core, Agent


def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.real"
    mdp = LocoEnv.make(env_id, use_box_feet=True)
    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "real_180.msh")
    agent = Agent.load(agent_file_path)


    # Initialize the model
    input_dim = 22  # Number of features in the substate
    output_dim = 1  # Number of actions
    max_action = 100.0
    model = TD3(input_dim, output_dim, max_action)

    # Load the model weights
    model_load_path = os.path.join(os.path.dirname(__file__), "td3_actor.pth")
    model.actor.load_state_dict(torch.load(model_load_path))
    print(f"Actor model weights loaded from {model_load_path}")

    critic_load_path = os.path.join(os.path.dirname(__file__), "td3_critic.pth")
    model.critic.load_state_dict(torch.load(critic_load_path))
    print(f"Critic model weights loaded from {critic_load_path}")

    # Run evaluation for a specified number of episodes
    num_episodes = 10
    for episode in range(num_episodes):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        episode_reward = 0.0
        step = 0

        while not done:
            # Get action from ankle model
            right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            model_action = model.select_action(right_ankle_substate_tensor).squeeze()

            # Replace expert action with model action
            right_ankle_action = model_action.item()
            action = agent.draw_action(state)
            action[7] = right_ankle_action

            # Take action in environment
            next_state, reward, done, _ = mdp.step(action)

            # Render the environment
            mdp.render()

            # Update state
            state = next_state
            right_ankle_substate = get_right_ankle_substate(state)
            step += 1

        print(f"Episode {episode + 1} completed with total reward: {episode_reward} and total steps: {step}")

if __name__ == '__main__':
    main()