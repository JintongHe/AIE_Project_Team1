import os
import torch
from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv

# Load the expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "agent_epoch_423_J_991.255877.msh")
agent = Agent.load(agent_file_path)

# Initialize the humanoid environment
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

# Number of episodes to run
num_episodes = 10

for episode in range(num_episodes):
    state = mdp.reset()  # Reset the environment for each episode
    done = False
    step = 0

    while not done:
        # Get action from expert
        action = agent.draw_action(state)

        # Take action in environment
        next_state, reward, done, _ = mdp.step(action)

        # Render the environment at every step
        mdp.render()

        # Update state
        state = next_state
        step += 1

    print(f"Episode {episode + 1} completed")

print("Interaction with the environment completed.")