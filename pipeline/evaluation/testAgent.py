import os
import torch
from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv

# Load the expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_100_prosthesis_inertia.msh")
agent = Agent.load(agent_file_path)

# Initialize the humanoid environment
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True)

# Number of episodes to run
num_episodes = 10
total_steps = 0
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
    total_steps += step
    print(f"Episode {episode + 1} completed")

print(f"Average steps per episode: {total_steps / num_episodes}")
