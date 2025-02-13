from mushroom_rl.core import Core, Agent
# from loco_mujoco import LocoEnv
#
#
# env = LocoEnv.make("HumanoidTorque.walk.perfect")
#
# agent = Agent.load("loco-mujoco/logs/loco_mujoco_evalution_2025-02-09_21-29-47/env_id___HumanoidTorque.walk.perfect/0/agent_epoch_49_J_959.235794.msh")
#
# core = Core(agent, env)
#
# core.evaluate(n_episodes=10, render=True)
#
# env.play_trajectory_from_velocity(n_steps_per_episode=500)
from loco_mujoco import LocoEnv
from utils import get_agent
import torch

# Initialize the humanoid environment without left ankle movement
env_id = "HumanoidTorque.walk.perfect"
mdp = LocoEnv.make(env_id, use_box_feet=True, disable_ankle_l=True)
ankle_idx = mdp.get_obs_idx("q_pelvis_tx")
print("Index of Left Ankle Joint:", ankle_idx)

print("Observation Space:", mdp.info.observation_space.low)
print("Observation Space Shape:", mdp.info.observation_space.shape)

print("Observation Variables:")
for obs in mdp.get_all_observation_keys():
    print(obs)
print("Action Space:", mdp.info.action_space)
print("Action Space Shape:", mdp.info.action_space.shape)

# Check if GPU is available
use_cuda = torch.cuda.is_available()
sw = None  # TensorBoard logging can be added later
# agent = get_agent(env_id, mdp, use_cuda, sw)
agent = Agent.load("C:/test/24782/project_folder/logs/0/agent_epoch_498_J_16.744988.msh")
# core = Core(agent, mdp)
# dataset = core.evaluate(n_episodes=1000, render=True)
# Reset the environment
state = mdp.reset()

# Run evaluation for 10 episodes
for episode in range(10):
    state = mdp.reset()  # Reset the environment for each episode
    done = False
    step = 0

    while not done:
        # Get action from agent
        action = agent.draw_action(state)

        # Print state and action values
        print(f"Episode {episode + 1}, Step {step + 1}")
        print(f"State: {state}")
        print(f"Action: {action}")

        # Take action in environment
        next_state, reward, done, _ = mdp.step(action)

        # 🔹 Render the environment at every step
        mdp.render()

        # Update state
        state = next_state
        step += 1




