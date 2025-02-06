from mushroom_rl.core import Core, Agent
from loco_mujoco import LocoEnv


env = LocoEnv.make("HumanoidTorque.walk")

agent = Agent.load("./Imitaion_Learning_Model/agent_epoch_223_J_885.160274.msh")

core = Core(agent, env)

core.evaluate(n_episodes=10, render=True)