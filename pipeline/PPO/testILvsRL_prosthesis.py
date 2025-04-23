from mushroom_rl.core import Agent
import matplotlib.pyplot as plt
import os
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from ModelsAndUtils import MLP, get_ankle_substate
from torch.distributions import Normal
import numpy as np
import scipy.signal
from RL import PolicyNet

def get_model_number(model_name):
    return int(model_name[4:6])
def get_model_number_IL(model_name):
    return int(model_name[10:12])

def get_average_steps(policy, env, agent, num_episodes=10, max_steps=2000, device="mps", state_dim=36):
    total_reward = 0.0
    for episode in range(num_episodes):
        state = env.reset()
        ankle_state = get_ankle_substate(state, state_dim )
        episode_reward = 0.0
        for step in range(max_steps):
            with torch.no_grad():
                ankle_state_tensor = torch.as_tensor(ankle_state, dtype=torch.float32).to(device)
                action = policy.get_best_action(ankle_state_tensor)
            body_action = agent.draw_action(state)
            body_action[7] = action.item()
            next_state, reward, done, _ = env.step(body_action)
            state = next_state
            ankle_state = get_ankle_substate(state, state_dim)
            episode_reward += reward
            if done:
                break
        total_reward += episode_reward
    average_reward = total_reward / num_episodes
    return average_reward

def main():
    device = 'cpu'
    env_id = "HumanoidTorque.walk.perfect"
    env = LocoEnv.make(env_id, use_box_feet=True)
    agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_140_prosthesis_inertia.msh")
    agent = Agent.load(agent_file_path)

    models_RL = [
        "new_16_states_survival_pros.pth",
        "new_22_states_survival_pros.pth",
        "new_36_states_survival_pros.pth",
    ]
    models_IL = [
        "mlp_state_16_hidden_128_prosthesis.pth",
        "mlp_state_22_hidden_128_prosthesis.pth",
        "mlp_state_36_hidden_128_prosthesis.pth",
    ]

    # RL results
    average_rewards_RL = []
    state_dims_RL = []
    for model in models_RL:
        state_dim = get_model_number(model)
        action_dim = 1
        policy = PolicyNet(state_dim, action_dim).to(device)
        policy_load_path = os.path.join(os.path.dirname(__file__), model)
        policy.load_state_dict(torch.load(policy_load_path))
        policy.eval()
        avg_reward = get_average_steps(policy, env, agent, num_episodes=10, max_steps=2000, device=device, state_dim=state_dim)
        average_rewards_RL.append(avg_reward)
        state_dims_RL.append(state_dim)
        print(f"RL Model: {model}, Average Reward: {avg_reward:.2f}")

    # IL results
    average_rewards_IL = []
    state_dims_IL = []
    for model in models_IL:
        state_dim = get_model_number_IL(model)
        action_dim = 1
        hidden_dim = 128
        policy = MLP(state_dim, hidden_dim, action_dim).to(device)
        policy_load_path = os.path.join(os.path.dirname(__file__), model)
        policy.load_state_dict(torch.load(policy_load_path, map_location=device))
        policy.eval()
        avg_reward = get_average_steps(policy, env, agent, num_episodes=10, max_steps=2000, device=device, state_dim=state_dim)
        average_rewards_IL.append(avg_reward)
        state_dims_IL.append(state_dim)
        print(f"IL Model: {model}, Average Reward: {avg_reward:.2f}")

    # Plot both
    plt.figure(figsize=(10, 5))
    plt.plot(state_dims_RL, average_rewards_RL, 'o-', color='skyblue', linewidth=2, markersize=8, label='RL')
    plt.plot(state_dims_IL, average_rewards_IL, 's-', color='orange', linewidth=2, markersize=8, label='IL')
    plt.xlabel('Number of States')
    plt.ylabel('Average Reward')
    plt.title('Average Reward vs Number of States (RL vs IL)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(sorted(set(state_dims_RL + state_dims_IL)))
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
