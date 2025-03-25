import matplotlib
matplotlib.use("TkAgg")  # Use TkAgg backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from loco_mujoco import LocoEnv
from mushroom_rl.core import Agent
from ModelsAndUtils import MLP, get_right_ankle_substate
import os

sns.set(style="whitegrid")

# Initialize the environment
mdp = LocoEnv.make("HumanoidTorque.walk.perfect", use_box_feet=True)

# Load expert agent
agent_file_path = os.path.join(os.path.dirname(__file__), "perfect_88_original.msh")
agent = Agent.load(agent_file_path)

# Load MLP model
model = MLP(22, 128, 1)
model_load_path = os.path.join(os.path.dirname(__file__), "mlp_state_22_hidden_128_perfect_88.pth")
model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
model.eval()

state = mdp.reset()
done = False

vail_agent_actions = []
mlp_agent_actions = []
time_steps = []

step = 0
while not done and step < 1000:
    # Extract substate for prediction
    right_ankle_substate = get_right_ankle_substate(state)
    substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Get MLP agent action
    mlp_action = model(substate_tensor).squeeze().item()

    # Get expert agent action
    expert_action = agent.draw_action(state)[7]

    # Record actions
    mlp_agent_actions.append(mlp_action * 500)
    vail_agent_actions.append(expert_action * 500)
    time_steps.append(step)

    # Step environment (using MLP action for ankle)
    action = agent.draw_action(state)
    action[7] = mlp_action
    state, _, done, _ = mdp.step(action)

    step += 1

# Calculate Mean Absolute Error
mae = np.abs(np.array(vail_agent_actions) - np.array(mlp_agent_actions))
mean_mae = np.mean(mae)

# Plotting using seaborn
fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

sns.lineplot(ax=axes[0], x=time_steps, y=vail_agent_actions, label='Expert Agent Action', linewidth=2)
sns.lineplot(ax=axes[0], x=time_steps, y=mlp_agent_actions, label='MLP Agent Action', linewidth=2, linestyle='--')
axes[0].set_ylabel('Action Value (Nm)', fontsize=16)
axes[0].set_title('Comparison of Expert and MLP (12 States) Agent Actions Over First 1000 Steps', fontsize=18)
axes[0].legend(fontsize=14)

sns.lineplot(ax=axes[1], x=time_steps, y=mae, color='red', linewidth=2)
axes[1].axhline(y=mean_mae, color='blue', linestyle='--', linewidth=2, label=f'Mean MAE = {mean_mae:.2f}')
axes[1].set_xlabel('Steps', fontsize=16)
axes[1].set_ylabel('Mean Absolute Error (Nm)', fontsize=16)
axes[1].set_title('Mean Absolute Error between Expert and MLP Actions', fontsize=18)
axes[1].legend(fontsize=14)

plt.tight_layout()
plt.show()

