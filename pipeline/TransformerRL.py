from mushroom_rl.core import Core, Agent
import matplotlib.pyplot as plt
import os
import numpy as np
from loco_mujoco import LocoEnv
import torch
import torch.nn as nn
import torch.nn.functional as F
from imitation_learning.utils import get_agent
import math

#Extract ankle features from state
def get_right_ankle_substate(state):
    # Indices of right ankle related features in the observation space
    #right_ankle_indices = [
    #    10,  # q_ankle_angle_r
    #    29   # dq_ankle_angle_r
    #]
    
    #right_ankle_substate = state[right_ankle_indices]
    
    #return right_ankle_substate
    return state

# Define the reward function based on the number of steps in a trajectory
def step_reward(state, action, next_state):
    return 1  # Reward is 1 for each step taken

#Temporary reward function
def ankle_training_reward(state, action, next_state):
    """Composite reward function for ankle coordination training"""
    # Extract relevant state information
    q_pelvis_tx, q_pelvis_tz, q_pelvis_ty = state[0], state[1], state[2]
    q_pelvis_tilt, q_pelvis_list = state[3], state[4]
    q_ankle_angle_r = state[10]  # Right ankle angle
    dq_pelvis_tx = state[18]     # Forward velocity
    dq_ankle_angle_r = state[29]  # Ankle angular velocity
    ankle_torque = action[7]      # Ankle control signal

    # 1. Forward Motion (30%)
    target_velocity = 1.25  # Standard walking speed from env defaults
    velocity_reward = 0.3 * np.exp(-2.0 * np.square(dq_pelvis_tx - target_velocity))

    # 2. Stability (30%)
    tilt_penalty = np.square(q_pelvis_tilt)
    list_penalty = np.square(q_pelvis_list)
    stability_reward = 0.3 * np.exp(-5.0 * (tilt_penalty + list_penalty))

    # 3. Ankle Behavior (30%)
    # Penalize extreme angles and rapid movements
    angle_penalty = np.square(q_ankle_angle_r)
    velocity_penalty = 0.1 * np.square(dq_ankle_angle_r)
    torque_penalty = 0.05 * np.square(ankle_torque)
    ankle_reward = 0.3 * np.exp(-2.0 * (angle_penalty + velocity_penalty + torque_penalty))

    # 4. Survival Bonus (10%)
    survival_bonus = 0.1 if (np.abs(q_pelvis_tilt) < 0.3 and 
                            np.abs(q_pelvis_list) < 0.3 and 
                            q_pelvis_ty > 0.7) else 0.0

    return velocity_reward + stability_reward + ankle_reward + survival_bonus

# Model for predicting right ankle action
#Model for predicting right ankle action
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_dim, input_dim)
        self.decoder = nn.Linear(input_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output[-1]

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def main():
    # Initialize the humanoid environment
    env_id = "HumanoidTorque.walk.perfect"
    mdp = LocoEnv.make(env_id, use_box_feet=True, reward_type="custom", reward_params=dict(reward_callback=ankle_training_reward))

    # Load the expert agent
    agent_file_path = os.path.join(os.path.dirname(__file__), "best_real_agent_141.msh")
    agent = Agent.load(agent_file_path)

    # Load the model
    input_dim = 36  # Number of features in the substate
    output_dim = 1  # Number of actions
    model = TransformerModel(input_dim, output_dim)
    model_load_path = os.path.join(os.path.dirname(__file__), "real_141_best.pth")
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    print(f"Model weights loaded from {model_load_path}")

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    epoch_rewards = []
    num_epochs = 1000
    batch_size = 400
    upper_batch_size = 10

    # Perform reinforcement learning
    for epoch in range(num_epochs):
        state = mdp.reset()  # Reset the environment for each episode
        right_ankle_substate = get_right_ankle_substate(state)
        done = False
        step = 0
        epoch_reward = 0.0
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        for _ in range(upper_batch_size):
            while not done:
                # Get action from expert
                action = agent.draw_action(state)
                # Get action from ankle model
                right_ankle_substate_tensor = torch.tensor(right_ankle_substate, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                model_action = model(right_ankle_substate_tensor).squeeze()

                # Replace expert action with model action
                right_ankle_action = model_action.item()
                action[7] = right_ankle_action

                # Take action in environment
                next_state, reward, done, _ = mdp.step(action)
                epoch_reward += reward

                # Store experience in batch
                batch_states.append(state)
                batch_actions.append(action)
                batch_rewards.append(reward)
                batch_next_states.append(next_state)

                # Update state
                state = next_state
                right_ankle_substate = get_right_ankle_substate(state)
                step += 1
                # Perform optimization step
                optimizer.zero_grad()
                loss = -torch.tensor(epoch_reward, dtype=torch.float32)  # Negative reward as loss
                loss.backward()
                optimizer.step()

        epoch_rewards.append(epoch_reward)
        print(f"Epoch {epoch + 1} completed with total reward: {epoch_reward}")

    # Plot the rewards after each epoch
    plt.plot(range(1, num_epochs + 1), epoch_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Reward After Each Epoch')
    plt.show()

    # Save the model weights
    model_save_path = os.path.join(os.path.dirname(__file__), "finetuned_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path}")

if __name__ == '__main__':
    main()